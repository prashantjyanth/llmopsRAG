import yaml, os, sys, time, logging
import mlflow
from core.agent_builder import ConfigReactAgent
from langgraph_supervisor import create_supervisor  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_groq import ChatGroq
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | workflow | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("workflow")

# --- MLflow Setup ---
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "genai_ops_demo")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

# --- Prometheus Metrics ---
WORKFLOW_RUNS = Counter("workflow_runs_total", "Total workflow runs executed")
WORKFLOW_ERRORS = Counter("workflow_errors_total", "Total workflow run errors")
WORKFLOW_LATENCY = Histogram("workflow_latency_seconds", "Latency of workflow runs in seconds")
WORKFLOW_ACTIVE_THREADS = Gauge("workflow_active_threads", "Active workflow threads")

# --- Token Usage Metrics ---
WORKFLOW_PROMPT_TOKENS = Histogram("workflow_prompt_tokens", "Prompt tokens per workflow run")
WORKFLOW_COMPLETION_TOKENS = Histogram("workflow_completion_tokens", "Completion tokens per workflow run")
WORKFLOW_TOTAL_TOKENS = Histogram("workflow_total_tokens", "Total tokens per workflow run")

# Seed initial values
WORKFLOW_PROMPT_TOKENS.observe(0.0)
WORKFLOW_COMPLETION_TOKENS.observe(0.0)
WORKFLOW_TOTAL_TOKENS.observe(0.0)
WORKFLOW_RUNS.inc(0)
WORKFLOW_ERRORS.inc(0)
WORKFLOW_LATENCY.observe(0.0)
WORKFLOW_ACTIVE_THREADS.set(0)
def extract_usage_from_messages(messages):
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

    for msg in messages:
        # Some messages carry usage in response_metadata
        if hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
            usage = msg.response_metadata.get("token_usage", None)
            if usage and isinstance(usage, dict):
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

        # Some carry usage in usage_metadata
        if hasattr(msg, "usage_metadata") and isinstance(msg.usage_metadata, dict):
            usage = msg.usage_metadata
            prompt_tokens += usage.get("input_tokens", 0)
            completion_tokens += usage.get("output_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)

    return prompt_tokens, completion_tokens, total_tokens


class WorkflowBuilder:
    def __init__(self, checkpointer=None, config_path="configs/workflow.yaml"):
        self.checkpointer = checkpointer
        start_http_server(8000)
        logger.info("📊 Prometheus metrics server started at :8000/metrics")

        self.workflowconfigs = self.load_configs(config_path)["workflows"]
        logger.info(f"Loaded workflow config: {self.workflowconfigs}")

        self.agent_configs = self.load_configs(
            self.workflowconfigs.get("agents_config_path", "configs/agents.yaml")
        )
        self.model_name = self.workflowconfigs.get("model", "llama3-70b-8192")
        self.agents = {}

        logger.info("Loading agents...")
        self.load_agents()

        logger.info("Loading model...")
        self.load_model()

        self.output = self.workflowconfigs.get("output", "last_message")
        self.workflow = None
        self.prompt_name = self.workflowconfigs.get("prompt_name")
        self.prompt = self._load_prompt(self.prompt_name)
        logger.info(f"Supervisor MLFLOW prompt: {self.prompt[:100]}..." if self.prompt!="" else "No prompt loaded")
        if not self.prompt:
            self.prompt = self.workflowconfigs.get(
                "default_prompt",
                "You are a helpful assistant. Use flight agent for flight queries and hotel agent for hotel queries."
            )

        # self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.app = None

        logger.info("Building workflow...")
        self.build()
        self.compile()
        logger.info("WorkflowBuilder initialized successfully")


    def _load_prompt(self, prompt_name: str) -> str:
        if not prompt_name:
            return ""
        try:
            prompt_text = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production").template  # type: ignore
            logger.info(f"[SUPERVISOR] ✅ Loaded prompt from MLflow: {prompt_name}")
            return prompt_text
        except Exception as e:
            logger.warning(f"[SUPERVISOR] ⚠️ Could not load prompt '{prompt_name}' from MLflow: {e}")
            return ""

    def load_model(self):
        self.model = ChatGroq(
            model=self.model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            verbose=True
        )
        logger.info(f"Model {self.model_name} initialized")

    def load_configs(self, path=None):
        if path is None:
            raise ValueError("Config path must not be None")
        with open(path, "r") as f:
            configs = yaml.safe_load(f)
        logger.info(f"Loaded config file: {path}")
        return configs

    def load_agents(self):
        self.agents = {k: ConfigReactAgent(self.checkpointer, k, v) for k, v in self.agent_configs["agents"].items()}
        logger.info(f"Loaded {len(self.agents)} agents: {list(self.agents.keys())}")

    def build(self):
        self.workflow = create_supervisor(
            model=self.model,
            agents=[v.agent for v in self.agents.values()],
            output_mode=self.output,
            prompt=self.prompt,
        )
        logger.info("Workflow supervisor created")

    def compile(self):
        if self.checkpointer is not None:
            app = self.workflow.compile(
                checkpointer=self.checkpointer,
                store=self.store
            )
        else:
            app = self.workflow.compile()  # 👈 stateless
        self.app = app
        logger.info("Workflow compiled successfully")


    def run(self, user_input, thread_id="default"):
        if not self.app:
            raise ValueError("Workflow not built. Call build() first.")

        if self.checkpointer is not None:
            config = {"configurable": {"thread_id": thread_id}}
        else:
            config = {}  # 👈 no checkpoints required
        WORKFLOW_RUNS.inc()
        WORKFLOW_ACTIVE_THREADS.inc()

        start_time = time.perf_counter()
        logger.info(f"🚀 Starting workflow run | thread_id={thread_id} | input='{user_input}'")

        try:
            with mlflow.start_run(run_name="workflow_trace", nested=True):
                mlflow.log_param("thread_id", thread_id)
                mlflow.log_param("user_input", user_input)

                response = self.app.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config
                )
                if isinstance(response, dict) and "messages" in response:
                    prompt_tokens, completion_tokens, total_tokens = extract_usage_from_messages(response["messages"])
                else:
                    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
                logger.info(f"Workflow response: {response}")
                logger.info(f"Token usage | prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
                latency = time.perf_counter() - start_time
                WORKFLOW_LATENCY.observe(latency)
                mlflow.log_metric("workflow_latency_sec", latency)
                mlflow.log_dict(response, "workflow_response.json")

                # Prometheus
                WORKFLOW_PROMPT_TOKENS.observe(prompt_tokens)
                WORKFLOW_COMPLETION_TOKENS.observe(completion_tokens)
                WORKFLOW_TOTAL_TOKENS.observe(total_tokens)

                # MLflow
                mlflow.log_metric("prompt_tokens", prompt_tokens)
                mlflow.log_metric("completion_tokens", completion_tokens)
                mlflow.log_metric("total_tokens", total_tokens)

                logger.info(
                    f"✅ Workflow run completed | latency={latency:.3f}s | "
                    f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens} | "
                    f"thread_id={thread_id}"
                )
                return response

        except Exception as e:
            WORKFLOW_ERRORS.inc()
            logger.error(f"❌ Workflow error | {e}")
            with mlflow.start_run(run_name="workflow_error", nested=True):
                mlflow.log_param("thread_id", thread_id)
                mlflow.log_param("user_input", user_input)
                mlflow.log_metric("error", 1)
            raise e

        finally:
            WORKFLOW_ACTIVE_THREADS.dec()
            logger.info(f"🛑 Workflow run ended | thread_id={thread_id}")


if __name__ == "__main__":
    builder = WorkflowBuilder()
    thread_id = "test_conversation"
    logger.info(f"🧪 Testing conversation memory (Thread: {thread_id})")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            logger.info("👋 Goodbye!")
            break
        response = builder.run(user_input, thread_id)
        print(f"Response: {response['messages']}")
