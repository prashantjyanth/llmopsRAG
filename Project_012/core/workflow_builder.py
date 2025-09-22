
"""Workflow construction and execution utilities.

`WorkflowBuilder` orchestrates multiple `ConfigReactAgent` instances under a
LangGraph supervisor. It loads configuration, compiles a workflow app, then
provides a `run` method that executes a single user input while logging
telemetry to MLflow and Prometheus. Structured `PROGRESS` log lines are
emitted for easy scraping / timeline reconstruction.

Environment variables:
    MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, GROQ_API_KEY
"""

import yaml, os, sys, time, logging
import mlflow
import mlflow.langchain   # ✅ MLflow autologging for LangChain
from prometheus_client import Counter, Histogram, Gauge, start_http_server
start_http_server(8000)

from core.agent_builder import ConfigReactAgent
from langgraph_supervisor import create_supervisor  # type: ignore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_groq import ChatGroq
import json


# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | workflow | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
MODEL_NAME_MAP = {
    "openai_gpt-oss-20b": "openai/gpt-oss-20b",
    "openai_gpt-oss-120b": "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant"
}
logger = logging.getLogger("workflow")
import threading
import psutil
# --- Resource Usage Metrics ---
CPU_USAGE = Gauge("workflow_cpu_usage_percent", "Workflow process CPU usage percent")
RAM_USAGE = Gauge("workflow_ram_usage_mb", "Workflow process RAM usage in MB")

def log_resource_usage(interval=30):
    process = psutil.Process(os.getpid())
    while True:
        cpu = process.cpu_percent(interval=1)
        mem = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"[Resource] CPU: {cpu:.2f}% | RAM: {mem:.2f} MB")
        CPU_USAGE.set(cpu)
        RAM_USAGE.set(mem)
        time.sleep(max(1, interval-1))

# Start resource logging in a background thread
threading.Thread(target=log_resource_usage, args=(30,), daemon=True).start()

# --- MLflow Setup ---
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "genai_ops_demo")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
logger.info(f"📡 MLflow tracking set to {TRACKING_URI} | experiment={EXPERIMENT}")

# ✅ enable autologging for LangChain workflows
mlflow.langchain.autolog()


# --- Prometheus Metrics ---
WORKFLOW_RUNS = Counter("workflow_runs_total", "Total workflow runs executed")
WORKFLOW_ERRORS = Counter("workflow_errors_total", "Total workflow run errors")
WORKFLOW_LATENCY = Histogram("workflow_latency_seconds", "Latency of workflow runs in seconds")
WORKFLOW_ACTIVE_THREADS = Gauge("workflow_active_threads", "Active workflow threads")

# --- Token Usage Metrics ---
WORKFLOW_PROMPT_TOKENS = Histogram("workflow_prompt_tokens", "Prompt tokens per workflow run")
WORKFLOW_COMPLETION_TOKENS = Histogram("workflow_completion_tokens", "Completion tokens per workflow run")
WORKFLOW_TOTAL_TOKENS = Histogram("workflow_total_tokens", "Total tokens per workflow run")


# --- Safety Wrapper for Counters ---
def safe_inc(counter, step=1):
    """Increment Prometheus counter with integer-only step.
    If float is passed, coerce to int and log warning.
    """
    if not isinstance(step, int):
        logger.warning(f"[METRICS] ⚠️ Non-integer increment {step} passed to {counter._name}, coercing to int")
        step = int(step)
    counter.inc(step)


# Seed initial values
WORKFLOW_PROMPT_TOKENS.observe(0.0)
WORKFLOW_COMPLETION_TOKENS.observe(0.0)
WORKFLOW_TOTAL_TOKENS.observe(0.0)
safe_inc(WORKFLOW_RUNS, 0)
safe_inc(WORKFLOW_ERRORS, 0)
WORKFLOW_LATENCY.observe(0.0)
WORKFLOW_ACTIVE_THREADS.set(0)


def extract_usage_from_messages(messages):
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

    for msg in messages:
        # Some messages carry usage in response_metadata
        if hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
            usage = msg.response_metadata.get("token_usage", None)
            if usage and isinstance(usage, dict):
                prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                total_tokens += int(usage.get("total_tokens", 0) or 0)

        # Some carry usage in usage_metadata
        if hasattr(msg, "usage_metadata") and isinstance(msg.usage_metadata, dict):
            usage = msg.usage_metadata
            prompt_tokens += int(usage.get("input_tokens", 0) or 0)
            completion_tokens += int(usage.get("output_tokens", 0) or 0)
            total_tokens += int(usage.get("total_tokens", 0) or 0)

    return prompt_tokens, completion_tokens, total_tokens


# --- Agent Metrics ---
AGENT_RUNS = Counter("agent_runs_total", "Total agent runs", ["agent_name"])
AGENT_ERRORS = Counter("agent_errors_total", "Total agent run errors", ["agent_name"])
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent latency (seconds)", ["agent_name"])
AGENT_TOKENS_USED = Histogram("agent_tokens_used", "Tokens used per agent run", ["agent_name"])


def _safe_get(obj, key, default=None):
    """Safely get key from dict or attribute from object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def process_workflow_metrics(workflow_response: dict):
    """
    Process workflow response and update Prometheus metrics.
    Works with HumanMessage, AIMessage, ToolMessage objects or dicts.
    """
    for msg in workflow_response.get("messages", []):
        name = _safe_get(msg, "name")
        meta = _safe_get(msg, "response_metadata", {}) or {}
        usage = _safe_get(msg, "usage_metadata", {}) or {}

        if not name:
            continue  # skip system/tool-only messages

        # --- Extract tokens ---
        input_tokens, output_tokens, total_tokens = 0, 0, 0
        if "token_usage" in meta:
            u = meta["token_usage"]
            input_tokens = int(u.get("prompt_tokens", 0) or 0)
            output_tokens = int(u.get("completion_tokens", 0) or 0)
            total_tokens = int(u.get("total_tokens", 0) or 0)
        else:
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total_tokens = int(usage.get("total_tokens", 0) or 0)

        # --- Extract latency ---
        latency = meta.get("total_time")
        if latency is not None:
            latency = float(latency)

        # --- Update Prometheus metrics ---
        AGENT_RUNS.labels(agent_name=name).inc()
        if latency is not None:
            AGENT_LATENCY.labels(agent_name=name).observe(latency)
        if total_tokens > 0:
            AGENT_TOKENS_USED.labels(agent_name=name).observe(total_tokens)

    return True


class WorkflowBuilder:
    """Build and execute a multi-agent supervisor workflow."""

    def __init__(self, checkpointer=None, config_path="configs/workflow.yaml"):
        self.checkpointer = checkpointer
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
        logger.info(f"Supervisor MLFLOW prompt name: {self.prompt_name}" if self.prompt_name else "No MLFLOW prompt name configured")
        self.prompt = self._load_prompt(self.prompt_name)
        logger.info(f"Supervisor MLFLOW prompt: {self.prompt[:100]}..." if self.prompt!="" else "No prompt loaded")
        if not self.prompt:
            self.prompt = self.workflowconfigs.get(
                "default_prompt",
                "You are a helpful assistant. Use flight agent for flight queries and hotel agent for hotel queries."
            )

        self.store = InMemoryStore()
        self.app = None

        logger.info("Building workflow...")
        self.build()
        self.compile()
        logger.info("WorkflowBuilder initialized successfully")

    def _load_prompt(self, prompt_name: str) -> str:
        """Load supervisor prompt from MLflow registry (best effort)."""
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
        """Instantiate LLM model for the supervisor layer."""
        self.model = ChatGroq(
            model=self.model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            verbose=True
        )
        logger.info(f"Model {self.model_name} initialized")

    def load_configs(self, path=None):
        """Load YAML config file and return parsed dict."""
        if path is None:
            raise ValueError("Config path must not be None")
        with open(path, "r") as f:
            configs = yaml.safe_load(f)
        logger.info(f"Loaded config file: {path}")
        return configs
    def load_governance_json(self, agent_name):
        """Load agent's evaluation JSON and return the best model name for the agent."""
        eval_dir = os.path.abspath('/app/eval_results')
        json_path = os.path.join(eval_dir, f"{agent_name}.json")
        if not os.path.exists(json_path):
            logger.warning(f"No evaluation JSON found for agent: {agent_name}")
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data.get("best_model")
        logger.info(f"Loaded best model for agent '{agent_name}': {model}")
        return model

    def load_agents(self):
        """Instantiate all agents declared in the agents config section."""

        self.agents = {k: ConfigReactAgent(self.checkpointer, k, v, MODEL_NAME_MAP.get(self.load_governance_json(k))) for k, v in self.agent_configs["agents"].items()}
        logger.info(f"Loaded {len(self.agents)} agents: {list(self.agents.keys())}")

    def build(self):
        """Create the supervisor graph (not yet compiled)."""
        self.workflow = create_supervisor(
            model=self.model,
            agents=[v.agent for v in self.agents.values()],
            output_mode=self.output,
            prompt=self.prompt,
        )
        logger.info("Workflow supervisor created")

    def compile(self):
        """Compile the workflow into an executable application object."""
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
        """Execute a single workflow step."""
        if not self.app:
            raise ValueError("Workflow not built. Call build() first.")

        start_time = time.perf_counter()
        logger.info(f"🚀 Starting workflow run | thread_id={thread_id} | input='{user_input}'")
        logger.info(f"PROGRESS | thread_id={thread_id} | phase=start")

        config = {"configurable": {"thread_id": thread_id}} if self.checkpointer is not None else {}

        safe_inc(WORKFLOW_RUNS)        # ✅ always int
        WORKFLOW_ACTIVE_THREADS.inc()

        try:
            with mlflow.start_run(run_name="workflow_trace", nested=True):
                mlflow.log_param("thread_id", thread_id)
                mlflow.log_param("user_input", user_input)

                logger.info(f"PROGRESS | thread_id={thread_id} | phase=invoke")
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

                # Prometheus tokens
                WORKFLOW_PROMPT_TOKENS.observe(prompt_tokens)
                WORKFLOW_COMPLETION_TOKENS.observe(completion_tokens)
                WORKFLOW_TOTAL_TOKENS.observe(total_tokens)
                process_workflow_metrics(response)

                # MLflow metrics
                mlflow.log_metric("prompt_tokens", prompt_tokens)
                mlflow.log_metric("completion_tokens", completion_tokens)
                mlflow.log_metric("total_tokens", total_tokens)

                logger.info(
                    f"✅ Workflow run completed | latency={latency:.3f}s | prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens} | thread_id={thread_id}"
                )
                logger.info(f"PROGRESS | thread_id={thread_id} | phase=complete | latency={latency:.3f}s")
                return response

        except Exception as e:
            safe_inc(WORKFLOW_ERRORS)  # ✅ always int
            logger.error(f"❌ Workflow error | {e}")
            logger.info(f"PROGRESS | thread_id={thread_id} | phase=error | message={str(e)[:80]}")
            with mlflow.start_run(run_name="workflow_error", nested=True):
                mlflow.log_param("thread_id", thread_id)
                mlflow.log_param("user_input", user_input)
                mlflow.log_metric("error", 1)
            raise e

        finally:
            WORKFLOW_ACTIVE_THREADS.dec()
            logger.info(f"🛑 Workflow run ended | thread_id={thread_id}")
            logger.info(f"PROGRESS | thread_id={thread_id} | phase=end")


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
