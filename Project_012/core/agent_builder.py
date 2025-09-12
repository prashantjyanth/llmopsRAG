import importlib, os, time, logging, traceback
import mlflow
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from prometheus_client import Counter, Histogram
import dotenv
dotenv.load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | agent | %(message)s",
)
logger = logging.getLogger("agent")

# --- MLflow Setup ---
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "genai_ops_demo")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
mlflow.langchain.autolog()

# --- Prometheus Metrics ---
AGENT_RUNS = Counter("agent_runs_total", "Total agent runs", ["agent_name"])
AGENT_ERRORS = Counter("agent_errors_total", "Total agent run errors", ["agent_name"])
AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent latency (seconds)", ["agent_name"])
AGENT_TOKENS_USED = Histogram("agent_tokens_used", "Tokens used per agent run", ["agent_name"])

# Seed metrics so they show up in /metrics
# for agent in ["flight_agent", "hotel_agent"]:
#     AGENT_RUNS.labels(agent).inc(0)
#     AGENT_ERRORS.labels(agent).inc(0)
#     AGENT_LATENCY.labels(agent).observe(0.0)
#     AGENT_TOKENS_USED.labels(agent).observe(0.0)


class ConfigReactAgent:
    def __init__(self, checkpoints, name: str, cfg: dict, model: str = "openai/gpt-oss-120b"):
        self.name = name
        self.checkpoints = checkpoints
        self.cfg = cfg
        self.model = model
        logger.info(f"🟢 Initializing agent: {name}")

        # --- Load Tools ---
        self.tools = self._load_tools(cfg.get("tools", []))
        logger.info(f"[{self.name}] Loaded tools: {self.tools}")

        # --- Initialize LLM ---
        self.llm = ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"))
        if not self.llm:
            raise ValueError(f"Could not initialize LLM with model {model}")
        logger.info(f"[{self.name}] Model initialized: {self.model}")

        # --- Load Prompt ---
        self.prompt_name = cfg.get("prompt_name")
        self.prompt_text = self._load_prompt(self.prompt_name)
        if not self.prompt_text:
            self.prompt_text = cfg.get("default_prompt", f"You are {self.name}, a helpful assistant.")
        logger.info(f"[{self.name}] Loaded prompt: {self.prompt_text[:80]}...")

        # --- Memory + Agent ---
        self.memory = MemorySaver()
        if checkpoints:
            self.agent = create_react_agent(
                name=self.name,
                model=self.llm,
                tools=self.tools,
                prompt=self.prompt_text,
                checkpointer=self.memory,
            )
        else:
            self.agent = create_react_agent(
                name=self.name,
                model=self.llm,
                tools=self.tools,
                prompt=self.prompt_text,
            )
        logger.info(f"[{self.name}] Agent created successfully")

    def build_agent(self):
        self.memory = MemorySaver()
        if self.checkpoints:
            self.agent = create_react_agent(
                name=self.name,
                model=self.llm,
                tools=self.tools,
                prompt=self.prompt_text,
                checkpointer=self.memory,
            )
        else:
            self.agent = create_react_agent(
                name=self.name,
                model=self.llm,
                tools=self.tools,
                prompt=self.prompt_text,
            )
        
        logger.info(f"[{self.name}] Agent built successfully")

    def _load_tools(self, tool_paths: list):
        tools = []
        for path in tool_paths:
            try:
                module_name, func_name = path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                tool_fn = getattr(module, func_name)
                tools.append(tool_fn)
                logger.info(f"[{self.name}] ✅ Loaded tool: {path}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Failed to load tool {path}: {e}")
        return tools

    def _load_prompt(self, prompt_name: str) -> str:
        if not prompt_name:
            return ""
        try:
            prompt_text = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production").template  # type: ignore
            logger.info(f"[{self.name}] ✅ Loaded prompt from MLflow: {prompt_name}")
            return prompt_text
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Could not load prompt '{prompt_name}' from MLflow: {e}")
            return ""

    def run(self, query: str, retries: int = 3, delay: float = 2.0, backoff: float = 2.0) -> str:
        logger.info(f"[{self.name}] 🚀 Starting run | query='{query}'")
        AGENT_RUNS.labels(agent_name=self.name).inc()

        attempt = 1
        start_time = time.perf_counter()

        while attempt <= retries:
            try:
                with mlflow.start_run(run_name=f"{self.name}_run_attempt{attempt}", nested=True):
                    logger.info(f"[{self.name}] Attempt {attempt}/{retries}")
                    logger.info(f"[{self.name}] Tools available: {[getattr(t, 'name', str(t)) for t in self.tools]}")

                    mlflow.log_param("agent_name", self.name)
                    mlflow.log_param("model", self.model)
                    mlflow.log_param("query", query)
                    mlflow.log_param("attempt", attempt)

                    messages = [HumanMessage(content=query)]
                    result = self.agent.invoke({"messages": messages})

                    latency = time.perf_counter() - start_time
                    AGENT_LATENCY.labels(agent_name=self.name).observe(latency)
                    mlflow.log_metric("latency_sec", latency)
                    mlflow.log_metric("response_length", len(str(result)))

                    # Token usage
                    usage = None
                    if isinstance(result, dict) and "usage" in result:
                        usage = result["usage"]
                    elif hasattr(result, "metadata") and "token_usage" in result.metadata:
                        usage = result.metadata["token_usage"]

                    if usage and "total_tokens" in usage:
                        AGENT_TOKENS_USED.labels(agent_name=self.name).observe(usage["total_tokens"])
                        mlflow.log_metric("total_tokens", usage["total_tokens"])

                    mlflow.log_dict({"response": str(result)}, f"agent_output_attempt{attempt}.json")

                    logger.info(f"[{self.name}] ✅ Run completed on attempt {attempt} | latency={latency:.3f}s")
                    return result  # success, return immediately

            except Exception as e:
                AGENT_ERRORS.labels(agent_name=self.name).inc()
                logger.error(f"[{self.name}] ❌ Error on attempt {attempt}: {e}")
                traceback.print_exc()

                if attempt == retries:
                    with mlflow.start_run(run_name=f"{self.name}_error", nested=True):
                        mlflow.log_param("agent_name", self.name)
                        mlflow.log_param("query", query)
                        mlflow.log_param("failed_attempts", attempt)
                        mlflow.log_metric("error", 1)
                    return {"error": str(e)}

                # exponential backoff before retry
                sleep_time = delay * (backoff ** (attempt - 1))
                logger.warning(f"[{self.name}] Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

                attempt += 1

    def get_agent_info(self):
        return {
            "name": self.name,
            "model": self.model,
            "tools_count": len(self.tools),
            "tools": [getattr(t, "name", str(t)) for t in self.tools],
            "has_checkpoints": self.memory is not None,
            "prompt_preview": self.prompt_text[:200],
        }


# --- Test Runner ---
if __name__ == "__main__":
    import yaml

    logger.info("=== Testing ConfigReactAgent with Checkpoints ===")
    try:
        with open("configs/agents.yaml", "r") as f:
            configs = yaml.safe_load(f)

        flight_cfg = configs["agents"]["flight_agent"]
        agent = ConfigReactAgent("flight_agent", flight_cfg)

        info = agent.get_agent_info()
        logger.info(f"📊 Agent Info: {info}")

        response1 = agent.run("Hello, I'm John and I need help booking flights")
        logger.info(f"Response 1: {response1}")

        response2 = agent.run("What's my name?")
        logger.info(f"Response 2: {response2}")

        response3 = agent.run("Cancel my booking with booking_id=FL12345")
        logger.info(f"Response 3: {response3}")

        logger.info("✅ All tests completed!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        traceback.print_exc()
