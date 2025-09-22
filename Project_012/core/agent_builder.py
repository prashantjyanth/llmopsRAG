"""Agent construction utilities.

This module defines the `ConfigReactAgent` class, a thin wrapper around a
LangGraph ReAct style agent configured from YAML + MLflow hosted prompts.

Key responsibilities:
    * Load and register tools dynamically from dotted import paths.
    * Fetch prompt templates from MLflow Model Registry (falling back to defaults).
    * Instantiate a Groq LLM client (model controlled via env / argument).
    * Provide retry with exponential backoff + MLflow nested run logging.
    * Expose Prometheus metrics for: runs, errors, latency, token usage.
    * (Optionally) enable conversational memory via `MemorySaver` checkpoints.

Environment variables consulted:
    MLFLOW_EXPERIMENT       - Name of (or path to) the MLflow experiment.
    MLFLOW_TRACKING_URI     - Tracking server URI.
    GROQ_API_KEY            - API key for Groq LLM backend.

Return types:
    The `run` method currently advertises `str` but may return a dict with an
    `error` key on failure or the raw agent result object. This is kept for
    backward-compat; callers should be robust to non-string responses.

NOTE: The test runner at the bottom may be outdated if the `__init__` signature
changes (it currently expects a `checkpoints` arg). It is retained for manual
smoke testing inside the container / repo context.
"""

import importlib, os, time, logging, traceback, sys
import mlflow
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from prometheus_client import Counter, Histogram
import dotenv
import json
import math
from typing import Optional, Callable, Union
dotenv.load_dotenv()

# --- Logging Setup ---
# Force logging to stdout (useful for Docker / k8s log collectors)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | agent | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("agent")

# --- MLflow Setup ---
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "genai_ops_demo")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
mlflow.langchain.autolog()
# def safe_observe(histogram, value):
#     if value is not None and isinstance(value, (int, float)) and not math.isnan(value):
#         histogram.observe(value)

# --- Prometheus Metrics ---
# AGENT_RUNS = Counter("agent_runs_total", "Total agent runs", ["agent_name"])
# AGENT_ERRORS = Counter("agent_errors_total", "Total agent run errors", ["agent_name"])
# AGENT_LATENCY = Histogram("agent_latency_seconds", "Agent latency (seconds)", ["agent_name"])
# AGENT_TOKENS_USED = Histogram("agent_tokens_used", "Tokens used per agent run", ["agent_name"])

# # Seed metrics so they show up in /metrics
# for agent in ["flight_agent", "hotel_agent"]:
#     AGENT_RUNS.labels(agent).inc(0)
#     AGENT_ERRORS.labels(agent).inc(0)
#     AGENT_LATENCY.labels(agent).observe(0.0)
#     AGENT_TOKENS_USED.labels(agent).observe(0.0)

class ConfigReactAgent:
    def __init__(self, checkpoints, name: str, cfg: dict, model: str = "llama-3.1-8b-instant"):
        """Initialize a configurable ReAct agent.

        Args:
            checkpoints: Truthy value enables stateful memory via `MemorySaver`.
                        (Could be a bool or structure; only truthiness is used.)
            name: Logical agent name (used in metrics, logging, MLflow runs).
            cfg: Dict containing keys:
                 - tools: list[str] of dotted import paths to tool callables
                 - prompt_name: (optional) MLflow prompt registry name
                 - default_prompt: fallback string if prompt not found
            model: LLM model identifier string for Groq backend.

        Side Effects:
            * Registers / seeds Prometheus metrics.
            * Attempts to fetch and cache an MLflow prompt template.
            * Builds the underlying LangGraph agent instance.

        Raises:
            ValueError: If LLM instantiation fails.
        """
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
        """(Re)build the underlying agent instance.

        Useful when tools, prompt, or checkpoint configuration changes after
        object creation. This mirrors the logic in `__init__` with minimal
        duplication for clarity rather than clever factoring.
        """
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
        """Dynamically import tool callables.

        Each dotted path is split into a module and attribute name; failures
        are logged and skipped (non-fatal) so a single bad tool does not
        prevent agent creation.

        Args:
            tool_paths: List of dotted import strings (e.g. "pkg.module.func").

        Returns:
            List of successfully resolved tool callables.
        """
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
        """Attempt to load an MLflow prompt template.

        Args:
            prompt_name: Name of the prompt in the MLflow registry. If empty or
                         retrieval fails, returns an empty string so caller can
                         apply fallback logic.

        Returns:
            Prompt template text or empty string if unavailable.
        """
        if not prompt_name:
            return ""
        try:
            prompt_text = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production").template  # type: ignore
            logger.info(f"[{self.name}] ✅ Loaded prompt from MLflow: {prompt_name}")
            return prompt_text
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Could not load prompt '{prompt_name}' from MLflow: {e}")
            return ""

    def run(
        self,
        query: str,
        retries: int = 3,
        delay: float = 2.0,
        backoff: float = 2.0,
        on_complete: Optional[Callable] = None
    ) -> Union[str, dict]:
        """Execute the agent with retries, metrics, MLflow logging, and structured progress logs."""

        def safe_observe(metric, value):
            if isinstance(value, (int, float)) and not math.isnan(value):
                metric.observe(value)

        logger.info(f"[{self.name}] 🚀 Starting run | query='{query}'")
        # AGENT_RUNS.labels(agent_name=self.name).inc()

        attempt = 1
        start_time = time.perf_counter()

        while attempt <= retries:
            try:
                with mlflow.start_run(run_name=f"{self.name}_run_attempt{attempt}", nested=True):
                    logger.info(json.dumps({
                        "event": "PROGRESS",
                        "agent": self.name,
                        "attempt": attempt,
                        "phase": "start",
                        "query": query[:60]
                    }))

                    mlflow.log_params({
                        "agent_name": self.name,
                        "model": self.model,
                        "query": query,
                        "attempt": attempt,
                        "retry_delay": delay,
                        "backoff_factor": backoff
                    })
                    mlflow.set_tag("phase", "invoke")

                    messages = [HumanMessage(content=query)]
                    logger.info(json.dumps({
                        "event": "PROGRESS",
                        "agent": self.name,
                        "attempt": attempt,
                        "phase": "invoke",
                        "tool_count": len(self.tools)
                    }))

                    result = self.agent.invoke({"messages": messages})
                    logger.info(f"[{self.name}] ✅✅✅✅✅✅✅✅✅✅✅✅ Received result: {result}")
                    latency = time.perf_counter() - start_time
                    # safe_observe(AGENT_LATENCY.labels(agent_name=self.name), latency)
                    mlflow.log_metrics({
                        "latency_sec": latency,
                        "response_length": len(str(result))
                    })

                    usage = None
                    if isinstance(result, dict) and "usage" in result:
                        usage = result["usage"]
                    elif hasattr(result, "metadata") and "token_usage" in result.metadata:
                        usage = result.metadata["token_usage"]

                    if usage and "total_tokens" in usage:
                        # safe_observe(AGENT_TOKENS_USED.labels(agent_name=self.name), usage["total_tokens"])
                        mlflow.log_metric("total_tokens", usage["total_tokens"])

                    mlflow.log_dict({"response": str(result)}, f"agent_output_attempt{attempt}.json")
                    mlflow.set_tag("status", "success")

                    logger.info(json.dumps({
                        "event": "PROGRESS",
                        "agent": self.name,
                        "attempt": attempt,
                        "phase": "complete",
                        "latency": round(latency, 3)
                    }))
                    logger.info(f"[{self.name}] ✅ Run completed on attempt {attempt} | latency={latency:.3f}s")

                    if on_complete:
                        on_complete(result, attempt, latency)

                    return result

            except Exception as e:
                # AGENT_ERRORS.labels(agent_name=self.name).inc()
                logger.error(f"[{self.name}] ❌ Error on attempt {attempt}: {e}")
                traceback.print_exc()

                if attempt == retries:
                    with mlflow.start_run(run_name=f"{self.name}_error", nested=True):
                        mlflow.log_params({
                            "agent_name": self.name,
                            "query": query,
                            "failed_attempts": attempt
                        })
                        mlflow.log_metric("error", 1)
                        mlflow.set_tag("status", "error")
                        mlflow.set_tag("error_type", type(e).__name__)
                    return {"error": str(e)}

                sleep_time = delay * (backoff ** (attempt - 1))
                logger.warning(f"[{self.name}] Retrying in {sleep_time:.1f}s...")
                logger.info(json.dumps({
                    "event": "PROGRESS",
                    "agent": self.name,
                    "attempt": attempt,
                    "phase": "retry_wait",
                    "sleep": round(sleep_time, 2)
                }))
                time.sleep(sleep_time)
                attempt += 1


    def get_agent_info(self):
        """Return lightweight diagnostic metadata about the agent."""
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
        # NOTE: Example call retained from earlier version; signature currently:
        # ConfigReactAgent(checkpoints, name, cfg, model=...)
        # Adjust 'checkpoints' argument as needed (e.g., pass True for enabling memory).
        agent = ConfigReactAgent(True, "flight_agent", flight_cfg)

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
