from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os, sys, time, requests, io, math, traceback
import pandas as pd
import uvicorn
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from langchain_groq import ChatGroq
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLMWrapper
from langchain.schema import BaseMessage
import logging

logger = logging.getLogger("EVALUATION")
logger.setLevel(logging.INFO)

# Remove any existing handlers (avoid duplicates if re-imported)
if logger.hasHandlers():
    logger.handlers.clear()

# Console handler → stdout (Docker captures this)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | tool | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# ---------- Normalization helpers ----------
def to_str(val):
    if isinstance(val, BaseMessage):
        return val.content
    if isinstance(val, list):
        if len(val) == 1:
            return to_str(val[0])
        return [to_str(v) for v in val]
    return "" if val is None else str(val)

def safe_nanmean(values):
    nums = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f):
                nums.append(f)
        except Exception:
            continue
    return sum(nums) / len(nums) if nums else 0.0

def sanitize_num(x, default=0.0):
    try:
        f = float(x)
        return f if math.isfinite(f) else default
    except Exception:
        return default

# ---------- Path setup ----------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(parent_dir)

from core.workflow_builder import WorkflowBuilder
from libs.mlflow_utils import MLflowManager

# ---------- App & Init ----------
app = FastAPI(debug=True)
app.mount("/metrics", make_asgi_app())

mlflow_manager = MLflowManager(tracking_uri="http://mlflow:5000")
builder = WorkflowBuilder()

# ---------- Prometheus ----------
EVAL_RUNS_TOTAL = Counter("eval_runs_total", "Total evaluation runs")
EVAL_ERRORS_TOTAL = Counter("eval_errors_total", "Total evaluation errors")
EVAL_FAITHFULNESS = Gauge("eval_faithfulness", "Faithfulness score", ["agent_name", "model_name", "prompt"])
EVAL_RELEVANCY = Gauge("eval_relevancy", "Answer relevancy score", ["agent_name", "model_name", "prompt"])
EVAL_CONTEXT_RECALL = Gauge("eval_context_recall", "Context recall score", ["agent_name", "model_name", "prompt"])
EVAL_LATENCY = Histogram("eval_latency_seconds", "Evaluation latency in seconds", ["agent_name", "model_name", "prompt"])

# ---------- Ragas LLM ----------
groq_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
ragas_llm = LangchainLLMWrapper(groq_llm)
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
context_recall.llm = ragas_llm

# ---------- Registry helpers ----------
def get_registered_prompts(name):
    try:
        uri = f"http://mlflow_registry_manager:8181/prompts/{name}"
        r = requests.get(uri, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompt: {e}")

def get_registered_models():
    try:
        uri = f"http://mlflow_registry_manager:8181/models"
        r = requests.get(uri, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {e}")

# ---------- Endpoint ----------
@app.post("/evaluate/agent")
async def evaluate_agent(file: UploadFile = File(...), agent_name: str = Form(...)):
    try:
        EVAL_RUNS_TOTAL.inc()

        # Load CSV
        content = await file.read()
        
        print("📂 Uploaded file size:", len(content))
        print("📂 First 200 bytes:", content[:200])
        df = pd.read_csv(io.BytesIO(content))

        required = {"agent_name", "question", "expected", "context"}
        if not required.issubset(df.columns):
            raise HTTPException(400, f"CSV must contain columns: {', '.join(sorted(required))}")

        if agent_name not in builder.agents:
            raise HTTPException(400, f"Agent '{agent_name}' not found in WorkflowBuilder")

        models = [m["name"] for m in get_registered_models()]
        prompt_name = builder.agents[agent_name].prompt_name
        prompts = get_registered_prompts(prompt_name)

        best_combo = None
        best_metrics = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "avg_latency": float("inf")
        }

        for model_name in models:
            for version, prompt_data in prompts.items():
                version_str = str(version)

                # Build agent once per combo
                agent = builder.agents[agent_name]
                agent.model = model_name
                agent.prompt_text = prompt_data.get("template", "")
                agent.build_agent()

                rows = df[df["agent_name"] == agent_name]
                if rows.empty:
                    continue

                eval_records = []
                latencies = []

                for _, row in rows.iterrows():
                    q = to_str(row["question"])
                    gt = to_str(row["expected"])

                    start = time.time()
                    try:
                        output = agent.run(q)
                        print(f"[RUN] model={model_name} v={version_str} q={q[:80]}... -> {to_str(output)[:80]}...")
                        logger.info(f"[EVAL] model={model_name} v={version_str} q={q[:80]}... -> {to_str(output)[:80]}...")
                    except Exception as run_err:
                        EVAL_ERRORS_TOTAL.inc()
                        print(f"[RUN ERROR] model={model_name} v={version_str} q={q[:80]}... -> {run_err}")
                        logger.error(f"[RUN ERROR] model={model_name} v={version_str} q={q[:80]}... -> {run_err}")
                        continue
                    finally:
                        latencies.append(time.time() - start)

                    eval_records.append({
                        "question": q,
                        "answer": to_str(output),
                        "contexts": [],  # add retrieved docs if RAG-based
                        "ground_truth": gt
                    })

                if not eval_records:
                    continue

                # Build dataset
                try:
                    dataset = Dataset.from_list(eval_records)
                    logger.info(f"[DATASET] Created dataset with {len(dataset)} records for model={model_name} v={version_str}")
                    print(f"[DATASET] Created dataset with {len(dataset)} records for model={model_name} v={version_str}")
                except Exception as ds_err:
                    EVAL_ERRORS_TOTAL.inc()
                    print("[DATASET ERROR] Sample:", eval_records[0])
                    print("Error:", ds_err)
                    continue

                # Ragas evaluation
                try:
                    ragas_results = ragas_evaluate(
                        dataset,
                        metrics=[faithfulness, answer_relevancy, context_recall]
                    )
                    print(f"[RAGAS] model={model_name} v={version_str} results: {ragas_results}")
                    logger.info(f"[RAGAS] model={model_name} v={version_str} results: {ragas_results}")
                except Exception as rg_err:
                    EVAL_ERRORS_TOTAL.inc()
                    print("[RAGAS ERROR] Evaluation failed:", rg_err)
                    continue

                # Lists of per-row metric values (may contain NaN)
                faith_list = ragas_results["faithfulness"]
                relev_list = ragas_results["answer_relevancy"]
                ctx_list = ragas_results["context_recall"]
                print(faith_list, relev_list, ctx_list)
                logger.info(f"[METRICS LISTS] faith: {faith_list}, relev: {relev_list}, ctx: {ctx_list}")
                # Aggregate with NaN-safe mean
                faith = sanitize_num(safe_nanmean(faith_list))
                relev = sanitize_num(safe_nanmean(relev_list))
                ctx_recall = sanitize_num(safe_nanmean(ctx_list))
                avg_latency = sanitize_num(sum(latencies) / len(latencies) if latencies else float("inf"))
                print(f"Eval {agent_name} | {model_name}:{version_str} => "
                      f"Faith: {faith:.4f}, Relev: {relev:.4f}, CtxRec: {ctx_recall:.4f}, "
                      f"Latency: {avg_latency:.2f}s")
                logger.info(f"Eval {agent_name} | {model_name}:{version_str} => "
                             f"Faith: {faith:.4f}, Relev: {relev:.4f}, CtxRec: {ctx_recall:.4f}, "
                             f"Latency: {avg_latency:.2f}s")

                # Prometheus (always sanitized floats)
                try:
                    EVAL_FAITHFULNESS.labels(agent_name, model_name, version_str).set(faith)
                    EVAL_RELEVANCY.labels(agent_name, model_name, version_str).set(relev)
                    EVAL_CONTEXT_RECALL.labels(agent_name, model_name, version_str).set(ctx_recall)
                    EVAL_LATENCY.labels(agent_name, model_name, version_str).observe(avg_latency)
                except Exception as prom_err:
                    print("[PROMETHEUS ERROR] Label/value issue:", prom_err)

                # Choose best combo
                if (faith > best_metrics["faithfulness"]) or (
                    faith == best_metrics["faithfulness"] and avg_latency < best_metrics["avg_latency"]
                ):
                    best_combo = (model_name, f"{agent_name}:v{version_str}")
                    best_metrics = {
                        "faithfulness": faith,
                        "answer_relevancy": relev,
                        "context_recall": ctx_recall,
                        "avg_latency": avg_latency
                    }
                    print(f"*** New best model: {best_combo} with faithfulness {faith:.4f} ***")
                    logger.info(f"*** New best model: {best_combo} with faithfulness {faith:.4f} ***")

        if not best_combo:
            raise HTTPException(400, f"No evaluations found for agent '{agent_name}'")

        # Final sanitize before JSON response (no NaN/inf allowed)
        metrics_out = {
            "faithfulness": sanitize_num(best_metrics["faithfulness"]),
            "answer_relevancy": sanitize_num(best_metrics["answer_relevancy"]),
            "context_recall": sanitize_num(best_metrics["context_recall"]),
            "avg_latency": sanitize_num(best_metrics["avg_latency"])
        }

        return {
            "best_model": best_combo[0],
            "best_prompt_version": best_combo[1],
            "metrics": metrics_out
        }

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 Unhandled exception in /evaluate/agent")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8101, reload=True, loop="asyncio", log_level="debug")