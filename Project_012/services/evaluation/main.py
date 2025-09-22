import time
import io
import os
import sys
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import logging
import functools
import random
import json
import re
 
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from mlflow import MlflowClient
import mlflow
from prometheus_client import Counter, Gauge, Histogram, start_http_server
start_http_server(8111)  # Prometheus metrics server
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset
 
# ---------------- Path setup ----------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(parent_dir)
 
from core.workflow_builder import WorkflowBuilder
from libs.mlflow_utils import MLflowManager
from langgraph.checkpoint.memory import InMemorySaver
 
# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
 
# --- Enhanced Logging Setup ---
class StdoutFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.INFO

class StderrFilter(logging.Filter):
    def filter(self, record):
        return record.levelno > logging.INFO

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.addFilter(StdoutFilter())
stdout_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.addFilter(StderrFilter())
stderr_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))

root_logger = logging.getLogger()
root_logger.handlers = []  # Remove default handlers
root_logger.addHandler(stdout_handler)
root_logger.addHandler(stderr_handler)
root_logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

EVAL_SUMMARY = Gauge(
    "eval_summary",
    "Summary of best evaluation for each agent",
    ["agent_name", "best_model", "best_prompt_version", "metric", "evaluation_timestamp"]
)
 
def update_eval_summary_metrics(results_dir="/app/eval_results"):
    global EVAL_SUMMARY
    logger.info(f"Scanning evaluation results in: {results_dir}")
    if not os.path.exists(results_dir):
        logger.warning(f"Results directory does not exist: {results_dir}")
        return
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(results_dir, fname)
            logger.debug(f"Processing file: {fpath}")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    agent = data.get("agent_name")
                    model = data.get("best_model")
                    prompt_version = str(data.get("best_prompt_version"))
                    metrics = data.get("metrics", {})
                    # Use file modification time as evaluation timestamp
                    eval_timestamp = str(int(os.path.getmtime(fpath)))
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"Setting metric: agent={agent}, model={model}, prompt_version={prompt_version}, metric={metric}, value={value}, evaluation_timestamp={eval_timestamp}")
                            EVAL_SUMMARY.labels(
                                agent_name=agent,
                                best_model=model,
                                best_prompt_version=prompt_version,
                                metric=metric,
                                evaluation_timestamp=eval_timestamp
                            ).set(value)
            except Exception as e:
                logger.warning(f"Could not read or parse {fname}: {e}")
 
# Call this function at startup to populate Prometheus with all eval results
update_eval_summary_metrics()
# ---------------- Retry decorator ----------------
def retry_on_429(max_retries=5, base_delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e):
                        wait = base_delay * (2 ** attempt) + random.random()
                        logger.warning(f"Rate limited (429). Retrying in {wait:.1f}s...")
                        time.sleep(wait)
                    else:
                        raise e
            raise Exception(f"Max retries reached for function {func.__name__}")
        return wrapper
    return decorator
 
# ---------------- LLM setup ----------------
eval_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
ragas_llm_wrapper = LangchainLLMWrapper(eval_llm)
 
@retry_on_429(max_retries=5)
def run_llm_with_retry(prompt: str):
    return eval_llm.invoke(prompt)
 
# ---------------- Init ----------------
mlflow_manager = MLflowManager(tracking_uri="http://mlflow:5000")
builder = WorkflowBuilder()
client = MlflowClient()
 

 
# ---------------- FastAPI ----------------
app = FastAPI()
 
# ---------------- Helper Functions ----------------
def get_registered_prompts(name):
    try:
        uri = f"http://mlflow_registry_manager:8181/prompts/{name}"
        resp = requests.get(uri)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompt: {e}")
 
def get_registered_models():
    try:
        uri = f"http://mlflow_registry_manager:8181/models"
        resp = requests.get(uri)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {e}")
 
# ---------- LLM-only multi-metric evaluation ----------
def llm_eval_scores(output: str, expected: str, context: str = "") -> dict:
    prompt = f"""
    You are an evaluator. Provide scores between 0 and 1 for the following response.
 
    Output: {output}
    Expected: {expected}
    Context: {context}
 
    Return JSON with keys correctness, faithfulness, relevancy.
    Example:
    {{ "correctness": 0.8, "faithfulness": 0.9, "relevancy": 0.85 }}
    """
    try:
        resp = eval_llm.invoke(prompt)
        text = str(resp).strip()
 
        scores = {"correctness": 0.0, "faithfulness": 0.0, "relevancy": 0.0}
 
        if text.startswith("{"):
            parsed = json.loads(text)
            for k in scores.keys():
                scores[k] = float(parsed.get(k, 0.0))
        else:
            nums = re.findall(r"\d*\.?\d+", text)
            for i, k in enumerate(scores.keys()):
                if i < len(nums):
                    scores[k] = float(nums[i])
 
        return {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
    except Exception as e:
        logger.error(f"LLM evaluator error: {e}")
        return {"correctness": 0.0, "faithfulness": 0.0, "relevancy": 0.0}
 
# ---------- Main Evaluation Endpoint ----------
@app.post("/evaluate/agent")
async def evaluate_agent(
    file: UploadFile = File(...),
    agent_name: str = Form(...),
    use_ragas: bool = Form(False)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
 
        if not {"agent_name", "question", "expected"}.issubset(df.columns):
            raise HTTPException(400, "CSV must contain columns: agent_name, question, expected")
 
        if "context" not in df.columns:
            df["context"] = ""
 
        best_combo = None
        best_metrics = {}
 
        models = [m["name"] for m in get_registered_models()]
        prompts = get_registered_prompts(builder.agents[agent_name].prompt_name)
 
        with mlflow.start_run(run_name=f"EVAL_{agent_name}", nested=False):
            for model_name in models:
                for version, prompt_data in prompts.items():
                    builder.agents[agent_name].model = model_name
                    builder.agents[agent_name].prompt_text = prompt_data["template"]
                    builder.agents[agent_name].build_agent()
 
                    results, latencies = [], []
                    correctness_sum, faith_sum, rel_sum, total_count = 0.0, 0.0, 0.0, 0
 
                    with mlflow.start_run(run_name=f"{agent_name}_{model_name}_v{version}", nested=True):
                        mlflow.log_params({
                            "agent_name": agent_name,
                            "model": model_name,
                            "prompt_version": version
                        })
 
                        for _, row in df.iterrows():
                            if row["agent_name"] != agent_name:
                                continue
                            question, expected, context = row["question"], row["expected"], row["context"]
                            start = time.time()
                            try:
                                output = builder.agents[agent_name].run(question)
                                latency = time.time() - start
                                latencies.append(latency)
 
                                if use_ragas:
                                    results.append({
                                        "question": question,
                                        "ground_truth": expected,
                                        "answer": str(output),
                                        "contexts": [context] if context else []
                                    })
                                else:
                                    scores = llm_eval_scores(str(output), expected, context)
                                    correctness_sum += scores["correctness"]
                                    faith_sum += scores["faithfulness"]
                                    rel_sum += scores["relevancy"]
                                    total_count += 1
                                    logger.info(f"Q----------------------------->: {question} | Scores: {scores} | Latency: {latency:.2f}s")
                                    logger.info(f"Output:-----------------------> {output}")

                                    mlflow.log_dict(
                                        {
                                            "question": question,
                                            "output": str(output),
                                            "expected": expected,
                                            "scores": scores,
                                            "latency": latency,
                                        },
                                        f"results/{agent_name}_{model_name}_v{version}_{total_count}.json"
                                    )
 
                            except Exception:
                                logger.exception(f"Error evaluating {model_name}:{version}")
 
                        # --- Summary ---
                        if use_ragas and results:
                            eval_dataset = Dataset.from_list(results)
                            ragas_result = evaluate(
                                eval_dataset,
                                metrics=[answer_correctness, faithfulness, answer_relevancy],
                                llm=ragas_llm_wrapper
                            )
                            df_scores = ragas_result.to_pandas()
 
                            acc = df_scores["answer_correctness"].mean()
                            faith = df_scores["faithfulness"].mean()
                            rel = df_scores["answer_relevancy"].mean()
 
                            mlflow.log_metrics({
                                "mean_correctness": acc,
                                "mean_faithfulness": faith,
                                "mean_relevancy": rel,
                                "mean_latency": sum(latencies) / len(latencies) if latencies else 0
                            })
 
                            if not best_combo or acc > best_metrics.get("correctness", 0):
                                best_combo = (model_name, version)
                                best_metrics = {"correctness": acc, "faithfulness": faith, "relevancy": rel,
                                                "latency": sum(latencies) / len(latencies)}
 
                        elif not use_ragas and total_count > 0:
                            acc = correctness_sum / total_count
                            faith = faith_sum / total_count
                            rel = rel_sum / total_count
                            avg_latency = sum(latencies) / len(latencies) if latencies else 0
 
                            mlflow.log_metrics({
                                "mean_correctness": acc,
                                "mean_faithfulness": faith,
                                "mean_relevancy": rel,
                                "mean_latency": avg_latency,
                                "questions_evaluated": total_count
                            })
 
                            if not best_combo or acc > best_metrics.get("correctness", 0):
                                best_combo = (model_name, version)
                                best_metrics = {"correctness": acc, "faithfulness": faith, "relevancy": rel,
                                                "latency": avg_latency}
 
        if not best_combo:
            raise HTTPException(400, f"No evaluations found for agent '{agent_name}'")
 
        prompt_obj = prompts.get(best_combo[1], {})
       
        client.set_prompt_alias(name=prompt_obj.get("name"), alias="production", version=prompt_obj.get("version"))
 
        # --- Save summary JSON ---
        summary = {
            "agent_name": agent_name,
            "best_model": best_combo[0],
            "best_prompt_version": best_combo[1],
            "metrics": best_metrics
        }
        # Add evaluation timestamp (current time, seconds since epoch)
        eval_timestamp = int(time.time())
        summary["evaluation_timestamp"] = eval_timestamp
        results_dir = "/app/eval_results"
        os.makedirs(results_dir, exist_ok=True)
        json_path = os.path.join(results_dir, f"{agent_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # Log timestamp in MLflow
        mlflow.log_param("evaluation_timestamp", eval_timestamp)

        # return summary
        update_eval_summary_metrics(results_dir=results_dir)
        mlflow.log_dict(summary, f"results/{agent_name}_summary.json")
        return summary
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
 
# ---------------- Run App ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("evaluation:app", host="0.0.0.0", port=8101, reload=True)

