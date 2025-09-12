from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mlflow import MlflowClient, MlflowException
import mlflow
import os, sys, time, requests
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import uvicorn

# from services.orch.main import ORCH_HEALTH

# --- Path setup ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(parent_dir)

from core.workflow_builder import WorkflowBuilder
from libs.mlflow_utils import MLflowManager
from langchain_groq import ChatGroq  # or OpenAI, Anthropic, etc.

# Example with Groq Llama3
eval_llm = ChatGroq(model="llama3-8b-8192",api_key=os.getenv("GROQ_API_KEY"))

# --- Init ---
mlflow_manager = MLflowManager(tracking_uri="http://mlflow:5000")
builder = WorkflowBuilder()

# --- Prometheus Metrics ---
EVAL_RUNS_TOTAL = Counter("eval_runs_total", "Total evaluation runs")
EVAL_ERRORS_TOTAL = Counter("eval_errors_total", "Total evaluation errors")
EVAL_ACCURACY = Gauge("eval_accuracy", "Evaluation accuracy", ["agent_name", "model_name", "prompt"])
EVAL_LATENCY = Histogram("eval_latency_seconds", "Evaluation latency in seconds", ["agent_name", "model_name", "prompt"])
EVAL_PROMPT_TOKENS = Gauge("eval_prompt_tokens", "Prompt tokens used", ["agent_name", "model_name", "prompt"])
EVAL_COMPLETION_TOKENS = Gauge("eval_completion_tokens", "Completion tokens used", ["agent_name", "model_name", "prompt"])
EVAL_TOTAL_TOKENS = Gauge("eval_total_tokens", "Total tokens used", ["agent_name", "model_name", "prompt"])


# --- Request model ---
class ComboEvalRequest(BaseModel):
    agent_name: str
    model_names: list
    prompts: list
    test_data_url: str
    thread_id: str = "eval"


# --- Helper functions ---
# def get_registered_models():
#     try:
#         client = MlflowClient(tracking_uri="http://mlflow:5000")
#         models = client.list_registered_models()
#         return [model.name for model in models]
#     except MlflowException as e:
#         raise HTTPException(status_code=500, detail=f"MLflow error: {e}")


def get_registered_prompts(name):
    try:
        uri = f"http://mlflow_registry_manager:8181/prompts/{name}"
        print(uri)
        response = requests.get(uri)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompt: {e}")

def get_registered_models():
    try:
        uri = f"http://mlflow_registry_manager:8181/models"
        response = requests.get(uri)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {e}")

def llm_accuracy_score(output: str, expected: str) -> float:
    """
    Returns a semantic similarity score between 0.0 and 1.0
    using an LLM as a grader.
    """
    prompt = f"""
    You are an evaluator. 
    Compare the following output with the expected answer.

    Output: {output}
    Expected: {expected}

    Return a single floating-point number between 0 and 1:
    - 1.0 if they mean exactly the same
    - 0.0 if they are completely different
    - Intermediate values (like 0.7, 0.85, etc.) if they are partially correct.
    Only return the number, nothing else.
    """

    resp = eval_llm.invoke(prompt)   # eval_llm could be ChatGroq, OpenAI, etc.
    try:
        score = float(str(resp).strip())
        return max(0.0, min(1.0, score))  # clamp between 0–1
    except:
        return 0.0

print(get_registered_models())
print(get_registered_prompts("flight_agent_prompt"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import time
import io

app = FastAPI()

@app.post("/evaluate/agent")
async def evaluate_agent(file: UploadFile = File(...), agent_name: str = Form(...)):
    try:
        # --- Load CSV ---
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        if not {"agent_name", "question", "expected"}.issubset(df.columns):
            raise HTTPException(400, "CSV must contain columns: agent_name, question, expected")

        best_combo = None
        best_metrics = {"accuracy": 0, "avg_latency": float("inf")}

        models = get_registered_models()
        prompts = get_registered_prompts(agent_name)

        for model_name in models.keys():
            for version, prompt_data in prompts.items():
                acc_count = 0
                total_count = 0
                latencies = []

                for _, row in df.iterrows():
                    if row["agent_name"] != agent_name:
                        continue

                    question = row["question"]
                    expected = row["expected"]

                    start = time.time()
                    try:
                        builder.agents[agent_name].model = model_name
                        builder.agents[agent_name].prompt_text = prompt_data["template"]
                        builder.agents[agent_name].build_agent()

                        output =  builder.agents[agent_name].run(question)  # assumes WorkflowBuilder.run()
                        latency = time.time() - start

                        correct = (output.strip().lower() == expected.strip().lower())
                        if correct:
                            acc_count += 1
                        total_count += 1
                        latencies.append(latency)

                        # Prometheus
                        EVAL_ACCURACY.labels(agent_name, model_name, version).set(acc_count / total_count)
                        EVAL_LATENCY.labels(agent_name, model_name, version).observe(latency)

                    except Exception as e:
                        EVAL_ERRORS_TOTAL.inc()
                        print(f"Error evaluating {model_name}:{version} - {e}")

                if total_count > 0:
                    accuracy = acc_count / total_count
                    avg_latency = sum(latencies) / len(latencies) if latencies else float("inf")

                    # Pick best combo
                    if (accuracy > best_metrics["accuracy"]) or (
                        accuracy == best_metrics["accuracy"] and avg_latency < best_metrics["avg_latency"]
                    ):
                        best_combo = (model_name, f"{agent_name}:v{version}")
                        best_metrics = {"accuracy": accuracy, "avg_latency": avg_latency}

        if not best_combo:
            raise HTTPException(400, f"No evaluations found for agent '{agent_name}'")

        return {
            "best_model": best_combo[0],
            "best_prompt_version": best_combo[1],
            "metrics": best_metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# def evaluate_all_agents(request: ComboEvalRequest):
#     try:
#         EVAL_RUNS_TOTAL.inc()

#         # Fetch test dataset
#         resp = requests.get(request.test_data_url)
#         resp.raise_for_status()
#         test_data = resp.json()  # [{"input": "...", "expected": "..."}]

#         all_agent_results = {}

#         for agent_name, agent in builder.agents.items():
#             best_combo = None
#             best_accuracy = -1
#             all_results = []

#             for model_name in request.model_names:
#                 builder.model_name = model_name
#                 builder.load_model()   # fixed to respect builder.model_name

#                 for prompt in request.prompts:
#                     if hasattr(builder, "prompt"):
#                         builder.prompt = prompt

#                     correct, total = 0, 0
#                     prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

#                     start_time = time.perf_counter()
#                     with mlflow.start_run(run_name=f"eval_{agent_name}_{model_name}", nested=True):
#                         mlflow.log_param("agent_name", agent_name)
#                         mlflow.log_param("model_name", model_name)
#                         mlflow.log_param("prompt", prompt)

#                         for item in test_data:
#                             user_input = item.get("input")
#                             expected = item.get("expected")

#                             result = agent.run(user_input)
#                             output = str(result)

#                             # --- Token usage extraction ---
#                             if isinstance(result, dict) and "usage" in result:
#                                 usage = result["usage"]
#                                 prompt_tokens += usage.get("prompt_tokens", 0)
#                                 completion_tokens += usage.get("completion_tokens", 0)
#                                 total_tokens += usage.get("total_tokens", 0)
#                             elif hasattr(result, "response_metadata") and "token_usage" in result.response_metadata:
#                                 usage = result.response_metadata["token_usage"]
#                                 prompt_tokens += usage.get("prompt_tokens", 0)
#                                 completion_tokens += usage.get("completion_tokens", 0)
#                                 total_tokens += usage.get("total_tokens", 0)
#                             elif hasattr(result, "usage_metadata"):
#                                 usage = result.usage_metadata
#                                 prompt_tokens += usage.get("input_tokens", 0)
#                                 completion_tokens += usage.get("output_tokens", 0)
#                                 total_tokens += usage.get("total_tokens", 0)

#                             if expected and expected in output:
#                                 correct += 1
#                             total += 1

#                         # latency
#                         latency = time.perf_counter() - start_time

#                         # accuracy
#                         accuracy = correct / total if total else 0
#                         all_results.append({
#                             "model": model_name,
#                             "prompt": prompt,
#                             "accuracy": accuracy,
#                             "latency": latency,
#                             "prompt_tokens": prompt_tokens,
#                             "completion_tokens": completion_tokens,
#                             "total_tokens": total_tokens
#                         })

#                         # --- Prometheus ---
#                         EVAL_ACCURACY.labels(agent_name, model_name, prompt).set(accuracy)
#                         EVAL_LATENCY.labels(agent_name, model_name, prompt).observe(latency)
#                         EVAL_PROMPT_TOKENS.labels(agent_name, model_name, prompt).set(prompt_tokens)
#                         EVAL_COMPLETION_TOKENS.labels(agent_name, model_name, prompt).set(completion_tokens)
#                         EVAL_TOTAL_TOKENS.labels(agent_name, model_name, prompt).set(total_tokens)

#                         # --- MLflow ---
#                         mlflow.log_metric("accuracy", accuracy)
#                         mlflow.log_metric("latency_seconds", latency)
#                         mlflow.log_metric("prompt_tokens", prompt_tokens)
#                         mlflow.log_metric("completion_tokens", completion_tokens)
#                         mlflow.log_metric("total_tokens", total_tokens)
#                         mlflow.log_dict(all_results[-1], f"{agent_name}_{model_name}_{prompt}_result.json")

#                         if accuracy > best_accuracy:
#                             best_accuracy = accuracy
#                             best_combo = {
#                                 "model": model_name,
#                                 "prompt": prompt,
#                                 "accuracy": accuracy,
#                                 "latency": latency,
#                                 "total_tokens": total_tokens
#                             }

#             all_agent_results[agent_name] = {
#                 "best_combo": best_combo,
#                 "all_results": all_results
#             }

#         return {
#             "results": all_agent_results,
#             "status": "success"
#         }

#     except Exception as e:
#         EVAL_ERRORS_TOTAL.inc()
#         raise HTTPException(status_code=500, detail=str(e))


# # --- FastAPI Service ---
# app = FastAPI(title="Evaluation Service", version="1.0.0")

# # Secure CORS config
# origins = [
#     "http://localhost:3000",      # local frontend
#     "http://127.0.0.1:3000",
#     "https://your-frontend.com"   # production frontend
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["Authorization", "Content-Type"],
# )

# # Prometheus metrics
# metrics_app = make_asgi_app()
# app.mount("/metrics", metrics_app)

# @app.get("/health")
# def health():
#     ORCH_HEALTH.set(1)  # ✅ mark healthy
#     return {"status": "healthy"}

# @app.get("/models")
# def list_models():
#     return {"models": get_registered_models()}

# @app.get("/prompts/{name}")
# def list_prompts(name: str):
#     return get_registered_prompts(name)

# @app.post("/evaluate")
# def evaluate(request: ComboEvalRequest):
#     return evaluate_all_agents(request)


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8101, reload=True)
