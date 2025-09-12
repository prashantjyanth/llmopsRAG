from fastapi import HTTPException
from mlflow import MlflowClient, MlflowException
import mlflow
from pydantic import BaseModel
import os,sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(parent_dir)
from core.workflow_builder import WorkflowBuilder
import requests
    

from libs.mlflow_utils import MLflowManager
from prometheus_client import Counter, Gauge, Histogram
import time

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


class ComboEvalRequest(BaseModel):
    agent_name: str
    model_names: list
    prompts: list
    test_data_url: str
    thread_id: str = "eval"


def get_registered_models():
    try:
        client = MlflowClient(tracking_uri="http://mlflow:5000")
        models = client.list_registered_models()
        return [model.name for model in models]
    except MlflowException as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {e}")


def get_registered_prompts(name):
    try:
        uri = f"http://localhost:5000/prompts/{name}"
        response = requests.get(uri)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompt: {e}")


def evaluate_all_agents(request: ComboEvalRequest):
    try:
        EVAL_RUNS_TOTAL.inc()

        # Fetch test dataset
        resp = requests.get(request.test_data_url)
        resp.raise_for_status()
        test_data = resp.json()  # [{"input": "...", "expected": "..."}]

        all_agent_results = {}

        for agent_name, agent in builder.agents.items():
            best_combo = None
            best_accuracy = -1
            all_results = []

            for model_name in request.model_names:
                builder.model_name = model_name
                builder.load_model()

                for prompt in request.prompts:
                    if hasattr(builder, "prompt"):
                        builder.prompt = prompt

                    correct, total = 0, 0
                    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

                    start_time = time.perf_counter()
                    with mlflow.start_run(run_name=f"eval_{agent_name}_{model_name}", nested=True):
                        mlflow.log_param("agent_name", agent_name)
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("prompt", prompt)

                        for item in test_data:
                            user_input = item.get("input")
                            expected = item.get("expected")

                            result = agent.run(user_input)
                            output = str(result)

                            # --- Token usage extraction ---
                            if isinstance(result, dict) and "usage" in result:
                                usage = result["usage"]
                                prompt_tokens += usage.get("prompt_tokens", 0)
                                completion_tokens += usage.get("completion_tokens", 0)
                                total_tokens += usage.get("total_tokens", 0)
                            elif hasattr(result, "response_metadata") and "token_usage" in result.response_metadata:
                                usage = result.response_metadata["token_usage"]
                                prompt_tokens += usage.get("prompt_tokens", 0)
                                completion_tokens += usage.get("completion_tokens", 0)
                                total_tokens += usage.get("total_tokens", 0)
                            elif hasattr(result, "usage_metadata"):
                                usage = result.usage_metadata
                                prompt_tokens += usage.get("input_tokens", 0)
                                completion_tokens += usage.get("output_tokens", 0)
                                total_tokens += usage.get("total_tokens", 0)

                            if expected and expected in output:
                                correct += 1
                            total += 1

                        # latency
                        latency = time.perf_counter() - start_time

                        # accuracy
                        accuracy = correct / total if total else 0
                        all_results.append({
                            "model": model_name,
                            "prompt": prompt,
                            "accuracy": accuracy,
                            "latency": latency,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens
                        })

                        # --- Prometheus ---
                        EVAL_ACCURACY.labels(agent_name, model_name, prompt).set(accuracy)
                        EVAL_LATENCY.labels(agent_name, model_name, prompt).observe(latency)
                        EVAL_PROMPT_TOKENS.labels(agent_name, model_name, prompt).set(prompt_tokens)
                        EVAL_COMPLETION_TOKENS.labels(agent_name, model_name, prompt).set(completion_tokens)
                        EVAL_TOTAL_TOKENS.labels(agent_name, model_name, prompt).set(total_tokens)

                        # --- MLflow ---
                        mlflow.log_metric("accuracy", accuracy)
                        mlflow.log_metric("latency_seconds", latency)
                        mlflow.log_metric("prompt_tokens", prompt_tokens)
                        mlflow.log_metric("completion_tokens", completion_tokens)
                        mlflow.log_metric("total_tokens", total_tokens)
                        mlflow.log_dict(all_results[-1], f"{agent_name}_{model_name}_{prompt}_result.json")

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_combo = {
                                "model": model_name,
                                "prompt": prompt,
                                "accuracy": accuracy,
                                "latency": latency,
                                "total_tokens": total_tokens
                            }

            all_agent_results[agent_name] = {
                "best_combo": best_combo,
                "all_results": all_results
            }

        return {
            "results": all_agent_results,
            "status": "success"
        }

    except Exception as e:
        EVAL_ERRORS_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(e))
