import os, time, functools
import mlflow
import inspect
import asyncio
from contextlib import contextmanager


class MLflowManager:
    def __init__(self, tracking_uri: str = None, experiment: str = "genai_ops_demo"):
        tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.langchain.autolog()
        
        
        
    def start_run(self, run_name: str = ""):
        return mlflow.start_run(run_name=run_name)
    @contextmanager
    def trace_span(self, name: str, inputs: dict = None):
        """Start an MLflow trace span for structured logging."""
        with mlflow.start_span(name=name, inputs=inputs or {}):
            yield

    def log_param(self, key: str, value):
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
    def log_latency(self, start_time: float):
        """Logs latency in seconds since start_time"""
        latency = time.perf_counter() - start_time
        mlflow.log_metric("latency_sec", latency)
        return latency

    def log_token_usage(self, result):
        """
        Logs token usage if available in LangChain / LangGraph output.
        Expected structure: result["usage"] or result.metadata["token_usage"]
        """
        usage = None
        if isinstance(result, dict) and "usage" in result:
            usage = result["usage"]
        elif hasattr(result, "metadata") and "token_usage" in result.metadata:
            usage = result.metadata["token_usage"]

        if usage:
            if "prompt_tokens" in usage:
                mlflow.log_metric("prompt_tokens", usage["prompt_tokens"])
            if "completion_tokens" in usage:
                mlflow.log_metric("completion_tokens", usage["completion_tokens"])
            if "total_tokens" in usage:
                mlflow.log_metric("total_tokens", usage["total_tokens"])

def mlflow_run(run_name_param: str | None = None, nested: bool = True):
    def decorator(fn):
        # Async route
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                run_name = run_name_param or fn.__name__
                with mlflow.start_run(run_name=run_name, nested=nested):
                    t0 = time.perf_counter()
                    result = fn(*args, **kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                    mlflow.log_metric("duration_sec", time.perf_counter() - t0)
                    return result
            return wrapper
        # Sync function
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                run_name = run_name_param or fn.__name__
                with mlflow.start_run(run_name=run_name, nested=nested):
                    t0 = time.perf_counter()
                    result = fn(*args, **kwargs)
                    mlflow.log_metric("duration_sec", time.perf_counter() - t0)
                    return result
            return wrapper
    return decorator

