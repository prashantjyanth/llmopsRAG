# import time, functools, requests, os
# from prometheus_client import Counter, Histogram
# REQS = Counter("orchestrator_requests_total", "Total /run requests")
# LAT  = Histogram("orchestrator_latency_seconds", "Latency /run")
# GOV_URL = os.getenv("GOV_URL", "http://governance:8200/scan")
# GOV_BLOCKING = os.getenv("GOV_BLOCKING", "true").lower() == "true"
# def governed(fn):
#     @functools.wraps(fn)
#     def wrapper(*args, **kwargs):
#         payload = kwargs.get("payload") or (args[1] if len(args) > 1 else None)
#         REQS.inc(); t0 = time.perf_counter()
#         result = fn(*args, **kwargs)
#         LAT.observe(time.perf_counter() - t0)
#         try:
#             resp = requests.post(GOV_URL, json={"input": payload.input, "output": result["answer"]}, timeout=2)
#             data = resp.json()
#             if not data.get("allow") and GOV_BLOCKING: return {"answer": "[blocked by governance]"}
#             ans = result["answer"]
#             for r in data.get("redactions", []): ans = ans.replace(r, "[REDACTED]")
#             result["answer"] = ans
#         except Exception:
#             pass
#         return result
#     return wrapper
import time, functools, os, inspect, asyncio
from prometheus_client import Counter, Histogram
import httpx  # <- non-blocking
# metrics
REQS = Counter("orchestrator_requests_total", "Total /run requests")
LAT  = Histogram("orchestrator_latency_seconds", "Latency /run")
GOV_URL = os.getenv("GOV_URL", "http://governance:8200/scan")
GOV_BLOCKING = os.getenv("GOV_BLOCKING", "true").lower() == "true"

def governed(fn):
    # If the endpoint is async, wrap with async; otherwise wrap sync.
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            REQS.inc()
            t0 = time.perf_counter()

            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result

            LAT.observe(time.perf_counter() - t0)

            # Governance only if we have both input and answer
            try:
                payload = kwargs.get("payload") or (args[1] if len(args) > 1 else None)
                inp = getattr(payload, "input", None) if payload is not None else None
                ans = result.get("answer") if isinstance(result, dict) else None
                if inp is None or ans is None:
                    return result

                async with httpx.AsyncClient(timeout=2) as client:
                    resp = await client.post(GOV_URL, json={"input": inp, "output": ans})
                    data = resp.json()

                if not data.get("allow") and GOV_BLOCKING:
                    return {"answer": "[blocked by governance]"}

                for r in data.get("redactions", []):
                    ans = ans.replace(r, "[REDACTED]")
                result["answer"] = ans
            except Exception:
                # Never break the main flow due to governance issues
                pass

            return result
        return wrapper

    # Sync functions
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        REQS.inc()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        LAT.observe(time.perf_counter() - t0)

        try:
            payload = kwargs.get("payload") or (args[1] if len(args) > 1 else None)
            inp = getattr(payload, "input", None) if payload is not None else None
            ans = result.get("answer") if isinstance(result, dict) else None
            if inp is None or ans is None:
                return result

            # In sync context you can keep requests; or use httpx in a threadpool if you prefer
            import requests
            resp = requests.post(GOV_URL, json={"input": inp, "output": ans}, timeout=2)
            data = resp.json()

            if not data.get("allow") and GOV_BLOCKING:
                return {"answer": "[blocked by governance]"}

            for r in data.get("redactions", []):
                ans = ans.replace(r, "[REDACTED]")
            result["answer"] = ans
        except Exception:
            pass
        return result
    return wrapper
