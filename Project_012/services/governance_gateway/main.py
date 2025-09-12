# from fastapi import FastAPI
# from pydantic import BaseModel
# from checks import pii_filter, toxicity_score, jailbreak_detect
# import yaml, os
# CFG = yaml.safe_load(open(os.getenv("GOV_CFG", "/app/configs/governance.yaml"), "r"))
# app = FastAPI(title="Governance Gateway")
# class ScanReq(BaseModel): input: str; output: str | None = None; context: dict | None = None
# @app.post("/scan")
# def scan(req: ScanReq):
#     reasons = []; redactions = pii_filter((req.input or "") + " " + (req.output or "")); tox = toxicity_score(req.output or "")
#     if jailbreak_detect(req.input or ""): reasons.append("jailbreak_detected")
#     allow = tox <= CFG.get("toxicity_threshold", 0.8)
#     if redactions: reasons.append("pii_found")
#     return {"allow": allow, "reasons": reasons, "redactions": redactions, "scores": {"toxicity": tox}}
# @app.get("/health")
# def health(): return {"ok": True}
from fastapi import FastAPI
from pydantic import BaseModel
from checks import evaluate_policy
from prometheus_client import generate_latest
import yaml, os

CFG = yaml.safe_load(open(os.getenv("GOV_CFG", "/app/configs/governance.yaml"), "r"))
app = FastAPI(title="Advanced Governance Gateway")

class ScanReq(BaseModel):
    input: str
    output: str | None = None
    context: dict | None = None

@app.post("/scan")
def scan(req: ScanReq):
    result = evaluate_policy(req.input, req.output)
    return result

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return generate_latest()
