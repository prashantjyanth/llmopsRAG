# import regex as re, yaml, os
# CFG = yaml.safe_load(open(os.getenv("GOV_CFG", "/app/configs/governance.yaml"), "r"))
# def pii_filter(text: str):
#     hits = []; 
#     for pat in CFG.get("pii_patterns", []):
#         for m in re.findall(pat, text or ""): hits.append(m)
#     return hits
# def toxicity_score(text: str) -> float:
#     bad = ["hate","kill","stupid"]; score = 0.0
#     for w in bad:
#         if w in (text or "").lower(): score += 0.5
#     return min(score, 1.0)
# def jailbreak_detect(text: str) -> bool:
#     for w in CFG.get("red_team_checks", {}).get("jailbreak_words", []):
#         if w in (text or "").lower(): return True
#     return False
import regex as re, yaml, os
import spacy
from transformers import pipeline
import mlflow
from prometheus_client import Counter, Histogram

# Load governance config
CFG = yaml.safe_load(open(os.getenv("GOV_CFG", "/app/configs/governance.yaml"), "r"))

# Load NLP models
NER = spacy.load("en_core_web_sm")
TOXICITY_MODEL = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

# Prometheus metrics
GOV_REQS = Counter("governance_requests_total", "Total governance requests")
GOV_BLOCKS = Counter("governance_blocked_total", "Total blocked requests")
GOV_PII_HITS = Counter("governance_pii_hits_total", "Total PII detections")
GOV_TOXICITY = Histogram("governance_toxicity_score", "Toxicity scores distribution")

# ---------------- PII Filter ----------------
def pii_filter(text: str):
    """Detect PII using regex + spaCy NER"""
    hits = []

    # Regex patterns from config
    for pat in CFG.get("pii_patterns", []):
        for m in re.findall(pat, text or ""):
            hits.append(m)

    # Named Entity Recognition (NER)
    doc = NER(text or "")
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "DATE", "MONEY"]:
            hits.append(f"{ent.label_}: {ent.text}")

    if hits:
        GOV_PII_HITS.inc(len(hits))

    return hits

# ---------------- Toxicity Score ----------------
def toxicity_score(text: str) -> float:
    """Score toxicity using ML model + fallback"""
    if not text:
        return 0.0
    try:
        preds = TOXICITY_MODEL(text[:512])  # truncate long text
        tox_val = 0.0
        for p in preds[0]:
            if "toxic" in p["label"].lower():
                tox_val = max(tox_val, p["score"])
        GOV_TOXICITY.observe(tox_val)
        return tox_val
    except Exception:
        # fallback simple wordlist
        bad = ["hate", "kill", "stupid"]
        score = 0.0
        for w in bad:
            if w in text.lower():
                score += 0.5
        score = min(score, 1.0)
        GOV_TOXICITY.observe(score)
        return score

# ---------------- Jailbreak Detection ----------------
def jailbreak_detect(text: str) -> bool:
    """Detect jailbreak attempts from config keywords"""
    for w in CFG.get("red_team_checks", {}).get("jailbreak_words", []):
        if w in (text or "").lower():
            return True
    return False

# ---------------- Policy Decision ----------------
def evaluate_policy(input_text: str, output_text: str):
    GOV_REQS.inc()
    reasons, allow, mode = [], True, "allow"

    # Run checks
    redactions = pii_filter((input_text or "") + " " + (output_text or ""))
    tox = toxicity_score(output_text or "")
    jb = jailbreak_detect(input_text or "")

    # Apply policies
    if redactions:
        reasons.append("pii_found")
        mode = CFG.get("pii_action", "redact")  # redact or block

    if tox > CFG.get("toxicity_threshold", 0.8):
        reasons.append("high_toxicity")
        allow, mode = False, "block"

    if jb:
        reasons.append("jailbreak_detected")
        allow, mode = False, "block"

    if not allow:
        GOV_BLOCKS.inc()

    # Log to MLflow
    with mlflow.start_run(run_name="governance_scan", nested=True):
        mlflow.log_params({
            "toxicity": tox,
            "reasons": ",".join(reasons),
            "mode": mode,
        })
        mlflow.log_dict({"input": input_text, "output": output_text}, "scan_payload.json")

    return {"allow": allow, "mode": mode, "reasons": reasons, "redactions": redactions, "scores": {"toxicity": tox}}
