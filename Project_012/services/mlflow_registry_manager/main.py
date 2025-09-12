# services/prompt_registry_main.py
import os, io, time, mlflow, yaml,sys
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from pydantic import BaseModel
from prometheus_client import Gauge, Counter, start_http_server

from libs.mlflow_utils.tracking import MLflowManager, mlflow_run
from telemetry import instrument
from middleware import governed
from mlflow.genai.prompts import register_prompt
from mlflow.genai import load_prompt
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException



EXPERIMENT = os.getenv("EXPERIMENT", "genai_ops_demo")
MLflowManager(experiment=EXPERIMENT)

app = FastAPI(title="Prompt Registry")
instrument(app)

PROMPTS_REGISTERED = Gauge("prompt_registry_prompts_registered_total", "Total prompts registered")
LAST_PROMPT_VERSION = Gauge("prompt_registry_last_prompt_version", "Last registered prompt version")
PROMPT_REGISTRY_HITS = Counter('prompt_registry_hits_total', 'Total prompt registry hits')
start_http_server(8001)

class RegisterReq(BaseModel):
    yaml_text: str | None = None  # alternative to file upload

class SimplePromptReq(BaseModel):
    name: str
    prompt_template: str
    description: str | None = None
    input_variables: list[str] | None = None
    tags: dict | None = None

class RegisterModelReq(BaseModel):
    model_name: str
    model_uri: str
    description: str | None = None
    tags: dict | None = None

def _read(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _prepare_template(root: str, item: dict) -> str:
    tmpl = item.get("template")
    if not tmpl and item.get("template_path"):
        tmpl = _read(os.path.join(root, item["template_path"]))
    for var_name, fpath in (item.get("include_files") or {}).items():
        _ = _read(os.path.join(root, fpath))  # read once (you can store if needed)
        if tmpl is not None and f"{{{{ {var_name} }}}}" not in tmpl:
            tmpl += f"\n\n# Included: {var_name}\n{{{{ {var_name} }}}}\n"
        # add minimal trace tags (length/path) so we can resolve at runtime if needed
        item.setdefault("tags", {})
        item["tags"][f"include_file::{var_name}"] = fpath
    return tmpl if tmpl is not None else ""

@app.get("/health")
def health(): 
    PROMPT_REGISTRY_HITS.inc()
    return {"ok": True}

@app.post("/register/prompt")
@governed
@mlflow_run("prompt_registry_register")
async def register_simple_prompt(req: SimplePromptReq):
    PROMPT_REGISTRY_HITS.inc()
    """Register a single prompt with name and template"""
    t0 = time.perf_counter()
    try:
        mlflow.log_param("experiment", EXPERIMENT)
        mlflow.log_param("prompt_name", req.name)
        mlflow.log_param("template_length", len(req.prompt_template))
        
        
        # Set MLflow tags for tracking
        mlflow.set_tags({
            "prompt.name": req.name,
            "prompt.source": "direct_registration"
        })
        
        # Register the prompt with MLflow
        rec = register_prompt(
            name=req.name,
            template=req.prompt_template,
        )
        
        # Update Prometheus metrics
        LAST_PROMPT_VERSION.set(float(rec.version))
        PROMPTS_REGISTERED.inc(1)
        
        # Log MLflow metrics
        mlflow.log_metric("register_latency_sec", time.perf_counter() - t0)
        mlflow.log_metric("prompts_registered", 1)
        
        return {
            "status": "ok",
            "name": rec.name,
            "version": rec.version,
            "uri": f"prompts:/{rec.name}/{rec.version}",
            "description": req.description,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        mlflow.log_param("registration_error", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to register prompt: {str(e)}")



@app.get("/prompts/{prompt_name}")
@governed
def get_prompt_details(prompt_name: str, version: int | None = None):
    PROMPT_REGISTRY_HITS.inc()
    """Get details for a specific prompt or all versions"""
    try:
        if version is None:
            # Get all versions as dict
            v = 1
            prompts: dict[int, dict] = {}
            latest_version = None

            while True:
                try:
                    p = load_prompt(f"prompts:/{prompt_name}/{v}")
                    prompts[v] = {
                        "name": prompt_name,
                        "version": v,
                        "template": p.template,
                        "is_latest": False
                    }
                    latest_version = v
                    v += 1
                except MlflowException:
                    break

            if not prompts:
                raise HTTPException(404, f"Prompt '{prompt_name}' not found")

            # Mark the latest one
            prompts[latest_version]["is_latest"] = True
            return prompts

        else:
            # Get specific version only
            p = load_prompt(f"prompts:/{prompt_name}/{version}")
            return {
                version: {
                    "name": prompt_name,
                    "version": version,
                    "template": p.template,
                    "is_latest": False
                }
            }

    except MlflowException:
        raise HTTPException(404, f"Prompt '{prompt_name}' version {version} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error retrieving prompt: {str(e)}")

# @mlflow_run("prompt_registry_register_yaml")
# def get_prompt_details(prompt_name: str, version: int | None = None):
#     PROMPT_REGISTRY_HITS.inc()
#     """Get details for a specific prompt"""
#     try:
#         if version is None:
#             # Get latest version
#             v = 1
#             latest_prompt = None
#             while True:
#                 try:
#                     p = load_prompt(f"prompts:/{prompt_name}/{v}")
#                     latest_prompt = p
#                     v += 1
#                 except MlflowException:
#                     break
            
#             if latest_prompt is None:
#                 raise HTTPException(404, f"Prompt '{prompt_name}' not found")
            
#             return {
#                 "name": prompt_name,
#                 "version": v - 1,
#                 "template": latest_prompt.template,
#                 "is_latest": True
#             }
#         else:
#             # Get specific version
#                 p = load_prompt(f"prompts:/{prompt_name}/{version}")
#                 return {
#                     "name": prompt_name,
#                     "version": version,
#                     "template": p.template,
#                     "is_latest": False
#                 }
#     except MlflowException:
#         raise HTTPException(404, f"Prompt '{prompt_name}' version {version} not found")
                
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(500, f"Error retrieving prompt: {str(e)}")
    


@app.post("/register/prompt/yaml")
@governed
@mlflow_run("prompt_registry_register_yaml")
async def register_yaml_prompt(
    yaml_file: UploadFile | None = File(default=None),
    body: RegisterReq | None = Body(default=None),
):
    PROMPT_REGISTRY_HITS.inc()
    """Register prompts from YAML format (original functionality)"""
    t0 = time.perf_counter()
    try:
        # resolve YAML content
        if yaml_file:
            raw = await yaml_file.read()
            yaml_text = raw.decode("utf-8")
            root_dir = os.path.dirname(yaml_file.filename) or os.getcwd()
        elif body and body.yaml_text:
            yaml_text = body.yaml_text
            root_dir = os.getcwd()
        else:
            raise HTTPException(status_code=400, detail="Provide either 'yaml_file' or 'yaml_text'.")

        spec = yaml.safe_load(io.StringIO(yaml_text)) or {}
        ml_cfg = (spec.get("mlflow") or {})
        tracking_uri = ml_cfg.get("tracking_uri")
        experiment = ml_cfg.get("experiment") or EXPERIMENT

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.log_param("experiment", experiment)

        prompts = spec.get("prompts") or []
        if not prompts:
            raise HTTPException(status_code=400, detail="No prompts found in YAML.")

        from mlflow.genai import register_prompt

        results = []
        for p in prompts:
            # minimal required fields
            name = p["name"]
            tags = p.get("tags", {})
            version_tag = p.get("version_tag")

            template_text = _prepare_template(root_dir, p)
            mlflow.set_tags({"prompt.name": name, "prompt.version_tag": version_tag or ""})

            rec = register_prompt(
                name=name,
                template=template_text,
                tags=tags,
            )
            LAST_PROMPT_VERSION.set(float(rec.version))
            results.append({"name": rec.name, "version": rec.version, "uri": f"prompts:/{rec.name}/{rec.version}"})

        PROMPTS_REGISTERED.inc(len(results))
        mlflow.log_metric("register_latency_sec", time.perf_counter() - t0)
        mlflow.log_metric("prompts_registered", len(results))
        return {"status": "ok", "registered": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/register/model")
@governed
@mlflow_run("model_registry_register")
async def register_model(req: RegisterModelReq):
    """Register a model in the MLflow Model Registry."""
    try:
        client = MlflowClient()
        # Register the model (if not already registered)
        try:
            client.get_registered_model(req.model_name)
        except Exception:
            client.create_registered_model(
                name=req.model_name,
                tags=req.tags or {},
                description=req.description or ""
            )
        # Create a new model version
        version = client.create_model_version(
            name=req.model_name,
            source=req.model_uri,
            run_id=None
        )
        return {
            "status": "ok",
            "name": req.model_name,
            "version": version.version,
            "model_uri": req.model_uri,
            "description": req.description,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")

@app.get("/models")
@governed
def list_registered_models():
    """List all registered models in MLflow Model Registry."""
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        print("registered models:")
        for m in models:
            print(f" - {m.name}")
        return [
            {
                "name": m.name,
                "description": m.description,
                "tags": m.tags,
                "latest_versions": [
                    {
                        "version": v.version,
                        "status": v.status,
                        "run_id": v.run_id,
                        "source": v.source,
                        "creation_timestamp": v.creation_timestamp,
                    }
                    for v in m.latest_versions
                ]
            }
            for m in models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
