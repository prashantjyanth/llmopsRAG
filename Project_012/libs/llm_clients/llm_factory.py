import os
from typing import Any, Dict
from langchain_groq.chat_models import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

# Optional: Print some environment variables to verify they're loaded
print("Environment variables loaded:")
print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
print(f"ORCHESTRATOR_URL: {os.getenv('ORCHESTRATOR_URL', 'Not set')}")
print(f"EXPERIMENT: {os.getenv('EXPERIMENT', 'Not set')}")

EXPERIMENT = os.getenv('EXPERIMENT', 'default_experiment')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI1', 'http://localhost:5000')

def _load_prompt_text(cfg: Dict[str, Any]) -> str:
    """Pull human prompt text from MLflow by name (or fallback to inline)."""
    name = cfg.get("prompt_name")
    if not name:
        return cfg.get("template", "{input}")
    import mlflow
    from mlflow.exceptions import MlflowException
    uri = f"prompts:/{name}/{cfg['prompt_version']}" if cfg.get("prompt_version") else f"prompts:/{name}@production"
    try:
        p = mlflow.genai.load_prompt(uri)
        return p.template
    except MlflowException as e:
        raise RuntimeError(f"MLflow prompt '{name}' load failed: {e}")

def llm_factory(cfg: Dict[str, Any]):
    """
    Build a classic LLMChain over ChatGroq.
    • No input_variables list required; the chain infers from the prompt template.
    • You can optionally keep a 'system' message; remove if you don't want it.
    """
    provider = cfg.get("llm_provider", "groq").lower()
    if provider != "groq":
        raise ValueError(f"Only 'groq' is wired here, got: {provider}")

    model_name  = cfg.get("llm_model", cfg.get("llm", "llama-3.1-70b-versatile"))
    temperature = float(cfg.get("temperature", 0.2))

    chat = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Load the prompt template text
    human_tmpl_text = _load_prompt_text(cfg)
    
    # Create a proper ChatPromptTemplate from the text
    from langchain.prompts import PromptTemplate
    prompt_template = PromptTemplate.from_template(human_tmpl_text)
    
    # Create the chain with the proper prompt template
    chain = LLMChain(llm=chat, prompt=prompt_template)

    class _Wrapper:
        def invoke(self, inputs: Dict[str, Any]) -> str:
            # Pass ALL inputs to the chain - LangChain will map them to prompt variables
            res = chain.invoke(inputs)  # All variables in inputs dict will be available to the prompt
            return res["text"] if isinstance(res, dict) and "text" in res else str(res)
    return _Wrapper()
