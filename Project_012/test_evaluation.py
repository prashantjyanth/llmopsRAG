from fastapi import HTTPException
from mlflow import MlflowClient, MlflowException
from pydantic import BaseModel
from core.workflow_builder import WorkflowBuilder
import requests
from libs.mlflow_utils import MLflowManager
import mlflow
import pandas as pd
from dotenv import load_dotenv
import os
import json
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
mlflow_manager = MLflowManager()
builder = WorkflowBuilder()

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Groq evaluator LLM (choose llama3-8b or llama3-70b)
evaluator = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0)

from langchain_groq import ChatGroq

# Initialize ChatGroq evaluator

def evaluate_with_llm(expected: str, output: str) -> float:
    """
    Use Groq LLaMA3-8B to score accuracy between expected and model output.
    Returns a float between 0 and 1.
    """
    score_prompt = f"""
You are an evaluator. Compare the model output with the expected answer.  

- Return only a single number between 0 and 1.  
- 1.0 = fully correct and semantically equivalent.  
- 0.0 = completely wrong or irrelevant.  
- Values in between reflect partial correctness.  
CRITICAL: do not add pre post text. Just return float number between 0 and 1.
Expected Answer: {expected}  
Model Output: {output}
"""
    try:
        resp = evaluator.invoke(score_prompt)
        score_text = resp.content.strip()
        return float(score_text)
    except Exception as e:
        print(f"⚠️ Error during LLM evaluation: {e}")
        return 0.0


class ComboEvalRequest(BaseModel):
    agent_name: str
    model_names: list
    prompts: list
    test_data_url: str
    thread_id: str = "eval"

def get_registered_models():
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        return [model.name for model in models]
    except MlflowException as e:
        raise HTTPException(status_code=500, detail=f"MLflow error: {e}")
def get_prompts(name):
    v = 1
    latest_prompt = None
    all_prompts = {}
    while True:
        try:
            p = mlflow.genai.load_prompt(f"prompts:/{name}/{v}")
            latest_prompt = p
            all_prompts[v] = p
            v += 1
        except Exception as e:
            break
    return all_prompts

def get_registered_prompts(name):
    try:
        uri = f"http://localhost:8000/prompts/{name}"
        response = requests.get(uri)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompt: {e}")

def evaluate_all_agents(test_csv: str, builder, output_json: str = "best_combos.json"):
    """
    Evaluate all agents across models and prompt versions using ChatGroq.
    Stores best combo (model + prompt + score) for each agent in JSON.
    """
    df = pd.read_csv(test_csv,sep="\t")
    print(df.head())
    print(df.columns)
    df = df[["agent_name", "input", "expected"]]
    all_agent_results = {}

    for agent_name, agent in builder.agents.items():
        best_combo = None
        best_score = -1

        for model_name in get_registered_models():
            agent.model = model_name
            agent.llm.model_name = model_name

            for prompt_version, prompt in get_prompts(agent.prompt_name).items():
                print(f"🔎 Evaluating Agent={agent_name}, Model={model_name}, Prompt={prompt_version}")
                total_score = 0
                total = 0
                agent.prompt_text = prompt.template
                agent.rebuild_agent()
                for _, row in df[df["agent_name"] == agent_name].iterrows():
                    user_input = row["input"]
                    expected = row["expected"]

                    try:
                        # Run the agent
                        config = {"configurable": {"thread_id": "test"}}
                        result = agent.run(user_input) 
                        print("______________________________\n",result)                       
                        output = str(result)

                        # Evaluate with LLM
                        score = evaluate_with_llm(expected, output)
                        total_score += score
                        total += 1
                    except Exception as e:
                        print(f"⚠️ Error running agent {agent_name} with {model_name}/{prompt_version}: {e}")
                        total += 1

                avg_score = total_score / total if total else 0

                if avg_score > best_score:
                    best_score = avg_score
                    best_combo = {
                        "model": model_name,
                        "prompt_version": prompt_version,
                        "avg_score": avg_score
                    }

        all_agent_results[agent_name] = best_combo

    # Save only best combos
    with open(output_json, "w") as f:
        json.dump(all_agent_results, f, indent=4)

    print(f"✅ Best combos saved to {output_json}")
    return all_agent_results

# Example usage:
if __name__ == "__main__":
    # Fill with your actual values
    # print(get_registered_models())
    # Example CSV file path and builder instance
    test_csv = "test_datasets/testcase.csv"  # Replace with your actual CSV file path
    evaluate_all_agents(test_csv, builder)
    # for agent_name, agent in builder.agents.items():
    #     print(f"Agent: {agent_name}, Prompts: {get_prompts(agent.prompt_name)}")
    # request = ComboEvalRequest(
    #     agent_name="flight_agent",
    #     model_names=["llama3-70b-8192", "other_model"],
    #     prompts=["Prompt A", "Prompt B"],
    #     test_data_url="http://your-test-data-api/dataset"
    # )
    # result = evaluate_all_agents(request)
    # print(result)