import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import pandas as pd
from utils.config_lodder import Config
from utils.time_decoretor import timeit
from utils.scorellm import score_with_llm
from utils.logger import CustomLogger
import dotenv
dotenv.load_dotenv()

cfg = Config().get()
logger = CustomLogger(name="evaluator").get_logger()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@timeit
def evaluate_prompts():
    df = pd.read_csv(cfg["files"]["test_data"])
    best_score = -1
    best_prompt = None
    best_model = None
    all_results = []

    for prompt_name in cfg["prompt_candidates"]:
        prompt_path = os.path.join(cfg["files"]["prompt_dir"], f"{prompt_name}.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = json.load(f)["template"]

        for model_name in cfg["model_candidates"]:
            full_prompt_id = f"{prompt_name}_{model_name}"
            total_score = 0
            results = []

            for _, row in df.iterrows():
                query = row["query"]
                expected = row["expected"]

                # Render prompt
                formatted_prompt = prompt_template.replace("{{input}}", query)

                try:
                    from langchain_groq import ChatGroq
                    from langchain.schema import HumanMessage
                    client = ChatGroq(
                        api_key=GROQ_API_KEY,
                        model_name=model_name
                    )
                    response = client([HumanMessage(content=formatted_prompt)])
                    generated = response.content.strip()
                    score = score_with_llm(generated, expected, GROQ_API_KEY)
                except Exception as e:
                    generated = f"[ERROR] {e}"
                    score = 0.0
                    logger.error(f"Error for prompt '{full_prompt_id}': {e}")

                total_score += score
                results.append({
                    "prompt": query,
                    "expected": expected,
                    "response": generated,
                    "score": round(score, 3)
                })

            avg_score = total_score / len(df)
            logger.info(f"[🔍] {full_prompt_id} → Avg Score: {avg_score:.3f}")
            all_results.append((prompt_name, model_name, avg_score, results))

            if avg_score > best_score:
                best_score = avg_score
                best_prompt = prompt_name
                best_model = model_name

    # Save best results
    if best_prompt and best_model:
        os.makedirs(cfg["files"]["best_prompt_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(cfg["files"]["best_model_file"]), exist_ok=True)

        shutil.copyfile(
            os.path.join(cfg["files"]["prompt_dir"], f"{best_prompt}.json"),
            os.path.join(cfg["files"]["best_prompt_dir"], f"{best_prompt}.json")
        )

        with open(cfg["files"]["best_model_file"], "w", encoding="utf-8") as f:
            f.write(best_model)

        logger.info(f"\n🏆 Best: Prompt = {best_prompt}, Model = {best_model}, Score = {best_score:.3f}")