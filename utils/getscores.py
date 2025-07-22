import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage


def score_with_llm(generated: str, expected: str) -> float:
    """
    Uses Groq LLM to score similarity between generated and expected answer (0 to 1).
    """
    try:
        client = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"  # or "llama3-8b-8192"
        )

        messages = [
            SystemMessage(content="You are a helpful evaluator that returns a number between 0 and 1."),
            HumanMessage(
                content=(
                    "Evaluate the similarity between the expected and generated answer.\n\n"
                    f"Expected: {expected}\nGenerated: {generated}\n\n"
                    "Give only a numeric score between 0.0 (no match) and 1.0 (perfect match)."
                )
            )
        ]

        response = client(messages)
        score_text = response.content.strip()
        score = float(score_text)
        return max(0.0, min(score, 1.0))

    except Exception as e:
        print(f"[❌ LLM Eval Error] {e}")
        return 0.0
