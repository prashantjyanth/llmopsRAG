import os, time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    latency_sec: float

class LLMClient:
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = None
        base_url = None
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)
        return LLMResponse(text or "", prompt_tokens, completion_tokens, total_tokens, self.model, self.provider, latency)
