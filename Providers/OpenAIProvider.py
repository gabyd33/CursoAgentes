import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from LLMProvider import LLMProvider

registro_llamadas: list[dict[str, Any]] = []


def _estimar_costo_openai(prompt_tokens: int, completion_tokens: int) -> float | None:
    input_price = os.environ.get("OPENAI_INPUT_PRICE_PER_MILLION_USD")
    output_price = os.environ.get("OPENAI_OUTPUT_PRICE_PER_MILLION_USD")
    if input_price is None or output_price is None:
        return None

    return (
        prompt_tokens * float(input_price) / 1_000_000
        + completion_tokens * float(output_price) / 1_000_000
    )


class OpenAIProvider(LLMProvider):
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en el entorno o .env.")

        self.model_id = os.environ.get("OPENAI_MODEL_ID", "gpt-5-nano")
        self.client = OpenAI(
            api_key=api_key,
            max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "0")),
            http_client=httpx.Client(
                timeout=httpx.Timeout(
                    connect=float(os.environ.get("OPENAI_CONNECT_TIMEOUT", "5.0")),
                    read=float(os.environ.get("OPENAI_READ_TIMEOUT", "15.0")),
                    write=float(os.environ.get("OPENAI_WRITE_TIMEOUT", "15.0")),
                    pool=float(os.environ.get("OPENAI_POOL_TIMEOUT", "15.0")),
                ),
            ),
        )

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 1,
        max_tokens: int = 1500,
        registro: int = 0,
    ) -> str | dict[str, Any]:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens debe ser un entero mayor que 0.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started_at = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        if registro != 1:
            return content

        usage = response.usage
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        cost_usd = _estimar_costo_openai(
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        call_record = {
            "proveedor": "openai",
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "costo_usd": cost_usd,
            "latencia_ms": latency_ms,
            "finnishReason": response.choices[0].finish_reason
        }
        registro_llamadas.append(call_record)

        return {
            "content": content,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            "costo_usd": cost_usd,
            "costo_usd_formateado": (
                f"${cost_usd:.8f}" if cost_usd is not None else None
            ),
            "latency_ms": latency_ms,
            "finnishReason": response.choices[0].finish_reason
        }

    def embeddings(self, textos: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(
            model=os.environ.get("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-large"),
            input=textos
        )
        return [
            item.embedding
            for item in sorted(resp.data, key=lambda item: item.index)
        ]