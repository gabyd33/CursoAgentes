# --- AWS Bedrock ---
import time
from typing import Any
from urllib import response

import boto3
import os
import httpx
from botocore.config import Config
from dotenv import load_dotenv

from importlib import import_module

HTTP_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=15.0,
    write=15.0,
    pool=15.0,
)


def _openai_timeout():
    return httpx.Timeout(
        connect=float(os.environ.get("OPENAI_CONNECT_TIMEOUT", "5.0")),
        read=float(os.environ.get("OPENAI_READ_TIMEOUT", "15.0")),
        write=float(os.environ.get("OPENAI_WRITE_TIMEOUT", "15.0")),
        pool=float(os.environ.get("OPENAI_POOL_TIMEOUT", "15.0")),
    )

# Valores de ejemplo — el precio real cambia con el tiempo, verificar en la consola de Bedrock antes de usar en producción
PRECIO_POR_MILLON_USD = {"input": 3.0, "output": 15.0}

registro_llamadas = []  # en una app real, esto va a un log estructurado o una base de datos (Clase 11)

def estimar_costo_usd(usage: dict, proveedor: str = "bedrock") -> float | None:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()

    input_tokens = usage.get("inputTokens", usage.get("prompt_tokens", usage.get("input_tokens")))
    output_tokens = usage.get("outputTokens", usage.get("completion_tokens", usage.get("output_tokens")))
    if input_tokens is None or output_tokens is None:
        raise ValueError("usage debe incluir tokens de entrada y salida.")

    if proveedor == "openai":
        input_price = os.environ.get("OPENAI_INPUT_PRICE_PER_MILLION_USD")
        output_price = os.environ.get("OPENAI_OUTPUT_PRICE_PER_MILLION_USD")
        if input_price is None or output_price is None:
            return None
        precios = {"input": float(input_price), "output": float(output_price)}
    else:
        precios = PRECIO_POR_MILLON_USD

    costo_input = input_tokens / 1_000_000 * precios["input"]
    costo_output = output_tokens / 1_000_000 * precios["output"]
    return costo_input + costo_output
    



# --- Anthropic directo ---
def llamar_anthropic(prompt, system=None, temperature=0.7, max_tokens=500):
    anthropic = import_module("anthropic")
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "0")),
        timeout=httpx.Timeout(
            connect=float(os.environ.get("ANTHROPIC_CONNECT_TIMEOUT", "5.0")),
            read=float(os.environ.get("ANTHROPIC_READ_TIMEOUT", "15.0")),
            write=float(os.environ.get("ANTHROPIC_WRITE_TIMEOUT", "15.0")),
            pool=float(os.environ.get("ANTHROPIC_POOL_TIMEOUT", "15.0")),
        ),
    )
    kwargs = {"model": os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6"),
              "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    # Nota: en algunos modelos recientes de Anthropic, `temperature` no es configurable
    # (devuelve error 400 si se manda con un valor distinto del default) — omitirlo si falla.
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


# --- OpenAI ---
def llamar_openai(
    prompt=None,
    system=None,
    temperature=0.7,
    max_tokens=500,
    registro=0,
    messages=None,
    tools=None,
    tool_choice=None,
):
    load_dotenv()
    openai = import_module("openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno o .env.")

    client = openai.OpenAI(
        api_key=api_key,
        max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "0")),
        http_client=httpx.Client(timeout=_openai_timeout()),
    )
    request_messages = messages
    if request_messages is None:
        request_messages = []
        if system:
            request_messages.append({"role": "system", "content": system})
        request_messages.append({"role": "user", "content": prompt})

    request_kwargs = {
        "model": os.environ.get("OPENAI_MODEL_ID", "gpt-5.4-mini"),
        "messages": request_messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if tools is not None:
        request_kwargs["tools"] = tools
    if tool_choice is not None:
        request_kwargs["tool_choice"] = tool_choice
    datos_registro = None
    try:
        inicio = time.perf_counter()
        resp = client.chat.completions.create(**request_kwargs)
        if( registro == 1 ):
            latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
            datos_registro = {
                "proveedor": "openai",
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
                "costo_usd": estimar_costo_usd(resp.usage, proveedor="openai"),
                "latencia_ms": latencia_ms,
            }
            registro_llamadas.append(datos_registro)
        
    except openai.RateLimitError:
        raise
    except openai.APIStatusError as e:
        raise RuntimeError(f"Error de la API de OpenAI ({e.status_code}): {e}") from e


    if tools is not None:
        return resp
    content = resp.choices[0].message.content
    if datos_registro is not None:
        return {
            "content": content,
            "usage": {
                "input_tokens": datos_registro["input_tokens"],
                "output_tokens": datos_registro["output_tokens"],
                "total_tokens": datos_registro["total_tokens"],
            },
            "costo_usd_formateado": (
                f"${datos_registro['costo_usd']:.8f}"
                if datos_registro["costo_usd"] is not None
                else None
            ),
            "latency_ms": datos_registro["latencia_ms"],
        }
    return content


def llamar_openai_stream(prompt, system=None, temperature=0.7, max_tokens=500):
    load_dotenv()
    openai = import_module("openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno o .env.")

    client = openai.OpenAI(
        api_key=api_key,
        max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "0")),
        http_client=httpx.Client(timeout=_openai_timeout()),
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_ID", "gpt-5.4-mini"),
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=True,
    )
    for event in stream:
        if event.choices:
            text = event.choices[0].delta.content
            if text:
                yield text


# --- Google Gemini ---
try:
    genai = import_module("google.genai")
    types = import_module("google.genai.types")
except ImportError:  # La librería puede no estar instalada en algunos entornos
    genai = None
    types = None


def llamar_google(prompt, system=None, temperature=0.7, max_tokens=500):
    if genai is None or types is None:
        raise ModuleNotFoundError("Instala la librería 'google-genai' para usar Gemini.")

    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=types.HttpOptions(
            timeout=15_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    resp = client.models.generate_content(
        model=os.environ.get("GOOGLE_MODEL_ID", "gemini-2.0-flash"),
        contents=prompt,
        config=config,
    )
    return resp.text



# --- Ollama (modelo local) ---
import ollama

def llamar_ollama(prompt, system=None, temperature=0.7, max_tokens=500, schema: dict[str, Any] = None):
    client = ollama.Client(
        timeout=HTTP_TIMEOUT,
        transport=httpx.HTTPTransport(retries=0),
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": os.environ.get("OLLAMA_MODEL_ID", "qwen3:8b"),
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if schema is not None:
        kwargs["format"] = schema

    resp = client.chat(**kwargs)
    if resp.get("done_reason") == "length":
        print(
            f"Advertencia: la respuesta se detuvo al alcanzar el máximo "
            f"de {max_tokens} tokens."
        )
    return resp["message"]["content"]
