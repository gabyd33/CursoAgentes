# --- AWS Bedrock ---
import boto3
import os
from importlib import import_module

def llamar_bedrock(prompt, system=None, temperature=0.7, max_tokens=500):
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    kwargs = {"modelId": os.environ.get("BEDROCK_MODEL_ID_CLAUDE", "global.anthropic.claude-sonnet-4-6"),
              "messages": [{"role": "user", "content": [{"text": prompt}]}],
              "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens}}
    if system:
        kwargs["system"] = [{"text": system}]
    resp = client.converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]

# --- Anthropic directo ---
def llamar_anthropic(prompt, system=None, temperature=0.7, max_tokens=500):
    anthropic = import_module("anthropic")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
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
def llamar_openai(prompt, system=None, temperature=0.7, max_tokens=500):
    openai = import_module("openai")
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_ID", "gpt-4o"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except openai.RateLimitError:
        raise
    except openai.APIStatusError as e:
        raise RuntimeError(f"Error de la API de OpenAI: {e.status_code}") from e


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

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
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

def llamar_ollama(prompt, system=None, temperature=0.7, max_tokens=500):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=os.environ.get("OLLAMA_MODEL_ID", "qwen3:8b"),
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    if resp["done_reason"] == "length":
        print(
            f"Advertencia: la respuesta se detuvo al alcanzar el máximo "
            f"de {max_tokens} tokens."
        )
    return resp["message"]["content"]