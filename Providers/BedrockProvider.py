from random import random
from time import time

import boto3
import os
from botocore.config import Config
import json

from LLMProvider import LLMProvider, generar_estructurado_con_reintentos

def llamar_bedrock(prompt, system=None, temperature=0.7, max_tokens=500):
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        config=Config(
            connect_timeout=float(os.environ.get("BEDROCK_MODEL_ID_CONNECT_TIMEOUT", 5.0)),
            read_timeout=float(os.environ.get("BEDROCK_MODEL_ID_READ_TIMEOUT", 15.0)),
            retries=json.loads(os.environ.get("BEDROCK_MODEL_ID_RETRIES", '{"max_attempts": 0}')),
        ),
    )
    kwargs = {"modelId": os.environ.get("BEDROCK_MODEL_ID_CLAUDE", "global.anthropic.claude-sonnet-4-6"),
              "messages": [{"role": "user", "content": [{"text": prompt}]}],
              "inferenceConfig": {"temperature": temperature, "maxTokens": max_tokens}}
    if system:
        kwargs["system"] = [{"text": system}]
    resp = client.converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]

class BedrockProvider(LLMProvider):
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                connect_timeout=float(os.environ.get("BEDROCK_MODEL_ID_CONNECT_TIMEOUT", 5.0)),
                read_timeout=float(os.environ.get("BEDROCK_MODEL_ID_READ_TIMEOUT", 15.0)),
                retries=json.loads(os.environ.get("BEDROCK_MODEL_ID_RETRIES", '{"max_attempts": 0}')),
            ),
        )
        self.model_id = os.environ.get("BEDROCK_MODEL_ID_CLAUDE", "global.anthropic.claude-sonnet-4-6")

    # Implementación del método generate para el proveedor Bedrock.
    # Este método utiliza la función llamar_bedrock para generar texto con el modelo Bedrock.
    def generate(self, prompt, system=None, temperature=0.7, max_tokens=500):
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens debe ser un entero mayor que 0.")

        if not hasattr(self, "provider") or not self.provider:
            self.provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()

        if self.provider not in {"ollama", "bedrock", "anthropic", "openai", "google"}:
            raise ValueError(f"Proveedor no reconocido: {self.provider!r}.")

        max_reintentos=3
        for intento in range(max_reintentos):
            try:
                if self.provider == "bedrock":
                    return llamar_bedrock(prompt, system=system, temperature=temperature, max_tokens=max_tokens)

                
            except Exception as e:
                if intento < max_reintentos - 1:
                    #espera = 2 ** intento  # 1s, 2s, 4s...     
                    espera = (2 ** intento) + random.uniform(0, 1)  # backoff exponencial + jitter        
                    print(
                        f"Error de {self.provider} "
                        f"({getattr(e, 'status_code', 'sin código HTTP')}): {e}; "
                        f"reintentando en {espera}s..."
                    )   
                # Si no es el último intento, continuar al siguiente intento
                    time.sleep(espera)
                    continue
                else:
                    raise RuntimeError(f"Error al generar el texto con el proveedor {self.provider}: {e}") from e


    # Implementación del método generate_structured para el proveedor Bedrock.
    # Este método construye la solicitud con el esquema JSON y llama al cliente de Bedrock para obtener la respuesta estructurada.
    def generate_structured(self, prompt, schema_model, system=None):
        # Override: usa el mecanismo nativo de Bedrock en vez del reintento genérico
        schema = schema_model.model_json_schema()
        schema["additionalProperties"] = False
        kwargs = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "outputConfig": {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
                "name": schema_model.__name__, "description": schema_model.__doc__ or "", "schema": json.dumps(schema),
            }}}},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        resp = self.client.converse(**kwargs)
        raw = resp["output"]["message"]["content"][0]["text"]
        return schema_model.model_validate_json(raw)

    