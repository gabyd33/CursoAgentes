from importlib import import_module
from random import random
from time import time

import boto3
import os
import httpx
from botocore.config import Config
import json

from LLMProvider import LLMProvider, generar_estructurado_con_reintentos

class AnthropicProvider(LLMProvider):
    def __init__(self):
        anthropic = import_module("anthropic")
        self.model_id = os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6")
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "0")),
            timeout=httpx.Timeout(
                connect=float(os.environ.get("ANTHROPIC_CONNECT_TIMEOUT", "5.0")),
                read=float(os.environ.get("ANTHROPIC_READ_TIMEOUT", "15.0")),
                write=float(os.environ.get("ANTHROPIC_WRITE_TIMEOUT", "15.0")),
                pool=float(os.environ.get("ANTHROPIC_POOL_TIMEOUT", "15.0")),
            ),
        )

    def generate(self, prompt, system=None, temperature=0.7, max_tokens=500):
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens debe ser un entero mayor que 0.")

        kwargs = {"model": os.environ.get("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6"),
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        # Nota: en algunos modelos recientes de Anthropic, `temperature` no es configurable
        # (devuelve error 400 si se manda con un valor distinto del default) — omitirlo si falla.
        resp = self.client.messages.create(**kwargs)
        return resp.content[0].text
