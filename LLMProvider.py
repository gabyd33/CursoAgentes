from abc import ABC, abstractmethod
import json
import re
from time import time
from typing import Any
import unicodedata

import Chunk

#no depende de SDKs concretos. Recibe una función generate, le pasa el prompt y valida el texto resultante con el modelo Pydantic; si falla la validación, vuelve a intentarlo.
def generar_estructurado_con_reintentos(generate, prompt: str, schema_model, system: str = None):
    """Genera texto, reintenta si no valida y devuelve una instancia Pydantic."""
    if hasattr(schema_model, "model_json_schema"):
        schema = schema_model.model_json_schema()
    else:
        schema = schema_model.schema()

    request = (
        f"{prompt}\n\nDevuelve únicamente JSON válido que cumpla este esquema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
    last_error = None

    for attempt in range(3):
        response = generate(request, system=system)
        content = response.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3].strip()

        try:
            if hasattr(schema_model, "model_validate_json"):
                return schema_model.model_validate_json(content)
            return schema_model.parse_raw(content)
        except Exception as error:
            last_error = error
            request = (
                f"{prompt}\n\nLa respuesta anterior no cumplió el esquema. "
                f"Error: {error}. Inténtalo de nuevo y devuelve únicamente JSON válido "
                f"que cumpla este esquema:\n{json.dumps(schema, ensure_ascii=False)}"
            )

    raise last_error





class LLMProvider(ABC):
    """Interfaz común para cualquier proveedor de modelos de lenguaje."""

    @abstractmethod
    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500, registro: int = 0) -> str | dict[str, Any]:
        """Devuelve texto o texto junto con métricas cuando registro está activo."""
        # to do: Implementar la generación de texto en la subclase concreta.
        raise NotImplementedError("Este método debe ser implementado en la subclase concreta.")

    @abstractmethod
    def embeddings(self, textos: list[str]) -> list[list[float]]:
        raise NotImplementedError("Este método debe ser implementado en la subclase concreta.")
    
    def generate_structured(self, prompt, schema_model, system=None):
        return generar_estructurado_con_reintentos(
            self.generate,
            prompt,
            schema_model,
            system=system,
        )

    @staticmethod
    def limpiar_texto(texto: str) -> str:
        texto = unicodedata.normalize("NFKC", texto)           # normaliza comillas/acentos
        texto = re.sub(r"[ \t]+", " ", texto)                   # espacios repetidos
        texto = re.sub(r"\n{3,}", "\n\n", texto)                # más de 2 saltos de línea seguidos
        texto = re.sub(r"Página\s+(\d+)\s+de\s+(\d+)", "", texto, flags=re.MULTILINE)  # elimina líneas de boilerplate "Página X de Y"
        return texto.strip()

    @staticmethod
    def construir_chunks_por_caracteres(texto: str, documento_id: str, tamano: int = 300, solapamiento: int = 50, fuente: str = "") -> list[Chunk.Chunk]:
        textos = LLMProvider.chunking_por_caracteres(texto, tamano, solapamiento)
        return [Chunk.Chunk(texto=t, documento_id=documento_id, posicion=i, fuente=fuente, fecha_ingesta=time()) for i, t in enumerate(textos)]

    @staticmethod
    def construir_chunks_por_parrafos(texto: str, documento_id: str, max_chars: int = 500, solapamiento: int = 50, fuente: str = "") -> list[Chunk.Chunk]:
        textos = LLMProvider.chunking_por_parrafos(texto, max_chars, solapamiento)
        return [Chunk.Chunk(texto=t, documento_id=documento_id, posicion=i, fuente=fuente, fecha_ingesta=time()) for i, t in enumerate(textos)]

    @staticmethod
    def chunking_por_caracteres(texto: str, tamano: int = 300, solapamiento: int = 50) -> list[str]:
        """Chunking de tamaño fijo con solapamiento, por caracteres (simple, no respeta oraciones)."""
        chunks = []
        paso = tamano - solapamiento
        for inicio in range(0, len(texto), paso):
            chunk = texto[inicio:inicio + tamano]
            if chunk.strip():
                chunks.append(chunk)
            if inicio + tamano >= len(texto):
                break
            
        return chunks


    @staticmethod
    def chunking_por_parrafos(texto: str, max_chars: int = 500, solapamiento: int = 50) -> list[str]:
        """Agrupa párrafos (separados por línea en blanco) hasta llegar al tamaño máximo."""
        parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        chunks, actual = [], ""
        for p in parrafos:
            if len(actual) + len(p) + 1 <= max_chars:
                actual = (actual + "\n\n" + p).strip()
            else:
                if actual:
                    chunks.append(actual)
                # aplicar solapamiento
                if solapamiento > 0 and len(actual) > solapamiento:
                    actual = actual[-solapamiento:]
                else:
                    actual = ""
                actual = (actual + "\n\n" + p).strip()
        if actual:
            chunks.append(actual)
        return chunks


