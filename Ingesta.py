import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parent
GUIDE_DIR = ROOT_DIR / "MDN-JavaScript-Guide"
OUTPUT_PATH = ROOT_DIR / "chunk_vectors.json"
BATCH_SIZE = 100

sys.path.insert(0, str(ROOT_DIR / "Providers"))

from LLMProvider import LLMProvider
from OpenAIProvider import OpenAIProvider


@lru_cache(maxsize=1)
def _openai_provider() -> OpenAIProvider:
    return OpenAIProvider()


def embed_openai(textos: list[str]) -> list[list[float]]:
    return _openai_provider().embeddings(textos)


def pipeline_ingesta(
    documentos: dict[str, str],
    tamano: int = 300,
    solapamiento: int = 50,
    embed_fn: Callable[[list[str]], list[list[float]]] = embed_openai,
) -> list[dict]:
    """Limpia documentos, crea chunks y devuelve cada chunk con su embedding."""
    if tamano < 1 or solapamiento < 0 or solapamiento >= tamano:
        raise ValueError("Se requiere tamano > 0 y 0 <= solapamiento < tamano.")

    chunks = []
    for documento_id, texto_crudo in documentos.items():
        texto = LLMProvider.limpiar_texto(texto_crudo)
        if not texto:
            continue

        chunks.extend(
            LLMProvider.construir_chunks_por_parrafos(
                texto,
                documento_id=documento_id,
                max_chars=tamano,
                solapamiento=solapamiento,
                fuente=documento_id,
            )
        )

    resultados = []
    for inicio in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[inicio:inicio + BATCH_SIZE]
        vectores = embed_fn([chunk.texto for chunk in lote])
        if len(vectores) != len(lote):
            raise RuntimeError("La función de embeddings devolvió una cantidad inesperada de vectores.")

        for chunk, vector in zip(lote, vectores):
            resultados.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "texto": chunk.texto,
                    "vector": vector,
                    "documento_id": chunk.documento_id,
                    "posicion": chunk.posicion,
                    "fuente": chunk.fuente,
                    "fecha_ingesta": chunk.fecha_ingesta,
                }
            )

    return resultados


def cargar_documentos(carpeta: Path) -> dict[str, str]:
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {carpeta}")

    return {
        archivo.relative_to(carpeta).with_suffix("").as_posix(): archivo.read_text(encoding="utf-8")
        for archivo in sorted(carpeta.rglob("*.md"))
    }


def main() -> None:
    documentos = cargar_documentos(GUIDE_DIR)
    resultados = pipeline_ingesta(documentos, tamano=1400, solapamiento=100)
    if not resultados:
        raise RuntimeError(f"No se generaron chunks desde {GUIDE_DIR}")

    OUTPUT_PATH.write_text(json.dumps(resultados, ensure_ascii=False), encoding="utf-8")
    print(f"Procesados {len(resultados)} chunks de {GUIDE_DIR}")
    print(f"Embeddings guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
