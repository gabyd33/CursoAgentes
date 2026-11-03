import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL_ID = os.environ.get("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-large")


# Función para obtener embeddings de OpenAI
def embed_openai(textos: list[str]) -> list[list[float]]:
    """Devuelve un embedding de OpenAI por cada texto, en el mismo orden."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno o .env.")
    if not textos:
        return []

    client = OpenAI(api_key=api_key)
    # Crear embeddings usando el modelo especificado 
    response = client.embeddings.create(model=EMBED_MODEL_ID, input=textos)
    return [
        # Ordenar los embeddings según el índice original para mantener el orden de los textos
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]


def similitud_coseno(a: list[float], b: list[float]) -> float:
    producto_punto = sum(x * y for x, y in zip(a, b))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(y * y for y in b))
    return producto_punto / (norma_a * norma_b)

frases = [
    "La ventana de contexto limita cuánto texto puede procesar un modelo por llamada.",
    "El límite de tokens de entrada y salida define cuánta información cabe en una consulta.",
    "Ualá es una fintech argentina que ofrece una tarjeta prepaga y una cuenta digital.",
    "Para hacer una salsa de tomate se requiere tomate, sal y pimienta.",
]

vectores = embed_openai(frases)
print(f"Dimensión de cada embedding: {len(vectores[0])}")

print("\nSimilitud entre frase 1 y 2 (ambas sobre contexto/tokens):", round(similitud_coseno(vectores[0], vectores[1]), 3))
print("Similitud entre frase 1 y 3 (temas no relacionados):       ", round(similitud_coseno(vectores[0], vectores[2]), 3))
print("Similitud entre frase 1 y 4 (temas no relacionados):       ", round(similitud_coseno(vectores[0], vectores[3]), 3))
print("Similitud entre frase 2 y 3 (temas no relacionados):       ", round(similitud_coseno(vectores[1], vectores[2]), 3))



def embed_openai(textos: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=os.environ.get("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-large"), input=textos)
    return [d.embedding for d in resp.data]