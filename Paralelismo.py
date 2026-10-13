import time
from concurrent.futures import ThreadPoolExecutor
from ollama import chat

preguntas = [
    "Definí 'token' en una oración.",
    "Definí 'embedding' en una oración.",
    "Definí 'temperatura' en una oración.",
    "Definí 'rag' en una oración.",
    "Definí 'reranking' en una oración.",
    "Definí 'greedy decoding' en una oración.",
]

def preguntar(pregunta):
    resp = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": pregunta
            }
        ]
    )
    return resp["message"]["content"]

# Secuencial
t0 = time.time()
resultados_seq = [preguntar(p) for p in preguntas]
t_seq = time.time() - t0

# Concurrente (3 llamadas en paralelo)
t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as pool:
    resultados_par = list(pool.map(preguntar, preguntas))
t_par = time.time() - t0

print(f"Secuencial: {t_seq:.2f}s")
print(f"Concurrente: {t_par:.2f}s")
print(f"Aceleración: {t_seq/t_par:.2f}x")

print("\nResultados:")
for r in resultados_par:
    print("-", r)


# Resultados de ejemplo
# Secuencial: 53.57s
# Concurrente: 28.87s
# Aceleración: 1.86x

#Resultados:
#- **Token** es una unidad básica en un lenguaje de programación que representa un elemento sintáctico, como una palabra clave, un operador o un símbolo.
##- **Definición en una oración:**  
#"El *embedding* es una representación numérica de palabras, frases o datos que captura su significado y contexto para facilitar su procesamiento en sistemas de inteligencia artificial."  

##Este concepto es clave en el procesamiento del lenguaje natural (NLP), donde las palabras se convierten en vectores (arrays de números) que reflejan su relación semántica con otras palabras.
#- La temperatura es la medida de la energía térmica de un cuerpo, que indica el grado de calor o frío que posee.