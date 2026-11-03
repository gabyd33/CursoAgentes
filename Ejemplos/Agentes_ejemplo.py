# --- Ollama (modelo local) ---
import ollama
import os

def llamar_ollama(prompt, system=None, temperature=0.7, max_tokens=500):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=os.environ.get("OLLAMA_MODEL_ID", "qwen2.5-coder:7b"),
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    texto = resp["message"]["content"]
    print("Respuesta de Ollama:", texto)
    return texto


# Ejemplo de uso
if __name__ == "__main__":
    prompt = "¿Decime un chiste?"
    respuesta = llamar_ollama(prompt)
