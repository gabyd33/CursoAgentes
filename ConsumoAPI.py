from ollama import chat

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "Sos un asistente técnico para una certificación de desarrollo "
                "de agentes de IA. Respondé en español, en un máximo de 3 "
                "oraciones, en tono directo y sin relleno."
            ),
        },
        {
            "role": "user",
            "content": "¿qué es BPE, para IA?"
        }
    ]
)

texto = response["message"]["content"]

print()
print("--------------------------------------------------------------------------------")
print(texto)
print()
print("Uso de tokens (entrada/salida):",
      response.get("prompt_eval_count"),
      "/",
      response.get("eval_count"))
print("Motivo de fin:", response.get("done_reason"))
print("Latencia total (ms):",
      response.get("total_duration", 0) / 1_000_000)
print("--------------------------------------------------------------------------------")


