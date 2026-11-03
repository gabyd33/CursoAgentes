from ollama import chat

messages = [
    {
        "role": "system",
        "content": (
            "SOLO responder utilizando la información del contexto. "
            "Si la respuesta no está en el contexto, decí explícitamente que no tenés esa información. "
            "Sos un asistente técnico para una certificación de desarrollo "
            "de agentes de IA. Respondé en español, en un máximo de 3 oraciones, "
            "en tono directo y sin relleno. Utilizá bullets para enumerar las oraciones o los puntos clave." 
            "Ejemplo:\n- Primer punto\n- Segundo punto\n- Tercer punto"
            "<contexto>"
                "El programa tiene 12 clases semanales de 3 horas cada una, modalidad virtual."
                "El Trabajo Práctico Final Integrador es único para toda la cohorte: un sistema RAG"
                "más un flujo con agentes, desplegado vía API."
            "</contexto>"
            

        )
    },
    {
        "role": "user",
        "content": "Estoy comparando Claude y un modelo abierto para mi TP. ¿Qué debería tener en cuenta?"
    }
]

# Turno 1
resp1 = chat(
    model="qwen3:8b",
    messages=messages
)

print("Turno 1:", resp1["message"]["content"])

# Agregamos la respuesta al historial
messages.append(resp1["message"])

# Nuevo turno del usuario
messages.append({
    "role": "user",
    "content": "Y de esos criterios, ¿cuál pesa más si el corpus tiene datos sensibles?"
})

# Turno 2
resp2 = chat(
    model="qwen3:8b",
    messages=messages
)

print("\nTurno 2:", resp2["message"]["content"])