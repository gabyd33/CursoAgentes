# La "herramienta" real: una función Python común y corriente
def consultar_estado_ticketv2(ticket_id: str) -> dict:
    tickets_mock = {"T-001": "En progreso", "T-002": "Resuelto", "T-003": "Esperando al cliente"}
    return {"ticket_id": ticket_id, "estado": tickets_mock.get(ticket_id, "No encontrado")}

def escalar_ticket(ticket_id : str, motivo: str) -> dict:
    return {"ticket_id": ticket_id, "motivo": motivo, "estado": "Escalado"}


# La descripción de esa función que le mandamos al modelo (JSON Schema de sus argumentos)
#tool_config = {
#    "tools": [{
#        "toolSpec": {
#            "name": "consultar_estado_ticket",
#            "description": "Consulta el estado actual de un ticket de soporte por su ID.",
#            "inputSchema": {"json": {
#                "type": "object",
#                "properties": {
#                    "ticket_id": {"type": "string", "description": "ID del ticket, ej. T-001"}
#                },
#                "required": ["ticket_id"],
#            }},
#        }
#    }]
#}

import json
from Ejemplos.LLMClient import LLMClient

client = LLMClient(provider="openai")

tools = [
    {
        "type": "function",
        "function": {
            "name": "consultar_estado_ticketv2",
            "description": "Obtiene el estado de un ticket de soporte",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Identificador del ticket"
                    }
                },
                "required": ["ticket_id"]
            }
        }
    }
]
    

messages = [
    {
        "role": "user",
        "content": "¿Cuál es el estado del ticket T-001?, si esta en progreso por favor escalar con motivo 'Necesita atención urgente'"
    }
]

response = client.generate_with_tools(
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message

messages.append(message)

print("Tool calls:", message.tool_calls)

if message.tool_calls:

    for tool_call in message.tool_calls:

        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(
            f"El modelo pide llamar a: "
            f"{function_name}({function_args})"
        )

        if function_name == "consultar_estado_ticketv2":
            resultado = consultar_estado_ticketv2(**function_args)
        elif function_name == "escalar_ticket":
            resultado = escalar_ticket(**function_args)
        else:
            resultado = {
                "error": (
                    f"La herramienta '{function_name}' está declarada, "
                    "pero no tiene una implementación."
                )
            }

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(resultado, ensure_ascii=False),
            }
        )

    # Segunda llamada:
    # devolvemos el resultado de la herramienta para que genere la respuesta final
    response_final = client.generate_with_tools(
        messages=messages,
        tools=tools,
    )

    print(
        "\nRespuesta final:",
        response_final.choices[0].message.content,
    )

else:

    print(message.content)