from ollama import chat, ResponseError
import time


def generar_con_reintentos(
    model,
    messages,
    options=None,
    max_reintentos=3
):
    """Llama a Ollama con reintentos y backoff exponencial."""

    for intento in range(max_reintentos):
        try:
            return chat(
                model=model,
                messages=messages,
                options=options or {}
            )

        except ResponseError as err:
            if intento < max_reintentos - 1:
                espera = 2 ** intento  # 1s, 2s, 4s...
                print(
                    f"Error de Ollama ({err.status_code}), "
                    f"reintentando en {espera}s..."
                )
                time.sleep(espera)
                continue

            raise

        except Exception as err:
            if intento < max_reintentos - 1:
                espera = 2 ** intento
                print(
                    f"Error inesperado: {err}. "
                    f"Reintentando en {espera}s..."
                )
                time.sleep(espera)
                continue

            raise

    raise RuntimeError("Se agotaron los reintentos.")


# Prueba
resp = generar_con_reintentos(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": "Decí 'ok' y nada más."
        }
    ]
)

print(resp["message"]["content"])