import Providers
import os
import time
from dotenv import load_dotenv
from ollama import chat, ResponseError
from concurrent.futures import ThreadPoolExecutor

#1. Cargue configuración desde variables de entorno, incluyendo `self.provider = os.environ.get("LLM_PROVIDER", "bedrock")` (ya resuelto en `__init__`).
#2. `generate(prompt, system=None, temperature=0.7, max_tokens=500)` llame a la función que corresponda **según `self.provider`** (`llamar_bedrock`, `llamar_anthropic`, `llamar_openai`, `llamar_google` o `llamar_ollama`, todas definidas en la sección 2) y devuelva **solo el texto** de la respuesta.
#3. Agregues manejo de reintentos ante rate limiting: si usás Bedrock, reutilizá `generar_con_reintentos` de la sección 8 tal cual; si usás otro proveedor, aplicá el mismo patrón (backoff exponencial) capturando la excepción de rate limit específica de tu SDK (`openai.RateLimitError` para OpenAI, `anthropic.RateLimitError` para Anthropic directo — para Google/Ollama alcanza con un `try/except Exception` genérico con backoff, no hace falta la excepción exacta).
#4. Que un error de validación (ej. `max_tokens` inválido) se relance con un mensaje claro, no un traceback críptico del SDK.
#5. (Opcional) Agregues un método `generate_batch(prompts)` que resuelva una lista de prompts de forma concurrente, usando el patrón de la sección 9.

### Criterio de éxito

#- `LLMClient().generate("¿Cuál es la capital de Francia?")` devuelve un string con la respuesta, sin lanzar excepción, **con el proveedor que hayas configurado en tu `.env`**.
#- Si se llama con un `max_tokens` inválido (ej. negativo), se obtiene un error claro y explicable, no un traceback críptico del SDK.
#- El cliente no tiene ninguna credencial hardcodeada — todo viene de `os.environ` / `.env`.
#- Si le mostrás tu código a un compañero que usa un proveedor distinto al tuyo, tiene que poder cambiar `LLM_PROVIDER` en su `.env` y que funcione sin tocar tu código (más allá de tener las funciones `llamar_*` de los otros proveedores disponibles).

class LLMClient:
    def __init__(self):
        load_dotenv()

    def generate(self, prompt: str, system: str = None, temperature: float = 0.7, max_tokens: int = 500) -> str:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens debe ser un entero mayor que 0.")

        if self.provider not in {"ollama", "bedrock", "anthropic", "openai", "google"}:
            raise ValueError(f"Proveedor no reconocido: {self.provider!r}.")

        max_reintentos=3
        for intento in range(max_reintentos):
            try:
                if self.provider == "ollama":
                    return Providers.llamar_ollama(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
                if self.provider == "bedrock":
                    return Providers.llamar_bedrock(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
                if self.provider == "anthropic":
                    return Providers.llamar_anthropic(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
                if self.provider == "openai":
                    return Providers.llamar_openai(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
                if self.provider == "google":
                    return Providers.llamar_google(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
                
            except Exception as e:
                if intento < max_reintentos - 1:
                    espera = 2 ** intento  # 1s, 2s, 4s...             
                    print(
                        f"Error de {self.provider} "
                        f"({getattr(e, 'status_code', 'sin código HTTP')}): {e}; "
                        f"reintentando en {espera}s..."
                    )   
                # Si no es el último intento, continuar al siguiente intento
                    time.sleep(espera)
                    continue
                else:
                    raise RuntimeError(f"Error al generar el texto con el proveedor {self.provider}: {e}") from e
   


# Prueba rápida
client = LLMClient()

client.model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
client.provider = os.environ.get("LLM_PROVIDER", "ollama")

try:
    respuesta = client.generate(
        "¿Cuál es la capital de Francia?", "Respondé en español, en un máximo de 3 oraciones, "
            "en tono directo y sin relleno. Utilizá bullets para enumerar las oraciones o los puntos clave." 
            "Ejemplo:\n- Primer punto\n- Segundo punto\n- Tercer punto""",0.9,900
    )

    print(respuesta)
except (ValueError, RuntimeError) as err:
    print(f"No se pudo generar la respuesta: {err}")