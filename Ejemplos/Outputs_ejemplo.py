from openai import OpenAI  # Importa el cliente de OpenAI.
from pydantic import BaseModel, ConfigDict, Field  # Importa herramientas para definir y validar datos.
from typing import Literal  # Permite limitar un campo a opciones concretas.
import os
from dotenv import load_dotenv

load_dotenv()  # Carga las variables del archivo .env.



class ConsultaSoporte(BaseModel):  # Define la estructura que debe tener la respuesta.
    categoria: Literal["facturacion", "tecnico", "cuenta", "otro"]  # Acepta solo una de estas categorías.
    urgencia: Literal["baja", "media", "alta"]  # Acepta solo estos niveles de urgencia.
    resumen: str = Field(..., max_length=200, description="Resumen breve del reclamo, en una oración")  # Exige un resumen de hasta 200 caracteres.
    model_config = ConfigDict(extra="forbid")  # Rechaza campos que no estén definidos en esta clase.


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # Usa la clave configurada en .env.


texto_libre = (  # Guarda un ejemplo de consulta urgente.
    "Hace dos días que no puedo entrar a mi cuenta y ya perdí "
    "una reunión importante por esto, necesito que lo resuelvan YA."
)

texto_libre_bajaUrgencia = (  # Guarda otro ejemplo, marcado como poco urgente.
    "hace 10minutos no puedo recordar la contraseña, pero no es urgente. "
)


completion = client.chat.completions.create(  # Envía la consulta al modelo y recibe su respuesta.
    model=os.environ.get("OPENAI_MODEL_ID", "gpt-5-nano"),  # Usa el modelo configurado en .env.
    messages=[  # Contiene los mensajes de la conversación.
        {
            "role": "user",  # Indica que el mensaje lo envía el usuario.
            "content": f"Clasificá esta consulta de soporte: '{texto_libre_bajaUrgencia}'"  # Le pide clasificar la consulta.
        }
    ],
    response_format={  # Solicita que la respuesta siga un formato estructurado.
        "type": "json_schema",  # Indica que se usará un esquema JSON.
        "json_schema": {
            "name": "consulta_soporte",  # Le da un nombre al esquema.
            "strict": True,  # Exige que la respuesta respete el esquema.
            "schema": ConsultaSoporte.model_json_schema(),  # Obtiene el esquema desde la clase Pydantic.
        }
    }
)

raw_json = completion.choices[0].message.content  # Extrae el JSON generado por el modelo.

print("JSON crudo devuelto por el modelo:", raw_json)  # Muestra el JSON como texto.

consulta_validada = ConsultaSoporte.model_validate_json(raw_json)  # Valida el JSON y lo convierte en un objeto Pydantic.

print("\nObjeto Python validado:", consulta_validada)  # Muestra el objeto ya validado.