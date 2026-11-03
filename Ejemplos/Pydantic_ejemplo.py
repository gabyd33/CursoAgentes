import json

# %pip install pydantic  # si hace falta instalarlo

print("-------------------------------------------")
print("Inicio")
print("-------------------------------------------")

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class ConsultaSoporte(BaseModel):
    categoria: Literal["facturacion", "tecnico", "cuenta", "otro"]
    urgencia: Literal["baja", "media", "alta"]
    resumen: str = Field(..., max_length=200, description="Resumen breve del reclamo, en una oración")
    requiere_escalamiento: bool = Field(..., description="Indica si la consulta requiere escalamiento") 


# El JSON Schema se genera solo a partir de la clase:
schema = ConsultaSoporte.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))

print("-------------------------------------------")
print("-------------------------------------------")

# Validación: un dict válido pasa, uno inválido lanza ValidationError con el detalle exacto de qué falló
dato_valido = {"categoria": "tecnico", "urgencia": "alta", "resumen": "El servicio no responde desde hace 2 horas.", "requiere_escalamiento": True}
consulta = ConsultaSoporte(**dato_valido)
print(consulta)

try:
    consulta2 = ConsultaSoporte(categoria="inventado", urgencia="alta", resumen="x", requiere_escalamiento=False)
except ValidationError as e:
    print("\nError de validación (esperado):")
    print(e)


print("-------------------------------------------")
print("Fin")
print("-------------------------------------------")