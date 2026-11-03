import re
import unicodedata

def limpiar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto)           # normaliza comillas/acentos
    texto = re.sub(r"[ \t]+", " ", texto)                   # espacios repetidos
    texto = re.sub(r"\n{3,}", "\n\n", texto)                # más de 2 saltos de línea seguidos
    texto = re.sub(r"Página\s+(\d+)\s+de\s+(\d+)", "", texto, flags=re.MULTILINE)  # elimina líneas de boilerplate "Página X de Y"
    return texto.strip()


texto_sucio = """   Página 1 de 3


Guía  de   configuración


Para configurar  las credenciales,   copiá .env.example a .env.\n\n\n\nCompletá la sección de tu proveedor.
"""
print(texto_sucio)
print("---------------------------")
print(repr(limpiar_texto(texto_sucio)))




#Practicá: agregá una regla a limpiar_texto que detecte y elimine una línea de boilerplate recurrente (ej. "Página X de Y" con una regex) — es el tipo de ruido que vas a encontrar en PDFs reales del corpus del TP.