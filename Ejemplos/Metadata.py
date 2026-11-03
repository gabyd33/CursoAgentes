import hashlib
from dataclasses import dataclass, field
import time
import re
import unicodedata

def limpiar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFKC", texto)           # normaliza comillas/acentos
    texto = re.sub(r"[ \t]+", " ", texto)                   # espacios repetidos
    texto = re.sub(r"\n{3,}", "\n\n", texto)                # más de 2 saltos de línea seguidos
    texto = re.sub(r"Página\s+(\d+)\s+de\s+(\d+)", "", texto, flags=re.MULTILINE)  # elimina líneas de boilerplate "Página X de Y"
    return texto.strip()


def chunking_por_caracteres(texto: str, tamano: int = 300, solapamiento: int = 50) -> list[str]:
    """Chunking de tamaño fijo con solapamiento, por caracteres (simple, no respeta oraciones)."""
    chunks = []
    paso = tamano - solapamiento
    for inicio in range(0, len(texto), paso):
        chunk = texto[inicio:inicio + tamano]
        if chunk.strip():
            chunks.append(chunk)
        if inicio + tamano >= len(texto):
            break
    return chunks


def chunking_por_parrafos(texto: str, max_chars: int = 500, solapamiento: int = 50) -> list[str]:
    """Agrupa párrafos (separados por línea en blanco) hasta llegar al tamaño máximo."""
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    chunks, actual = [], ""
    for p in parrafos:
        if len(actual) + len(p) + 1 <= max_chars:
            actual = (actual + "\n\n" + p).strip()
        else:
            if actual:
                chunks.append(actual)
            # aplicar solapamiento
            if solapamiento > 0 and len(actual) > solapamiento:
                actual = actual[-solapamiento:]
            else:
                actual = ""
            actual = (actual + "\n\n" + p).strip()
    if actual:
        chunks.append(actual)
    return chunks


@dataclass
class Chunk:
    texto: str
    documento_id: str
    posicion: int           # orden del chunk dentro del documento
    chunk_id: str = field(init=False)
    fuente: str #como guardo este valor?    
    fecha_ingesta: float  # con un timestamp

    def __post_init__(self):
        # id determinístico: mismo documento + misma posición -> mismo id, siempre
        #para que sirve esto?
        # Esto asegura que cada chunk tenga un identificador único basado en su contenido y posición dentro del documento.
        # Así, si el mismo texto se procesa nuevamente, se generará el mismo chunk_id, lo que facilita la deduplicación y el seguimiento de chunks.
        # que es self.posicion:03d
        # El :03d formatea el número de posición con 3 dígitos, rellenando con ceros a la izquierda si es necesario.
        # Por ejemplo, si la posición es 5, se formateará como "005". Esto asegura que los IDs de los chunks tengan un formato consistente.
        hash_contenido = hashlib.sha256(self.texto.encode()).hexdigest()[:8]
        self.chunk_id = f"{self.documento_id}-{self.posicion:03d}-{hash_contenido}"


def construir_chunks(texto: str, documento_id: str, tamano: int = 300, solapamiento: int = 50, fuente: str = "",) -> list[Chunk]:
    textos = chunking_por_caracteres(texto, tamano, solapamiento)
    return [Chunk(texto=t, documento_id=documento_id, posicion=i, fuente=fuente, fecha_ingesta=time.time()) for i, t in enumerate(textos)]


corpus_ejemplo = """La Converse API de Bedrock es una interfaz unificada para conversación. Funciona igual para
distintos modelos: cambia el modelId, no la forma de llamar. Soporta mensajes de sistema,
parámetros de inferencia como temperatura y top_p, y tool calling con un schema JSON de argumentos.

Para usarla se necesita un cliente de boto3 apuntando al servicio bedrock-runtime, y credenciales
configuradas vía variables de entorno o un perfil de AWS."""

chunks = construir_chunks(limpiar_texto(corpus_ejemplo), documento_id="doc-converse-api", fuente="ejemploFuente")
for c in chunks:
   # print(f"[{c.chunk_id}] (pos {c.posicion}) {c.texto[:60]}..." f" Fuente: {c.fuente} Fecha ingesta: {c.fecha_ingesta}")


    documento_largo = corpus_ejemplo * 3  # simulamos un documento más largo repitiendo el de ejemplo

# Probamos diferentes tamaños de chunk y solapamientos para ver cómo afecta al número de chunks generados.
#overlap indica cuántos caracteres se repiten entre chunks consecutivos.
#cada chunk tendrá como máximo 'tamano' caracteres, y se solapará 'overlap' caracteres con el siguiente chunk.
#que quiere decir solapamiento
#significa que los chunks consecutivos compartirán 'overlap' caracteres. Por ejemplo, si overlap=30, los últimos 30 caracteres de un chunk serán los primeros 30 caracteres del siguiente chunk.

for tamano, overlap in [(150, 0), (150, 30), (400, 0), (400, 80)]:
    chunks_resultantes = chunking_por_caracteres(documento_largo, tamano=tamano, solapamiento=overlap)
    print(f"tamano={tamano:>4} overlap={overlap:>3} -> {len(chunks_resultantes)} chunks")
    for i, c in enumerate(chunks_resultantes):
        print(f"  [{i:03d}] ({len(c)} chars) {c!r}")
    print()

print("---------------------------")

    #¿En qué casos el chunking por párrafos corta "mejor" (en límites de idea) que el de tamaño fijo?
    # como averiguo esto? 
    # Podés comparar visualmente los chunks generados por ambos métodos y ver cuál respeta mejor los límites de las ideas o párrafos.
    # que significa "mejor" en límites de idea? Significa que los chunks generados respetan los párrafos o las ideas completas, en lugar de cortar arbitrariamente en medio de una oración o párrafo.

for max_chars, overlap in [(150, 0), (150, 30), (400, 0), (400, 80)]:
    chunks_resultantes = chunking_por_parrafos(documento_largo, max_chars=max_chars, solapamiento=overlap)
    print(f"max_chars={max_chars:>4} overlap={overlap:>3} -> {len(chunks_resultantes)} chunks")
    for i, c in enumerate(chunks_resultantes):
        print(f"  [{i:03d}] ({len(c)} chars) {c!r}")
    print()