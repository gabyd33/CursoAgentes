import hashlib
from dataclasses import dataclass, field
from time import time

@dataclass
class Chunk:
    texto: str
    documento_id: str
    posicion: int           # orden del chunk dentro del documento
    chunk_id: str = field(init=False)
    fuente: str = ""
    fecha_ingesta: float = field(default_factory=time)

    def __post_init__(self):
        # id determinístico: mismo documento + misma posición -> mismo id, siempre
        hash_contenido = hashlib.sha256(self.texto.encode()).hexdigest()[:8]
        self.chunk_id = f"{self.documento_id}-{self.posicion:03d}-{hash_contenido}"

    