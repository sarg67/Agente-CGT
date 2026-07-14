"""
Analizador de vacíos (gaps) en la base de conocimiento.

Lee logs/metricas.jsonl, filtra las preguntas que activaron el safeguard
(las que el agente no pudo responder), las agrupa por tema según palabras
clave compartidas y sugiere qué tipo de documento cubriría cada vacío.
Genera logs/gaps_detectados.txt.

Uso: python src/analizador_gaps.py
"""

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

ARCHIVO_METRICAS = "logs/metricas.jsonl"
RUTA_GAPS = "logs/gaps_detectados.txt"

# Cada tema se define por sus palabras clave y el documento que cubriría
# el vacío si empezaran a llegar muchas preguntas laborales de ese tipo.
TEMAS = {
    "pensiones": {
        "claves": ["pension", "pensione", "jubilaci", "retiro", "cesantia", "vejez"],
        "documento": (
            "Guía del régimen de pensiones y seguridad social aplicable a "
            "trabajadores de IMSS Bienestar."
        ),
    },
    "salario y prestaciones": {
        "claves": ["salario", "sueldo", "pago", "pagan", "aguinaldo", "bono",
                   "vales", "gana", "prestacion", "prestacione"],
        "documento": (
            "Tabulador de sueldos y catálogo de prestaciones económicas."
        ),
    },
    "permisos y licencias": {
        "claves": ["permiso", "licencia", "incapacidad", "ausencia", "falta"],
        "documento": "Reglamento de permisos, licencias e incapacidades.",
    },
    "vacaciones y descansos": {
        "claves": ["vacacion", "vacacione", "descanso", "feriado", "asueto"],
        "documento": "Política y calendario oficial de vacaciones y descansos.",
    },
    "sanciones y disciplina": {
        "claves": ["sancion", "sancione", "despido", "destitu", "inhabilita",
                   "suspension"],
        "documento": "Catálogo de sanciones y procedimiento disciplinario.",
    },
}

TEMA_OTROS = "otros / sin tema laboral claro"
SUGERENCIA_OTROS = (
    "Revisar manualmente. Suelen ser preguntas sobre otras instituciones "
    "(IMSS, PEMEX, etc.) o temas ajenos al ámbito laboral: en esos casos "
    "el bloqueo es correcto y NO representa un vacío en la base."
)


def cargar_safeguard(ruta=ARCHIVO_METRICAS):
    if not os.path.exists(ruta):
        return []
    preguntas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            reg = json.loads(linea)
            if reg.get("tipo_respuesta") == "safeguard":
                preguntas.append(reg["pregunta"])
    return preguntas


def sin_acentos(texto):
    """Minúsculas y sin acentos, para que 'pensión' case con 'pension'."""
    nfkd = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clasificar_tema(pregunta):
    """Devuelve el tema cuyas palabras clave aparecen en la pregunta."""
    texto = sin_acentos(pregunta)
    for tema, info in TEMAS.items():
        if any(re.search(rf"\b{clave}", texto) for clave in info["claves"]):
            return tema
    return TEMA_OTROS


def generar(preguntas):
    lineas = []
    lineas.append("ANÁLISIS DE VACÍOS (GAPS) EN LA BASE DE CONOCIMIENTO")
    lineas.append("=" * 55)
    lineas.append(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lineas.append(f"Preguntas bloqueadas analizadas: {len(preguntas)}")
    lineas.append("")

    if not preguntas:
        lineas.append("No hay preguntas bloqueadas registradas todavía.")
        return "\n".join(lineas) + "\n"

    # 1. Frecuencia global
    lineas.append("1. PREGUNTAS SIN RESPUESTA MÁS FRECUENTES")
    for pregunta, veces in Counter(preguntas).most_common():
        lineas.append(f"   [{veces}x] {pregunta}")
    lineas.append("")

    # 2. Agrupadas por tema
    lineas.append("2. AGRUPADAS POR TEMA")
    por_tema = defaultdict(list)
    for pregunta in preguntas:
        por_tema[clasificar_tema(pregunta)].append(pregunta)

    # Temas laborales primero, "otros" al final.
    orden = list(TEMAS.keys()) + [TEMA_OTROS]
    for tema in orden:
        if tema not in por_tema:
            continue
        grupo = por_tema[tema]
        lineas.append(f"   == {tema} == ({len(grupo)} pregunta(s))")
        for pregunta, veces in Counter(grupo).most_common():
            lineas.append(f"      [{veces}x] {pregunta}")
        sugerencia = TEMAS[tema]["documento"] if tema in TEMAS else SUGERENCIA_OTROS
        lineas.append(f"      Sugerencia de documento: {sugerencia}")
        lineas.append("")

    lineas.append("NOTA: no todo bloqueo es un vacío real. Las preguntas sobre")
    lineas.append("otras instituciones o temas ajenos se bloquean a propósito.")
    lineas.append("Un tema laboral con muchas preguntas sí sugiere agregar un")
    lineas.append("documento a la base.")
    return "\n".join(lineas) + "\n"


def main():
    preguntas = cargar_safeguard()
    reporte = generar(preguntas)
    os.makedirs(os.path.dirname(RUTA_GAPS), exist_ok=True)
    with open(RUTA_GAPS, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(reporte)
    print(f"Reporte guardado en {RUTA_GAPS}")


if __name__ == "__main__":
    main()
