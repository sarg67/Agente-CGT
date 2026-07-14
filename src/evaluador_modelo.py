"""
Evaluador de calidad del modelo.

Corre la cadena RAG directamente (sin Streamlit) contra un conjunto fijo
de 10 preguntas con criterios de evaluación, calcula PASS/FAIL por
pregunta y una puntuación total, y guarda el detalle en
logs/evaluacion_modelo.txt.

El modelo y sus parámetros están en variables al inicio para poder
cambiarlos y comparar versiones fácilmente.

Uso: python src/evaluador_modelo.py
"""

import os
import unicodedata
from datetime import datetime
from operator import itemgetter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser

from app import (
    INSTRUCCION_CONFIRMADO,
    MENSAJE_FUERA_DE_CONTEXTO,
    NOMBRE_COLECCION,
    PROMPT,
    RUTA_CHROMA,
    TOKEN_FUERA_DE_CONTEXTO,
)

load_dotenv()

# --- Configuración del modelo (cambiar aquí para comparar versiones) ---
MODELO_LLM = "command-r-plus-08-2024"
MODELO_EMBEDDINGS = "embed-multilingual-v3.0"
TEMPERATURA = 0
TOP_K = 6

RUTA_EVALUACION = "logs/evaluacion_modelo.txt"


def normalizar(texto):
    """Minúsculas y sin acentos, para evaluar criterios de forma robusta."""
    nfkd = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Atajos de citación reutilizables por los criterios.
def cita_cgt(t):
    return "cgt" in t or "condiciones generales" in t


def cita_lftse(t):
    return "lftse" in t or "federal de los trabajadores" in t


def cita_lgra(t):
    return "lgra" in t or "responsabilidades administrativas" in t


def cita_premios(t):
    return "ley de premios" in t or "premios, estimulos" in t


DERECHOS = ["vacacion", "aguinaldo", "salario", "sueldo", "licencia",
            "permiso", "estabilidad", "seguridad social", "escalafon",
            "sindicaliz", "descanso", "prestacion"]

# Cada caso: (id, pregunta, descripción del criterio, función evaluadora).
# La evaluadora recibe (texto_normalizado, es_safeguard) y devuelve bool.
CASOS = [
    ("P1", "¿Cuántos días de vacaciones corresponden con 6 meses de servicio?",
     "Menciona 10 días y cita CGT o LFTSE",
     lambda t, sg: ("10" in t or "diez" in t) and "dia" in t
     and (cita_cgt(t) or cita_lftse(t))),

    ("P2", "¿Cuánto es el aguinaldo?",
     "Menciona número de días y cita CGT",
     lambda t, sg: any(c.isdigit() for c in t) and "dia" in t and cita_cgt(t)),

    ("P3", "¿Qué pasa si me despiden injustificadamente?",
     "Menciona reinstalación o indemnización y cita LFTSE",
     lambda t, sg: ("reinstal" in t or "indemniz" in t) and cita_lftse(t)),

    ("P4", "¿Qué sanciones tiene un servidor público corrupto?",
     "Menciona suspensión, destitución o inhabilitación y cita LGRA",
     lambda t, sg: ("suspension" in t or "destitu" in t or "inhabilita" in t)
     and cita_lgra(t)),

    # La maternidad se expresa en la norma como "descanso" en MESES
    # (un mes antes del parto, dos después), no como "días de licencia":
    # el criterio acepta esas formas reales para no marcar falsos FAIL.
    ("P5", "¿Tengo derecho a licencia de maternidad?",
     "Menciona el periodo de licencia/descanso de maternidad y cita CGT o LFTSE",
     lambda t, sg: ("licencia" in t or "descanso" in t or "mes" in t)
     and (cita_cgt(t) or cita_lftse(t))),

    ("P6", "¿Qué es la Ley de Premios?",
     "Explica su propósito y cita Ley de Premios",
     lambda t, sg: cita_premios(t)
     and ("premio" in t or "estimulo" in t or "recompensa" in t)),

    ("P7", "¿Ante qué tribunal se resuelven los conflictos laborales?",
     "Menciona el TFCA y cita LFTSE",
     lambda t, sg: ("tfca" in t or "conciliacion y arbitraje" in t
                    or "tribunal federal de conciliacion" in t) and cita_lftse(t)),

    ("P8", "¿Cuáles son los días de descanso obligatorios?",
     "Menciona el calendario oficial y cita CGT o LFTSE",
     lambda t, sg: ("calendario" in t or "descanso obligatorio" in t)
     and (cita_cgt(t) or cita_lftse(t))),

    ("P9", "¿Qué derechos tiene un trabajador de base?",
     "Menciona al menos 2 derechos y cita CGT o LFTSE",
     lambda t, sg: (cita_cgt(t) or cita_lftse(t))
     and sum(1 for d in DERECHOS if d in t) >= 2),

    ("P10", "¿Quién ganó el mundial?",
     "Debe activar el safeguard, NO responder la pregunta",
     lambda t, sg: sg),
]


def construir_cadena():
    embeddings = CohereEmbeddings(model=MODELO_EMBEDDINGS)
    vectorstore = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=RUTA_CHROMA,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatCohere(model=MODELO_LLM, temperature=TEMPERATURA)

    def formatear_contexto(docs):
        return "\n\n".join(
            f"[Fuente: {d.metadata.get('fuente', 'desconocida')}]\n{d.page_content}"
            for d in docs
        )

    def normalizar_fuera_de_contexto(texto):
        if TOKEN_FUERA_DE_CONTEXTO in texto:
            return MENSAJE_FUERA_DE_CONTEXTO
        return texto

    return (
        {
            "context": itemgetter("question") | retriever | formatear_contexto,
            "question": itemgetter("question"),
            "historial": itemgetter("historial"),
            "instruccion_confirmacion": itemgetter("instruccion_confirmacion"),
        }
        | PROMPT
        | llm
        | StrOutputParser()
        | normalizar_fuera_de_contexto
    )


def evaluar():
    cadena = construir_cadena()
    resultados = []
    for id_caso, pregunta, criterio, evaluador in CASOS:
        respuesta = cadena.invoke(
            {
                "question": pregunta,
                "historial": "(evaluación automática)",
                # Se simula usuario ya confirmado para evaluar la respuesta
                # directa (no la pregunta de verificación).
                "instruccion_confirmacion": INSTRUCCION_CONFIRMADO,
            }
        )
        es_safeguard = respuesta == MENSAJE_FUERA_DE_CONTEXTO
        aprobo = evaluador(normalizar(respuesta), es_safeguard)
        resultados.append((id_caso, pregunta, criterio, respuesta, aprobo))
        print(f"{id_caso}: {'PASS' if aprobo else 'FAIL'}")
    return resultados


def escribir_reporte(resultados):
    aprobados = sum(1 for *_, ok in resultados if ok)
    total = len(resultados)
    lineas = []
    lineas.append("EVALUACIÓN DE CALIDAD DEL MODELO")
    lineas.append("=" * 55)
    lineas.append(f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lineas.append(f"Modelo LLM: {MODELO_LLM} (temperatura={TEMPERATURA}, k={TOP_K})")
    lineas.append(f"Modelo de embeddings: {MODELO_EMBEDDINGS}")
    lineas.append("")
    lineas.append(f"PUNTUACIÓN TOTAL: {aprobados}/{total}")
    lineas.append("")
    lineas.append("RESULTADO POR PREGUNTA")
    lineas.append("-" * 55)
    for id_caso, pregunta, criterio, respuesta, ok in resultados:
        lineas.append(f"{id_caso} [{'PASS' if ok else 'FAIL'}] {pregunta}")
        lineas.append(f"   Criterio: {criterio}")
        lineas.append(f"   Respuesta: {respuesta}")
        lineas.append("")

    fallidas = [(i, p, c, r) for i, p, c, r, ok in resultados if not ok]
    lineas.append("PREGUNTAS FALLIDAS (detalle)")
    lineas.append("-" * 55)
    if fallidas:
        for id_caso, pregunta, criterio, respuesta in fallidas:
            lineas.append(f"{id_caso} {pregunta}")
            lineas.append(f"   Criterio no cumplido: {criterio}")
            lineas.append(f"   Respuesta obtenida: {respuesta}")
            lineas.append("")
    else:
        lineas.append("Ninguna. El modelo aprobó todos los casos.")

    reporte = "\n".join(lineas) + "\n"
    os.makedirs(os.path.dirname(RUTA_EVALUACION), exist_ok=True)
    with open(RUTA_EVALUACION, "w", encoding="utf-8") as f:
        f.write(reporte)
    return aprobados, total


def main():
    if not os.getenv("COHERE_API_KEY"):
        raise RuntimeError("Falta COHERE_API_KEY en el archivo .env.")
    resultados = evaluar()
    aprobados, total = escribir_reporte(resultados)
    print(f"\nPuntuación total: {aprobados}/{total}")
    print(f"Reporte guardado en {RUTA_EVALUACION}")


if __name__ == "__main__":
    main()
