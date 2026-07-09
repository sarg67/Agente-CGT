"""
Interfaz Streamlit del agente de preguntas y respuestas sobre las
Condiciones Generales de Trabajo (CGT) de IMSS Bienestar.
"""

import os
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "cgt_imss_bienestar"

MENSAJE_BIENVENIDA = (
    "Hola, soy el Asistente Laboral de IMSS Bienestar. "
    "¿En qué te puedo ayudar?"
)

MENSAJE_FUERA_DE_CONTEXTO = (
    "Solo puedo responder preguntas relacionadas con las condiciones "
    "laborales y el marco normativo de los trabajadores de IMSS Bienestar. "
    "¿Tienes alguna duda sobre tus derechos o prestaciones?"
)

PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente que responde preguntas sobre normatividad laboral
de IMSS Bienestar: las Condiciones Generales de Trabajo (CGT) y las
leyes relacionadas (LFTSE, LISSSTE, LGRA, Ley de Premios). Usa
únicamente el siguiente contexto extraído de los documentos para
responder, y cita siempre de qué documento proviene la información.
Puedes apoyarte en el historial de la conversación para entender
preguntas de seguimiento.

Reglas para decidir si respondes:
1. Si la pregunta trata de temas laborales en sentido amplio (trabajo,
   derechos, prestaciones, vacaciones, salario, vales, bonos,
   aguinaldo, estímulos, permisos, licencias, pensiones, sanciones,
   leyes laborales, etc.), SIEMPRE respóndela, AUNQUE el contexto no
   contenga la respuesta exacta. Si el contexto no cubre el detalle
   (por ejemplo montos, fechas o una prestación específica), dilo con
   honestidad y comparte lo más cercano que sí establece la
   normatividad. En caso de duda, asume que la pregunta es laboral.
2. SOLO si la pregunta es claramente ajena al ámbito laboral (por
   ejemplo: deportes, cocina, entretenimiento, política, ciencia,
   temas personales sin relación con el trabajo), responde EXACTAMENTE
   con este texto y nada más (sin mencionar los documentos, el
   contexto ni lo que encontraste):
"{mensaje_fuera_de_contexto}"

Nunca uses el texto de la regla 2 como saludo o preámbulo de una
respuesta laboral: resérvalo únicamente para preguntas ajenas. Y al
revés: cuando rechaces una pregunta ajena, usa siempre ese texto
exacto, nunca un rechazo con otras palabras.

Ejemplos de cómo aplicar las reglas:
- "¿Me corresponden vales de despensa?" → Es laboral (prestaciones):
  respóndela. Si los documentos no mencionan vales de despensa,
  dilo y explica las prestaciones económicas que sí establecen.
- "¿Cuánto gana una enfermera?" → Es laboral (salario): respóndela
  con lo que digan los documentos sobre sueldos, aunque no haya cifras.
- "¿Quién ganó el partido de ayer?" → Es ajena: responde el texto
  exacto de la regla 2.
- "¿Por quién debería votar?" → Es ajena (política electoral, aunque
  exista el voto sindical): responde el texto exacto de la regla 2.

Historial de la conversación:
{historial}

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
).partial(mensaje_fuera_de_contexto=MENSAJE_FUERA_DE_CONTEXTO)


@st.cache_resource
def cargar_cadena_rag():
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    vectorstore = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=RUTA_CHROMA,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

    def formatear_contexto(docs):
        return "\n\n".join(
            f"[Fuente: {doc.metadata.get('fuente', 'desconocida')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )

    def normalizar_fuera_de_contexto(texto):
        # Garantiza el mensaje exacto cuando el modelo rechaza, sin
        # descartar respuestas largas que mencionen la frase de paso.
        limpio = texto.strip().strip('"')
        es_rechazo = limpio.startswith(
            "Solo puedo responder preguntas relacionadas"
        ) and len(limpio) <= len(MENSAJE_FUERA_DE_CONTEXTO) + 60
        return MENSAJE_FUERA_DE_CONTEXTO if es_rechazo else texto

    return (
        {
            "context": itemgetter("question") | retriever | formatear_contexto,
            "question": itemgetter("question"),
            "historial": itemgetter("historial"),
        }
        | PROMPT
        | llm
        | StrOutputParser()
        | normalizar_fuera_de_contexto
    )


# Máximo de mensajes previos que se incluyen en el prompt, para no
# exceder los límites de tokens de la API.
MAX_MENSAJES_HISTORIAL = 8


def formatear_historial(mensajes):
    if not mensajes:
        return "(la conversación apenas comienza)"
    lineas = []
    for m in mensajes[-MAX_MENSAJES_HISTORIAL:]:
        rol = "Usuario" if m["rol"] == "user" else "Asistente"
        lineas.append(f"{rol}: {m['contenido']}")
    return "\n".join(lineas)


st.set_page_config(page_title="Agente CGT - IMSS Bienestar")
st.title("Agente CGT - IMSS Bienestar")
st.write(
    "Pregunta sobre las Condiciones Generales de Trabajo de IMSS Bienestar."
)

if not os.getenv("COHERE_API_KEY"):
    st.error("Falta COHERE_API_KEY. Define la variable en tu archivo .env.")
    st.stop()

if not os.path.isdir(RUTA_CHROMA):
    st.error(
        f"No se encontró la base vectorial en '{RUTA_CHROMA}'. "
        "Corre primero: python src/ingestor.py"
    )
    st.stop()

# La bienvenida vive como primer mensaje del historial: así siempre
# se muestra al iniciar y "Nueva conversación" la restaura.
HISTORIAL_INICIAL = [{"rol": "assistant", "contenido": MENSAJE_BIENVENIDA}]

if "mensajes" not in st.session_state:
    st.session_state.mensajes = list(HISTORIAL_INICIAL)

if st.button("Nueva conversación"):
    st.session_state.mensajes = list(HISTORIAL_INICIAL)

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])

# st.chat_input envía la pregunta al presionar Enter
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)
    with st.spinner("Buscando respuesta..."):
        cadena = cargar_cadena_rag()
        respuesta = cadena.invoke(
            {
                "question": pregunta,
                "historial": formatear_historial(st.session_state.mensajes),
            }
        )
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
    st.rerun()
