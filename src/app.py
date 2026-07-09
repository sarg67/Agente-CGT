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

Si el contexto y el historial no contienen información relevante para
responder la pregunta, o la pregunta no trata sobre condiciones
laborales o el marco normativo de los trabajadores, responde
EXACTAMENTE con este texto y nada más (sin mencionar los documentos,
el contexto ni lo que encontraste):
"{mensaje_fuera_de_contexto}"

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
        # Garantiza el mensaje exacto aunque el modelo agregue comillas
        # o alguna palabra alrededor.
        if "Solo puedo responder preguntas relacionadas" in texto:
            return MENSAJE_FUERA_DE_CONTEXTO
        return texto

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

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if st.button("Nueva conversación"):
    st.session_state.mensajes = []

if not st.session_state.mensajes:
    with st.chat_message("assistant"):
        st.write(MENSAJE_BIENVENIDA)

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
