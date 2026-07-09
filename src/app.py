"""
Interfaz Streamlit del agente de preguntas y respuestas sobre las
Condiciones Generales de Trabajo (CGT) de IMSS Bienestar.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "cgt_imss_bienestar"

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

Si el contexto no contiene información relevante para responder la
pregunta, o la pregunta no trata sobre condiciones laborales o el
marco normativo de los trabajadores, responde EXACTAMENTE con este
texto y nada más (sin mencionar los documentos, el contexto ni lo
que encontraste):
"{mensaje_fuera_de_contexto}"

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
            "context": retriever | formatear_contexto,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
        | normalizar_fuera_de_contexto
    )


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

pregunta = st.text_input("Escribe tu pregunta:")

if st.button("Preguntar") and pregunta:
    with st.spinner("Buscando respuesta..."):
        cadena = cargar_cadena_rag()
        respuesta = cadena.invoke(pregunta)
    st.markdown("### Respuesta")
    st.write(respuesta)
