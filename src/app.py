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

PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente que responde preguntas sobre las Condiciones
Generales de Trabajo (CGT) de IMSS Bienestar. Usa únicamente el
siguiente contexto extraído del documento para responder. Si la
respuesta no está en el contexto, di que no la encontraste en el
documento.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
)


@st.cache_resource
def cargar_cadena_rag():
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    vectorstore = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=RUTA_CHROMA,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatCohere(model="command-r-plus-08-2024", temperature=0)

    def formatear_contexto(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | formatear_contexto,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
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
