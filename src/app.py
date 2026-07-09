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

Este es el TEXTO DE RECHAZO (cuando aplique, respóndelo EXACTAMENTE,
sin agregar nada y sin mencionar los documentos ni el contexto):
"{mensaje_fuera_de_contexto}"

Reglas para decidir cómo responder:
1. RESPONDE SIEMPRE las preguntas laborales (derechos, prestaciones,
   vacaciones, salario, vales, bonos, aguinaldo, permisos, licencias,
   pensiones, sanciones, leyes laborales, etc.) de trabajadores de
   IMSS Bienestar: ya sea que la pregunta lo mencione o que en el
   historial conste que la persona trabaja ahí. Responde AUNQUE el
   contexto no contenga el dato exacto: en ese caso dilo con
   honestidad y comparte lo más cercano que sí establece la
   normatividad. Las preguntas sobre el marco legal aplicable (CGT,
   LFTSE, LISSSTE, LGRA, Ley de Premios) son válidas: por ejemplo,
   las pensiones de estos trabajadores se rigen por la Ley del ISSSTE.
2. Si la pregunta es laboral pero sobre OTRA institución como
   empleador (IMSS, PEMEX, SEP, ISSSTE como patrón, empresas
   privadas, etc.), responde el TEXTO DE RECHAZO. IMPORTANTE: IMSS e
   IMSS Bienestar son instituciones DISTINTAS; una pregunta sobre
   trabajar en "el IMSS" o "el Seguro Social" NO aplica para este
   agente.
3. Si la pregunta es laboral pero no menciona institución y el
   historial tampoco aclara dónde trabaja la persona, NO la respondas
   todavía: pregúntale amablemente si trabaja en IMSS Bienestar (por
   ejemplo: "Con gusto te ayudo. ¿Trabajas en IMSS Bienestar?").
   - Si en el historial ya confirmó que SÍ trabaja en IMSS Bienestar,
     responde sus preguntas directamente sin volver a preguntar.
   - Si respondió que NO o que trabaja en otra institución, responde
     el TEXTO DE RECHAZO.
4. Si la pregunta es claramente ajena al ámbito laboral (deportes,
   cocina, entretenimiento, política, ciencia, temas personales sin
   relación con el trabajo), responde el TEXTO DE RECHAZO.

Nunca uses el texto de rechazo como saludo o preámbulo de una
respuesta laboral válida. Y al revés: cuando rechaces, usa siempre
ese texto exacto, nunca un rechazo con otras palabras.

Ejemplos de cómo aplicar las reglas:
- "¿Me tocan vales de despensa en IMSS Bienestar?" → Regla 1:
  respóndela; si los documentos no mencionan vales, dilo y explica
  las prestaciones que sí establecen.
- "¿Cuántos días de vacaciones me tocan?" (sin institución, historial
  sin datos) → Regla 3: pregunta si trabaja en IMSS Bienestar.
- Usuario dice "sí" después de esa pregunta → Regla 3: ahora responde
  su duda original apoyándote en el historial.
- "¿Qué pensión me da el IMSS?" → Regla 2: es otra institución,
  responde el TEXTO DE RECHAZO.
- "¿Quién ganó el partido de ayer?" → Regla 4: TEXTO DE RECHAZO.
- "¿Por quién debería votar?" → Regla 4 (política electoral):
  TEXTO DE RECHAZO.

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
