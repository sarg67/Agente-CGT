"""
Interfaz Streamlit del agente de preguntas y respuestas sobre las
Condiciones Generales de Trabajo (CGT) de IMSS Bienestar.
"""

import os
import re
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

# Sentinela que el LLM emite cuando no procede responder; el código
# la detecta y muestra MENSAJE_FUERA_DE_CONTEXTO en su lugar.
TOKEN_FUERA_DE_CONTEXTO = "FUERA_DE_CONTEXTO"

PREGUNTA_CONFIRMACION = "Con gusto te ayudo. ¿Laboras en IMSS Bienestar?"

PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente que responde preguntas de trabajadores de IMSS
Bienestar sobre su normatividad laboral: las Condiciones Generales de
Trabajo (CGT) y las leyes aplicables (LFTSE, LISSSTE, LGRA, Ley de
Premios). Usa únicamente el siguiente contexto extraído de los
documentos para responder, y cita siempre de qué documento proviene
la información. Apóyate en el historial para entender preguntas de
seguimiento.

{instruccion_confirmacion}

Reglas, aplícalas en este orden:
1. Si la pregunta es claramente ajena al ámbito laboral (deportes,
   cocina, entretenimiento, política, ciencia, temas personales sin
   relación con el trabajo), responde EXACTAMENTE:
   {token_fuera_de_contexto}
   Esta decisión depende SOLO del TEMA de la pregunta, NUNCA de lo
   que contenga o no contenga el contexto. Si la pregunta menciona
   trabajo, sueldo, bonos, prestaciones, permisos o cualquier asunto
   laboral, NO uses esta palabra aunque el contexto no traiga la
   respuesta: aplica la regla 2.
2. En cualquier otro caso, responde con la información del contexto
   citando la fuente. Si el contexto no cubre el detalle exacto o el
   supuesto específico que preguntan (por ejemplo montos, una
   prestación que no aparece, o un caso particular como tener menos
   antigüedad de la que pide la norma), dilo con honestidad: aclara
   que la normatividad consultada no especifica ese supuesto y
   comparte la información más cercana disponible. Si el mensaje del
   usuario es solo una confirmación (como "sí, laboro ahí"), responde
   la duda que dejó pendiente en el historial.

Cuando respondas {token_fuera_de_contexto}, responde únicamente esa
palabra, sin agregar texto, comillas ni explicación. Nunca la uses
como parte de una respuesta normal.

Ejemplo de la regla 2: si preguntan "¿me tocan vales de despensa?" y
el contexto no menciona vales pero sí otras prestaciones (premios,
estímulos, aguinaldo), aclara que los vales no aparecen en la
normatividad consultada y comparte las prestaciones que sí establece.
Otro ejemplo: si preguntan por vacaciones con 3 meses de antigüedad y
la norma solo habla de más de 6 meses, explica que ese supuesto no
viene especificado y comparte la regla de los 6 meses. Otro más: "¿de
cuánto es el bono de puntualidad?" es laboral; si el contexto no trae
ese bono ni su monto, dilo con honestidad y menciona los estímulos
que sí establece la normatividad. En estos tres ejemplos está
PROHIBIDO responder {token_fuera_de_contexto}.

Historial de la conversación:
{historial}

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
).partial(token_fuera_de_contexto=TOKEN_FUERA_DE_CONTEXTO)

# Instrucción que se inserta en el prompt según el estado (guardado en
# código, no inferido por el LLM) de si la persona confirmó laborar
# en IMSS Bienestar.
INSTRUCCION_CONFIRMADO = (
    "La persona YA CONFIRMÓ que labora en IMSS Bienestar. Nunca "
    "preguntes dónde trabaja: responde sus preguntas directamente."
)
INSTRUCCION_SIN_CONFIRMAR = (
    "Aún NO se sabe si la persona labora en IMSS Bienestar. Si su "
    "pregunta es laboral y no menciona IMSS Bienestar, no la "
    f'respondas todavía: responde solo esto y nada más: '
    f'"{PREGUNTA_CONFIRMACION}"'
)


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
        # El LLM emite la sentinela cuando no procede responder; aquí
        # se convierte en el mensaje de rechazo para el usuario.
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


# Filtro previo al retrieval: preguntas sobre otras instituciones se
# rechazan por palabras clave, sin gastar llamadas a la API. "IMSS"
# solo cuenta como otra institución si NO va seguido de "Bienestar".
PATRONES_OTRAS_INSTITUCIONES = [
    r"\bimss\b(?!\s*[-–]?\s*bienestar)",
    r"\bseguro\s+social\b",
    r"\b(pemex|cfe|sedena|semar|sat|sep|infonavit)\b",
    # ISSSTE solo como empleador (la Ley del ISSSTE sí aplica aquí)
    r"(trabaj\w+|labor\w+|emplead\w+)\s+(en|de|del|para)\s+(el\s+)?issste\b",
]


def menciona_otra_institucion(texto):
    texto = texto.lower()
    return any(re.search(p, texto) for p in PATRONES_OTRAS_INSTITUCIONES)


REGEX_IMSS_BIENESTAR = re.compile(r"imss[\s-]*bienestar", re.IGNORECASE)
REGEX_AFIRMACION = re.compile(
    r"\bs[ií]\b|\bclaro\b|\bas[ií] es\b|\bcorrecto\b|\bafirmativo\b",
    re.IGNORECASE,
)
REGEX_NEGACION = re.compile(r"\bno\b", re.IGNORECASE)


def actualizar_confirmacion(pregunta):
    """Actualiza en session_state si la persona labora en IMSS Bienestar.

    El estado vive en código (no lo infiere el LLM): se confirma si la
    pregunta menciona IMSS Bienestar, o por un sí/no cuando el mensaje
    anterior del asistente fue la pregunta de confirmación.
    """
    if REGEX_IMSS_BIENESTAR.search(pregunta):
        st.session_state.confirmado = True
        return

    mensajes = st.session_state.mensajes
    respondiendo_confirmacion = (
        mensajes
        and mensajes[-1]["rol"] == "assistant"
        and mensajes[-1]["contenido"] == PREGUNTA_CONFIRMACION
    )
    if respondiendo_confirmacion:
        if REGEX_NEGACION.search(pregunta):
            st.session_state.confirmado = False
        elif REGEX_AFIRMACION.search(pregunta):
            st.session_state.confirmado = True


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
    st.session_state.confirmado = None
    st.session_state.pregunta_pendiente = None

if st.button("Nueva conversación"):
    st.session_state.mensajes = list(HISTORIAL_INICIAL)
    st.session_state.confirmado = None
    st.session_state.pregunta_pendiente = None

for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])

# st.chat_input envía la pregunta al presionar Enter
pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)

    # ¿Este mensaje responde a la pregunta "¿Laboras en IMSS Bienestar?"?
    era_respuesta_confirmacion = (
        st.session_state.mensajes
        and st.session_state.mensajes[-1]["rol"] == "assistant"
        and st.session_state.mensajes[-1]["contenido"] == PREGUNTA_CONFIRMACION
    )
    actualizar_confirmacion(pregunta)

    # La pregunta pendiente solo se responde si la persona acaba de
    # confirmar; en cualquier otro caso queda obsoleta y se descarta.
    pendiente = None
    if era_respuesta_confirmacion and st.session_state.confirmado:
        pendiente = st.session_state.pregunta_pendiente
    st.session_state.pregunta_pendiente = None

    if menciona_otra_institucion(pregunta) or st.session_state.confirmado is False:
        respuesta = MENSAJE_FUERA_DE_CONTEXTO
    else:
        with st.spinner("Buscando respuesta..."):
            cadena = cargar_cadena_rag()
            instruccion = (
                INSTRUCCION_CONFIRMADO
                if st.session_state.confirmado
                else INSTRUCCION_SIN_CONFIRMAR
            )
            respuesta = cadena.invoke(
                {
                    # Tras confirmar, se responde la pregunta original
                    # sin pedirle a la persona que la repita.
                    "question": pendiente or pregunta,
                    "historial": formatear_historial(st.session_state.mensajes),
                    "instruccion_confirmacion": instruccion,
                }
            )
        if respuesta == PREGUNTA_CONFIRMACION:
            st.session_state.pregunta_pendiente = pregunta

    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
    st.rerun()
