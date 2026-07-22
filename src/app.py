"""
Interfaz Streamlit del agente de preguntas y respuestas sobre las
Condiciones Generales de Trabajo (CGT) de IMSS Bienestar.
"""

import json
import os
import re
import time
from datetime import datetime
from operator import itemgetter

import streamlit as st
from dotenv import load_dotenv

import monitor
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "cgt_imss_bienestar"

MENSAJE_BIENVENIDA = (
    "Hola, soy el Asistente Laboral de IMSS Bienestar, un agente de "
    "inteligencia artificial. Respondo con base en las Condiciones "
    "Generales de Trabajo y su marco normativo. ¿En qué te puedo ayudar?"
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
   citando la fuente. Cuando una prestación tenga condiciones,
   requisitos o proporcionalidad —por ejemplo que su monto dependa de
   los días o años trabajados, una antigüedad mínima, o plazos de
   pago—, inclúyelas siempre que aparezcan en el contexto, aunque estén
   en otro artículo; no te limites a la cifra principal.
   Encuadre importante: si la norma establece la regla general (por
   ejemplo, que una prestación es proporcional al tiempo trabajado)
   aunque no dé la cifra o fórmula exacta, presenta esa regla COMO la
   respuesta —porque sí responde la pregunta— y menciona solo como
   aclaración secundaria qué parte no viene especificada (por ejemplo,
   que la determine el Ejecutivo Federal). No lo plantees como si la
   norma no dijera nada. Solo cuando el contexto no tenga NINGUNA
   información aplicable, aclara con honestidad que la normatividad
   consultada no especifica ese supuesto y comparte lo más cercano
   disponible. Si el mensaje del usuario es solo una confirmación (como
   "sí, laboro ahí"), responde la duda que dejó pendiente en el
   historial.

Conversión de unidades de tiempo: si el usuario expresa su antigüedad o
tiempo trabajado en días y la norma usa meses o años, conviértelo tú
mismo ANTES de concluir (aproxima 1 mes = 30 días, 1 año = 365 días),
muestra el cálculo, compáralo con el umbral de la norma y concluye de
forma DEFINITIVA (sí cumple / no cumple). Reglas: nunca compares días
contra meses sin convertir; si el número convertido ya supera el umbral
NO digas "está cerca de cumplir" (supéralo es cumplir); y NO digas que
"la normatividad no especifica" cuando sí puedes convertir y comparar
(en ese caso la norma SÍ da la respuesta). Ejemplos: 190 días ÷ 30 ≈
6.3 meses, que es MÁS de 6 → SÍ cumple "más de seis meses continuos".
150 días ÷ 30 = 5 meses, menos de 6 → NO cumple.

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
que sí establece la normatividad. Otro más: "¿puedo trabajar en IMSS
Bienestar y en otra institución al mismo tiempo?" es laboral
(compatibilidad de empleos): respóndela con lo que la normatividad
diga sobre desempeñar dos empleos o incompatibilidades; que mencione
otra institución junto a IMSS Bienestar no la hace ajena. En estos
cuatro ejemplos está PROHIBIDO responder {token_fuera_de_contexto}.

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
    f'"{PREGUNTA_CONFIRMACION}" '
    "Esto también aplica cuando la persona menciona su profesión u "
    'oficio sin decir dónde trabaja (ej. "soy enfermera, ¿qué '
    'prestaciones tengo?"): es laboral, así que pregunta primero.'
)

# Query expansion: reformula la pregunta a una consulta de búsqueda
# concisa antes de recuperar. El relleno ("cuántos", "me corresponden")
# y sobre todo el nombre de la institución diluyen la señal semántica y
# hacen que preguntas verbosas no encuentren el artículo correcto.
#
# Recibe también el historial porque las preguntas de seguimiento no
# traen tema propio: "¿qué dicen las CGT específicamente?" se
# reformulaba como "CGT normas laborales" y recuperaba artículos al
# azar. Con el historial hereda el tema de la pregunta anterior.
REFORMULAR_PROMPT = ChatPromptTemplate.from_template(
    """Reformula la pregunta del usuario como una consulta de búsqueda
breve (3 a 8 palabras) que conserve solo los términos clave del tema
laboral. Quita el relleno (por ejemplo "cuántos", "me corresponden",
"en un año") y NO incluyas nombres de instituciones (IMSS Bienestar,
etc.), porque restan precisión a la búsqueda.

IMPORTANTE — preguntas de seguimiento: si la pregunta no tiene tema
propio y depende de lo hablado antes (por ejemplo "¿qué dicen las CGT
específicamente?", "¿en qué artículo?", "¿y en ese caso?", "¿cuánto
es?"), toma el tema de la última pregunta del usuario en el historial
y construye la consulta con ESE tema. Ejemplo: si antes se preguntó
por maternidad y ahora preguntan "¿qué dicen las CGT?", la consulta
correcta es "maternidad descanso licencia", no "CGT normas". Nunca
devuelvas una consulta sin tema laboral concreto.

Historial de la conversación:
{historial}

Pregunta: {question}
Consulta:"""
)


# Instituciones-empleador que, si quedan en la consulta de búsqueda,
# hacen match con el encabezado repetido de cada página del CGT
# ("... IMSS-BIENESTAR ...") y sepultan el artículo que sí responde la
# pregunta. El prompt de reformulación ya pide no incluirlas, pero el LLM
# es variable y a veces las deja; esto las quita de forma determinista.
REGEX_RUIDO_CONSULTA = re.compile(
    r"imss[\s-]*bienestar|\bimss\b|\bbienestar\b", re.IGNORECASE
)


def limpiar_consulta(consulta):
    """Elimina de la consulta reformulada los nombres de la institución
    empleadora (IMSS, Bienestar, IMSS-BIENESTAR) y normaliza espacios y
    puntuación de los bordes. Si quedara vacía, devuelve la original."""
    limpia = REGEX_RUIDO_CONSULTA.sub(" ", consulta)
    limpia = re.sub(r"\s+", " ", limpia).strip(" .,-–")
    return limpia or consulta


@st.cache_resource
def cargar_cadena_rag():
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    vectorstore = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=RUTA_CHROMA,
    )
    # k=10: da margen para preguntas que combinan varios artículos o
    # documentos (p. ej. aguinaldo en CGT Art. 41 y LFTSE Art. 42 Bis).
    # Se usa búsqueda por similitud a propósito: MMR con fetch_k probó
    # perder fuentes relevantes (dejaba fuera la LFTSE), y fetch_k no
    # es un parámetro válido para la búsqueda por similitud de Chroma.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    llm = ChatCohere(model="command-a-03-2025", temperature=0)

    # Reformula la pregunta a una consulta de búsqueda antes de recuperar.
    reformulador = REFORMULAR_PROMPT | llm | StrOutputParser()

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

    def recuperar_docs(entrada):
        consulta = limpiar_consulta(reformulador.invoke(entrada))
        return retriever.invoke(consulta)

    generar_respuesta = (
        {
            "context": lambda x: formatear_contexto(x["docs"]),
            "question": itemgetter("question"),
            "historial": itemgetter("historial"),
            "instruccion_confirmacion": itemgetter("instruccion_confirmacion"),
        }
        | PROMPT
        | llm
        | StrOutputParser()
        | normalizar_fuera_de_contexto
    )

    # Se expone "docs" junto con la respuesta (en vez de devolver solo el
    # texto) para poder registrar qué documentos se usaron en cada consulta.
    return RunnablePassthrough.assign(docs=recuperar_docs) | RunnablePassthrough.assign(
        respuesta=generar_respuesta
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


REGEX_IMSS_BIENESTAR = re.compile(r"imss[\s-]*bienestar", re.IGNORECASE)


def menciona_otra_institucion(texto):
    # Si menciona IMSS Bienestar explícitamente, la pregunta es para
    # este agente aunque también nombre otras instituciones (ej.
    # "¿puedo trabajar en IMSS Bienestar e IMSS al mismo tiempo?").
    if REGEX_IMSS_BIENESTAR.search(texto):
        return False
    texto = texto.lower()
    return any(re.search(p, texto) for p in PATRONES_OTRAS_INSTITUCIONES)
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

# Registro unificado de métricas de uso. Cada pregunta genera una línea
# JSON; el voto 👍/👎 actualiza esa misma línea (por su id).
ARCHIVO_METRICAS = "logs/metricas.jsonl"


def registrar_metrica(pregunta, tipo_respuesta, tiempo_respuesta, respuesta,
                       documentos_usados, pregunta_pendiente=None):
    """Escribe una línea en metricas.jsonl y devuelve su id, para que el
    voto posterior pueda actualizar ese mismo registro."""
    st.session_state.metrica_contador = (
        st.session_state.get("metrica_contador", 0) + 1
    )
    metrica_id = (
        f"{datetime.now():%Y%m%d%H%M%S%f}-{st.session_state.metrica_contador}"
    )
    registro = {
        "id": metrica_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pregunta": pregunta,
        "tipo_respuesta": tipo_respuesta,  # respondida / safeguard / verificacion
        "tiempo_respuesta": round(tiempo_respuesta, 3),
        "calificacion": None,  # se llena con el voto 👍/👎
        "respuesta": respuesta,
        "documentos_usados": documentos_usados,
        # Solo se llena cuando tipo_respuesta es "verificacion": la
        # pregunta que quedó pendiente hasta confirmar la institución.
        "pregunta_pendiente": pregunta_pendiente,
    }
    os.makedirs(os.path.dirname(ARCHIVO_METRICAS), exist_ok=True)
    with open(ARCHIVO_METRICAS, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    # Copia remota en el bucket de OCI (falla en silencio si no hay acceso).
    monitor.subir_metricas()
    return metrica_id


def actualizar_calificacion(metrica_id, calificacion):
    """Reescribe metricas.jsonl para fijar la calificación del registro."""
    if not os.path.exists(ARCHIVO_METRICAS):
        return
    registros = []
    with open(ARCHIVO_METRICAS, encoding="utf-8") as f:
        for linea in f:
            if not linea.strip():
                continue
            reg = json.loads(linea)
            if reg.get("id") == metrica_id:
                reg["calificacion"] = calificacion
            registros.append(reg)
    with open(ARCHIVO_METRICAS, "w", encoding="utf-8") as f:
        for reg in registros:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
    # Sincronizar la copia remota para que incluya la calificación.
    monitor.subir_metricas()


def registrar_feedback(indice):
    """Traduce el voto 👍/👎 y lo guarda en el registro de métricas."""
    voto = st.session_state.get(f"feedback_{indice}")
    if voto is None:
        return
    metrica_id = st.session_state.mensajes[indice].get("metrica_id")
    if metrica_id:
        actualizar_calificacion(metrica_id, "positivo" if voto == 1 else "negativo")


for i, mensaje in enumerate(st.session_state.mensajes):
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])
        # Botones 👍/👎 en cada respuesta del agente que tenga métrica
        # asociada (excluye la bienvenida inicial).
        if mensaje["rol"] == "assistant" and mensaje.get("metrica_id"):
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                on_change=registrar_feedback,
                args=(i,),
            )

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

    inicio = time.perf_counter()
    docs = []
    if menciona_otra_institucion(pregunta):
        respuesta = MENSAJE_FUERA_DE_CONTEXTO
    elif st.session_state.confirmado is False:
        respuesta = MENSAJE_FUERA_DE_CONTEXTO
    else:
        with st.spinner("Buscando respuesta..."):
            cadena = cargar_cadena_rag()
            instruccion = (
                INSTRUCCION_CONFIRMADO
                if st.session_state.confirmado
                else INSTRUCCION_SIN_CONFIRMAR
            )
            resultado = cadena.invoke(
                {
                    # Tras confirmar, se responde la pregunta original
                    # sin pedirle a la persona que la repita.
                    "question": pendiente or pregunta,
                    "historial": formatear_historial(st.session_state.mensajes),
                    "instruccion_confirmacion": instruccion,
                }
            )
            respuesta = resultado["respuesta"]
            docs = resultado["docs"]
        if respuesta == PREGUNTA_CONFIRMACION:
            st.session_state.pregunta_pendiente = pregunta
    tiempo_respuesta = time.perf_counter() - inicio

    # Clasificar el tipo de respuesta para el monitoreo de calidad.
    if respuesta == MENSAJE_FUERA_DE_CONTEXTO:
        tipo_respuesta = "safeguard"
    elif respuesta == PREGUNTA_CONFIRMACION:
        tipo_respuesta = "verificacion"
    else:
        tipo_respuesta = "respondida"

    # Fuentes de los documentos recuperados para esta consulta (sin
    # duplicados, en el orden en que el retriever los devolvió).
    documentos_usados = []
    for doc in docs:
        fuente = doc.metadata.get("fuente", "desconocida")
        if fuente not in documentos_usados:
            documentos_usados.append(fuente)

    pregunta_pendiente_registro = (
        pregunta if tipo_respuesta == "verificacion" else None
    )
    metrica_id = registrar_metrica(
        pregunta, tipo_respuesta, tiempo_respuesta, respuesta,
        documentos_usados, pregunta_pendiente_registro,
    )

    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    st.session_state.mensajes.append(
        {"rol": "assistant", "contenido": respuesta, "metrica_id": metrica_id}
    )
    st.rerun()
