"""
Dashboard de monitoreo del Agente-CGT.

Ejecutar:
    streamlit run src/dashboard.py --server.port 8502

Variables de entorno (.env):
    DASHBOARD_PASSWORD=<contraseña>
    COHERE_API_KEY=<key>            # para la sección de sugerencias

Los logs se leen primero desde OCI Object Storage (src/monitor.py) y,
si eso falla, del archivo local logs/metricas.jsonl como respaldo.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
RUTA_LOG = BASE / "logs" / "metricas.jsonl"
RUTA_MANIFIESTO = BASE / "manifiesto_ingesta.json"
RUTA_LOG_INGESTA = BASE / "logs" / "ingesta_errores.log"

# Umbral de anomalía (ajustable)
UMBRAL_LATENCIA_S = 10.0

st.set_page_config(page_title="Dashboard Agente-CGT", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Carga de datos (Object Storage → local como respaldo)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def cargar_datos() -> tuple[pd.DataFrame, str]:
    origen = "local"
    texto = None
    try:
        import monitor
        texto = monitor.descargar_metricas()
        origen = "OCI Object Storage"
    except Exception:
        texto = None

    if texto is None:
        if not RUTA_LOG.exists():
            return pd.DataFrame(), origen
        texto = RUTA_LOG.read_text(encoding="utf-8")

    registros = []
    for linea in texto.splitlines():
        linea = linea.strip()
        if linea:
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(registros)
    if df.empty:
        return df, origen
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df, origen


inter, origen = cargar_datos()
st.title("📊 Dashboard — Agente-CGT")
st.caption(f"Fuente de logs: {origen} · Actualizado: "
           f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

if inter.empty:
    st.warning("Aún no hay interacciones registradas en logs/metricas.jsonl.")
    st.stop()


# ---------------------------------------------------------------------------
# Filtro de fecha
# ---------------------------------------------------------------------------

rango = st.radio("Periodo", ["Hoy", "Últimos 7 días", "Últimos 30 días", "Todo"],
                 horizontal=True)
ahora = datetime.now(timezone.utc)
limites = {"Hoy": 1, "Últimos 7 días": 7, "Últimos 30 días": 30}
if rango in limites:
    inter_f = inter[inter["timestamp"] >= ahora - timedelta(days=limites[rango])]
else:
    inter_f = inter

if inter_f.empty:
    st.info("Sin datos en el periodo seleccionado.")
    st.stop()


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def porcentaje(parte, total):
    return (parte / total) if total else 0.0


total = len(inter_f)
tiempo_prom = inter_f["tiempo_respuesta"].mean()

respondidas = (inter_f["tipo_respuesta"] == "respondida").sum()
safeguard = (inter_f["tipo_respuesta"] == "safeguard").sum()
verificacion = (inter_f["tipo_respuesta"] == "verificacion").sum()

calif_validas = inter_f["calificacion"].dropna()
pct_negativo = (calif_validas == "negativo").mean() if len(calif_validas) else None

st.subheader("Observabilidad")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latencia promedio", f"{tiempo_prom:.2f} s")
c2.metric("Tasa de errores", "N/D")
c3.metric("Costo por petición", "N/D")
c4.metric("Tokens usados", "N/D")

st.subheader("Calidad")
c5, c6, c7, c8, c9 = st.columns(5)
c5.metric("Total de preguntas", total)
c6.metric("Respondidas", f"{porcentaje(respondidas, total):.1%}")
c7.metric("Bloqueadas (safeguard)", f"{porcentaje(safeguard, total):.1%}")
c8.metric("En verificación", f"{porcentaje(verificacion, total):.1%}")
c9.metric("Feedback negativo", f"{pct_negativo:.1%}"
          if pct_negativo is not None else "sin datos")

# Gráfica de preguntas por día
serie = inter_f.set_index("timestamp").resample("D").size()
st.bar_chart(serie, height=180)


# ---------------------------------------------------------------------------
# Alertas por anomalías
# ---------------------------------------------------------------------------

def _detectar_anomalias() -> list[str]:
    alertas = []
    if pd.notna(tiempo_prom) and tiempo_prom > UMBRAL_LATENCIA_S:
        alertas.append(f"Latencia promedio alta: {tiempo_prom:.1f}s "
                       f"(umbral {UMBRAL_LATENCIA_S}s)")
    return alertas


anomalias = _detectar_anomalias()
if anomalias:
    st.error("⚠️ **Anomalías detectadas:**\n\n" + "\n".join(f"- {a}" for a in anomalias))
else:
    st.success("✅ Sin anomalías en el periodo.")


# ---------------------------------------------------------------------------
# Tabla de preguntas recientes
# ---------------------------------------------------------------------------

st.subheader("Preguntas recientes")
tabla = inter_f.sort_values("timestamp", ascending=False)[
    ["timestamp", "pregunta", "tipo_respuesta", "calificacion", "tiempo_respuesta"]
].head(50).copy()
tabla["calificacion"] = tabla["calificacion"].map(
    {"positivo": "👍", "negativo": "👎"}).fillna("—")
tabla["timestamp"] = tabla["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
tabla = tabla.rename(columns={
    "timestamp": "Fecha",
    "pregunta": "Pregunta",
    "tipo_respuesta": "Tipo",
    "calificacion": "Calificación",
    "tiempo_respuesta": "Tiempo (s)",
})
st.dataframe(tabla, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Estado del pipeline de documentos
# ---------------------------------------------------------------------------

st.subheader("Pipeline de documentos")
if RUTA_MANIFIESTO.exists():
    manifiesto = json.loads(RUTA_MANIFIESTO.read_text(encoding="utf-8"))
    fecha = manifiesto.get("fecha") or manifiesto.get("timestamp") \
        or datetime.fromtimestamp(RUTA_MANIFIESTO.stat().st_mtime).isoformat()
    st.info(f"Última actualización del índice vectorial: **{fecha}**")
else:
    st.warning("No se encontró manifiesto_ingesta.json — el índice quizá "
               "no se ha construido.")

if RUTA_LOG_INGESTA.exists() and RUTA_LOG_INGESTA.stat().st_size > 0:
    with st.expander("Errores del proceso de ingesta"):
        st.code(RUTA_LOG_INGESTA.read_text(encoding="utf-8")[-3000:])
else:
    st.caption("Sin errores registrados en la ingesta.")
