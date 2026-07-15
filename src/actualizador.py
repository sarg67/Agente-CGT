"""
Curaduría y sincronización de documentos.

Compara los PDFs de documentos/ contra lo que hay indexado en ChromaDB
(usando el manifiesto de hashes) y detecta documentos nuevos, modificados
o eliminados. Si hay cambios, actualiza SOLO los documentos afectados
en la base vectorial (reingesta incremental), sin reprocesar el resto.
Genera un reporte de curaduría en logs/reporte_curaduria.txt.

Uso: python src/actualizador.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestor import (
    NOMBRE_COLECCION,
    NOMBRES_FUENTES,
    PAUSA_SEGUNDOS,
    RUTA_CHROMA,
    RUTA_DOCUMENTOS,
    RUTA_MANIFIESTO,
    TAMANO_LOTE,
    calcular_hashes,
)

load_dotenv()

RUTA_REPORTE = "logs/reporte_curaduria.txt"

# Registro de fechas de modificación (mtime) por archivo. Es aparte del
# manifiesto de hashes: permite detectar que un documento fue "tocado"
# (re-guardado o reemplazado) aunque su contenido sea idéntico, algo
# relevante para la curaduría (¿sigue siendo la versión oficial?).
RUTA_MTIMES = "manifiesto_mtime.json"


def leer_manifiesto():
    """Devuelve {archivo: hash} de la última ingesta, o {} si no existe."""
    if not os.path.exists(RUTA_MANIFIESTO):
        return {}
    with open(RUTA_MANIFIESTO, encoding="utf-8") as f:
        return json.load(f)


def escribir_manifiesto(hashes):
    with open(RUTA_MANIFIESTO, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)


def calcular_mtimes():
    """Devuelve {archivo: fecha_de_modificación} de los PDFs actuales."""
    mtimes = {}
    for ruta in sorted(Path(RUTA_DOCUMENTOS).glob("*.pdf")):
        mtimes[ruta.name] = ruta.stat().st_mtime
    return mtimes


def leer_mtimes():
    if not os.path.exists(RUTA_MTIMES):
        return {}
    with open(RUTA_MTIMES, encoding="utf-8") as f:
        return json.load(f)


def escribir_mtimes(mtimes):
    with open(RUTA_MTIMES, "w", encoding="utf-8") as f:
        json.dump(mtimes, f, indent=2, ensure_ascii=False)


def detectar_tocados(hashes_act, hashes_prev, mtimes_act, mtimes_prev):
    """Archivos cuya fecha cambió pero cuyo contenido es idéntico."""
    tocados = []
    for archivo in hashes_act:
        mismo_contenido = (
            archivo in hashes_prev and hashes_act[archivo] == hashes_prev[archivo]
        )
        fecha_cambio = (
            archivo in mtimes_prev and mtimes_act.get(archivo) != mtimes_prev[archivo]
        )
        if mismo_contenido and fecha_cambio:
            tocados.append(archivo)
    return sorted(tocados)


def listar_documentos():
    """Lista los PDFs de documentos/ con su fecha de modificación."""
    docs = []
    for ruta in sorted(Path(RUTA_DOCUMENTOS).glob("*.pdf")):
        mod = datetime.fromtimestamp(ruta.stat().st_mtime)
        docs.append((ruta.name, mod.strftime("%Y-%m-%d %H:%M:%S")))
    return docs


def detectar_cambios(hashes_actuales, hashes_previos):
    """Clasifica los documentos en nuevos, modificados y eliminados."""
    nuevos = [f for f in hashes_actuales if f not in hashes_previos]
    eliminados = [f for f in hashes_previos if f not in hashes_actuales]
    modificados = [
        f
        for f in hashes_actuales
        if f in hashes_previos and hashes_actuales[f] != hashes_previos[f]
    ]
    return sorted(nuevos), sorted(modificados), sorted(eliminados)


def ruta_documento(archivo):
    """Ruta tal como se guarda en el metadato 'source' de cada chunk."""
    return str(Path(RUTA_DOCUMENTOS) / archivo)


def cargar_chunks(archivo):
    """Carga un PDF y devuelve sus chunks con el metadato de fuente."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    paginas = PyPDFLoader(ruta_documento(archivo)).load()
    chunks = splitter.split_documents(paginas)
    fuente = NOMBRES_FUENTES.get(Path(archivo).stem, Path(archivo).stem)
    for chunk in chunks:
        chunk.metadata["fuente"] = fuente
    return chunks


def agregar_por_lotes(vectorstore, chunks):
    """Agrega chunks en lotes con pausa, respetando el límite de la API."""
    total_lotes = (len(chunks) + TAMANO_LOTE - 1) // TAMANO_LOTE
    for i in range(0, len(chunks), TAMANO_LOTE):
        lote = chunks[i : i + TAMANO_LOTE]
        vectorstore.add_documents(lote)
        num_lote = i // TAMANO_LOTE + 1
        if num_lote < total_lotes:
            time.sleep(PAUSA_SEGUNDOS)


def abrir_base():
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    return Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=RUTA_CHROMA,
    )


def aplicar_cambios(nuevos, modificados, eliminados):
    """Actualiza en ChromaDB solo los documentos afectados.

    Un PDF corrupto o vacío no debe tumbar toda la ejecución: cada
    documento se procesa con manejo de errores. Devuelve (acciones,
    fallidos): las acciones legibles y la lista de archivos que no se
    pudieron procesar (para no registrarlos en el manifiesto y que se
    reintenten en la siguiente corrida).
    """
    acciones = []
    fallidos = []
    vectorstore = abrir_base()
    coleccion = vectorstore._collection

    # Eliminar de la base los documentos borrados (por su ruta 'source').
    for archivo in eliminados:
        coleccion.delete(where={"source": ruta_documento(archivo)})
        acciones.append(f"Eliminados de la base los fragmentos de '{archivo}'.")

    for archivo in modificados:
        # Cargar ANTES de borrar: si el PDF nuevo falla, se conserva la
        # versión anterior en lugar de dejar el documento sin fragmentos.
        try:
            chunks = cargar_chunks(archivo)
        except Exception as error:
            fallidos.append(archivo)
            acciones.append(
                f"ERROR al reindexar '{archivo}': {error}. "
                "Se conserva la versión anterior."
            )
            continue
        coleccion.delete(where={"source": ruta_documento(archivo)})
        agregar_por_lotes(vectorstore, chunks)
        acciones.append(
            f"Reindexado '{archivo}' ({len(chunks)} fragmentos) por modificación."
        )

    for archivo in nuevos:
        try:
            chunks = cargar_chunks(archivo)
        except Exception as error:
            fallidos.append(archivo)
            acciones.append(
                f"ERROR al indexar documento nuevo '{archivo}': {error}. "
                "Documento omitido (archivo inválido o vacío)."
            )
            continue
        agregar_por_lotes(vectorstore, chunks)
        acciones.append(
            f"Indexado documento nuevo '{archivo}' ({len(chunks)} fragmentos)."
        )

    return acciones, fallidos


def escribir_reporte(documentos, nuevos, modificados, eliminados, tocados,
                     acciones):
    os.makedirs(os.path.dirname(RUTA_REPORTE), exist_ok=True)
    lineas = []
    lineas.append("REPORTE DE CURADURÍA DE DOCUMENTOS")
    lineas.append("=" * 50)
    lineas.append(f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lineas.append("")
    lineas.append("Documentos revisados:")
    for nombre, fecha_mod in documentos:
        lineas.append(f"  - {nombre} (modificado: {fecha_mod})")
    lineas.append("")

    hay_cambios = nuevos or modificados or eliminados or tocados
    if not hay_cambios:
        lineas.append("Cambios detectados: ninguno.")
        lineas.append("")
        lineas.append("Base vectorial actualizada. Sin cambios detectados.")
    else:
        lineas.append("Cambios detectados:")
        lineas.append(f"  Nuevos:      {nuevos or 'ninguno'}")
        lineas.append(f"  Modificados: {modificados or 'ninguno'}")
        lineas.append(f"  Eliminados:  {eliminados or 'ninguno'}")
        lineas.append(f"  Tocados (fecha cambió, contenido igual): "
                      f"{tocados or 'ninguno'}")
        lineas.append("")
        lineas.append("Acciones tomadas:")
        for accion in acciones:
            lineas.append(f"  - {accion}")

    with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


def main():
    if not os.getenv("COHERE_API_KEY"):
        raise RuntimeError(
            "Falta COHERE_API_KEY. Define la variable en tu archivo .env."
        )

    documentos = listar_documentos()
    hashes_actuales = calcular_hashes(RUTA_DOCUMENTOS)
    hashes_previos = leer_manifiesto()
    mtimes_actuales = calcular_mtimes()
    mtimes_previos = leer_mtimes()
    nuevos, modificados, eliminados = detectar_cambios(
        hashes_actuales, hashes_previos
    )
    tocados = detectar_tocados(
        hashes_actuales, hashes_previos, mtimes_actuales, mtimes_previos
    )

    print("Documentos revisados:")
    for nombre, fecha_mod in documentos:
        print(f"  - {nombre} (modificado: {fecha_mod})")

    if not (nuevos or modificados or eliminados or tocados):
        print("\nBase vectorial actualizada. Sin cambios detectados.")
        escribir_reporte(documentos, [], [], [], [], [])
        escribir_mtimes(mtimes_actuales)
        return

    print("\nCambios detectados:")
    print(f"  Nuevos:      {nuevos or 'ninguno'}")
    print(f"  Modificados: {modificados or 'ninguno'}")
    print(f"  Eliminados:  {eliminados or 'ninguno'}")
    print(f"  Tocados (fecha cambió, contenido igual): {tocados or 'ninguno'}")

    acciones = []
    fallidos = []
    if nuevos or modificados or eliminados:
        if not os.path.isdir(RUTA_CHROMA):
            raise RuntimeError(
                f"No existe la base vectorial '{RUTA_CHROMA}'. "
                "Corre primero: python src/ingestor.py"
            )
        print("\nAplicando cambios en la base vectorial...")
        acciones, fallidos = aplicar_cambios(nuevos, modificados, eliminados)

    for archivo in tocados:
        acciones.append(
            f"'{archivo}': la fecha cambió pero el contenido es idéntico; "
            "no se reindexa. Revisar si sigue siendo la versión oficial."
        )

    # Los documentos que fallaron no se registran en el manifiesto: así
    # se reintentan en la siguiente corrida en vez de darse por hechos.
    manifiesto = {
        archivo: hash_
        for archivo, hash_ in hashes_actuales.items()
        if archivo not in fallidos
    }
    escribir_manifiesto(manifiesto)
    escribir_mtimes(mtimes_actuales)
    escribir_reporte(
        documentos, nuevos, modificados, eliminados, tocados, acciones
    )

    print("\nAcciones tomadas:")
    for accion in acciones:
        print(f"  - {accion}")
    print(f"\nReporte guardado en {RUTA_REPORTE}")


if __name__ == "__main__":
    main()
