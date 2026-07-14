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


def leer_manifiesto():
    """Devuelve {archivo: hash} de la última ingesta, o {} si no existe."""
    if not os.path.exists(RUTA_MANIFIESTO):
        return {}
    with open(RUTA_MANIFIESTO, encoding="utf-8") as f:
        return json.load(f)


def escribir_manifiesto(hashes):
    with open(RUTA_MANIFIESTO, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)


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

    Devuelve una lista de acciones tomadas (texto legible).
    """
    acciones = []
    vectorstore = abrir_base()
    coleccion = vectorstore._collection

    # Eliminar de la base los documentos borrados y las versiones viejas
    # de los modificados (se borran por su ruta 'source').
    for archivo in eliminados:
        coleccion.delete(where={"source": ruta_documento(archivo)})
        acciones.append(f"Eliminados de la base los fragmentos de '{archivo}'.")

    for archivo in modificados:
        coleccion.delete(where={"source": ruta_documento(archivo)})
        chunks = cargar_chunks(archivo)
        agregar_por_lotes(vectorstore, chunks)
        acciones.append(
            f"Reindexado '{archivo}' ({len(chunks)} fragmentos) por modificación."
        )

    for archivo in nuevos:
        chunks = cargar_chunks(archivo)
        agregar_por_lotes(vectorstore, chunks)
        acciones.append(
            f"Indexado documento nuevo '{archivo}' ({len(chunks)} fragmentos)."
        )

    return acciones


def escribir_reporte(documentos, nuevos, modificados, eliminados, acciones):
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

    hay_cambios = nuevos or modificados or eliminados
    if not hay_cambios:
        lineas.append("Cambios detectados: ninguno.")
        lineas.append("")
        lineas.append("Base vectorial actualizada. Sin cambios detectados.")
    else:
        lineas.append("Cambios detectados:")
        lineas.append(f"  Nuevos:      {nuevos or 'ninguno'}")
        lineas.append(f"  Modificados: {modificados or 'ninguno'}")
        lineas.append(f"  Eliminados:  {eliminados or 'ninguno'}")
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
    nuevos, modificados, eliminados = detectar_cambios(
        hashes_actuales, hashes_previos
    )

    print("Documentos revisados:")
    for nombre, fecha_mod in documentos:
        print(f"  - {nombre} (modificado: {fecha_mod})")

    if not (nuevos or modificados or eliminados):
        print("\nBase vectorial actualizada. Sin cambios detectados.")
        escribir_reporte(documentos, [], [], [], [])
        return

    print("\nCambios detectados:")
    print(f"  Nuevos:      {nuevos or 'ninguno'}")
    print(f"  Modificados: {modificados or 'ninguno'}")
    print(f"  Eliminados:  {eliminados or 'ninguno'}")

    if not os.path.isdir(RUTA_CHROMA):
        raise RuntimeError(
            f"No existe la base vectorial '{RUTA_CHROMA}'. "
            "Corre primero: python src/ingestor.py"
        )

    print("\nAplicando cambios en la base vectorial...")
    acciones = aplicar_cambios(nuevos, modificados, eliminados)
    escribir_manifiesto(hashes_actuales)
    escribir_reporte(documentos, nuevos, modificados, eliminados, acciones)

    print("\nAcciones tomadas:")
    for accion in acciones:
        print(f"  - {accion}")
    print(f"\nReporte guardado en {RUTA_REPORTE}")


if __name__ == "__main__":
    main()
