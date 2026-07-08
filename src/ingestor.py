"""
Lee todos los PDFs de la carpeta documentos/, los divide en chunks
(guardando en cada chunk el documento de origen como metadato) y los
almacena en una base vectorial ChromaDB local persistente.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RUTA_DOCUMENTOS = "documentos"
RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "cgt_imss_bienestar"

# Nombre legible de cada documento, usado como fuente al citar.
# Si un PDF no está aquí, se usa el nombre del archivo sin extensión.
NOMBRES_FUENTES = {
    "CGT_IMSS_BIENESTAR": "Condiciones Generales de Trabajo IMSS Bienestar (CGT)",
    "LFTSE": "Ley Federal de los Trabajadores al Servicio del Estado (LFTSE)",
    "LISSSTE": "Ley del ISSSTE (LISSSTE)",
    "LGRA": "Ley General de Responsabilidades Administrativas (LGRA)",
    "Ley_de_Premios_Estimulos_y_Recompensas_Civiles": (
        "Ley de Premios, Estímulos y Recompensas Civiles"
    ),
}


def cargar_y_dividir_pdfs(ruta_documentos: str):
    """Carga cada PDF de la carpeta y devuelve {fuente: lista_de_chunks}."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks_por_fuente = {}
    pdfs = sorted(Path(ruta_documentos).glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No se encontraron PDFs en '{ruta_documentos}'.")

    for ruta_pdf in pdfs:
        fuente = NOMBRES_FUENTES.get(ruta_pdf.stem, ruta_pdf.stem)
        print(f"Cargando: {ruta_pdf.name}")

        paginas = PyPDFLoader(str(ruta_pdf)).load()
        chunks = splitter.split_documents(paginas)
        for chunk in chunks:
            chunk.metadata["fuente"] = fuente

        chunks_por_fuente[fuente] = chunks

    return chunks_por_fuente


# La API key de prueba (Trial) de Cohere permite máx. 40 llamadas y
# 100k tokens por minuto: se ingesta por lotes con pausa entre cada uno.
TAMANO_LOTE = 96
PAUSA_SEGUNDOS = 30


def construir_base_vectorial(chunks, ruta_chroma: str):
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    vectorstore = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=ruta_chroma,
    )

    total_lotes = (len(chunks) + TAMANO_LOTE - 1) // TAMANO_LOTE
    for i in range(0, len(chunks), TAMANO_LOTE):
        lote = chunks[i : i + TAMANO_LOTE]
        vectorstore.add_documents(lote)
        num_lote = i // TAMANO_LOTE + 1
        print(f"  Lote {num_lote}/{total_lotes} guardado ({len(lote)} chunks)")
        if num_lote < total_lotes:
            time.sleep(PAUSA_SEGUNDOS)

    return vectorstore


def main():
    if not os.getenv("COHERE_API_KEY"):
        raise RuntimeError(
            "Falta COHERE_API_KEY. Define la variable en tu archivo .env."
        )

    chunks_por_fuente = cargar_y_dividir_pdfs(RUTA_DOCUMENTOS)

    print("\nChunks generados por documento:")
    for fuente, chunks in chunks_por_fuente.items():
        print(f"  {fuente}: {len(chunks)} chunks")

    todos = [c for chunks in chunks_por_fuente.values() for c in chunks]
    print(f"\nTotal: {len(todos)} chunks.")

    print(f"Generando embeddings y guardando en ChromaDB ({RUTA_CHROMA})...")
    construir_base_vectorial(todos, RUTA_CHROMA)
    print("Base vectorial creada correctamente.")


if __name__ == "__main__":
    main()
