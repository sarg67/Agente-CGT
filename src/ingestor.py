"""
Lee el PDF de las Condiciones Generales de Trabajo (CGT), lo divide en
chunks y los guarda en una base vectorial ChromaDB local persistente.
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RUTA_PDF = "documentos/CGT_IMSS_BIENESTAR.pdf"
RUTA_CHROMA = "chroma_db"
NOMBRE_COLECCION = "cgt_imss_bienestar"


def cargar_y_dividir_pdf(ruta_pdf: str):
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(paginas)


def construir_base_vectorial(chunks, ruta_chroma: str):
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=NOMBRE_COLECCION,
        persist_directory=ruta_chroma,
    )


def main():
    if not os.getenv("COHERE_API_KEY"):
        raise RuntimeError(
            "Falta COHERE_API_KEY. Define la variable en tu archivo .env."
        )

    print(f"Cargando PDF desde: {RUTA_PDF}")
    chunks = cargar_y_dividir_pdf(RUTA_PDF)
    print(f"Documento dividido en {len(chunks)} chunks.")

    print(f"Generando embeddings y guardando en ChromaDB ({RUTA_CHROMA})...")
    construir_base_vectorial(chunks, RUTA_CHROMA)
    print("Base vectorial creada correctamente.")


if __name__ == "__main__":
    main()
