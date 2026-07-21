"""
Re-OCR del CGT (PDF escaneado) en español, con limpieza de texto.

El PDF del CGT es un escaneo cuya capa de texto original es de mala calidad
(pasajes ilegibles como "salario asignadd;a,ﬁﬁs ﬁjomada"). Este script lo
vuelve a reconocer con tesseract en español, le quita el membrete repetido
de cada página ("GOBIERNO DE MÉXICO / SERVICIOS DE SALUD / IMSS-BIENESTAR")
y normaliza el texto. El resultado se guarda en
documentos/CGT_IMSS_BIENESTAR.ocr.txt.

Ese archivo de texto queda versionado y es el que lee el ingestor, de modo
que la ingesta (y el reindexado) NO necesitan tesseract ni el modelo de OCR:
solo hace falta correr este script una vez en local cuando cambie el PDF.

Uso (una sola vez, en local):
  1. Descargar el modelo español de tesseract (spa.traineddata) del repo
     oficial tesseract-ocr/tessdata_best y dejarlo en una carpeta, p. ej.
     ./tessdata/spa.traineddata
  2. TESSDATA_PREFIX=./tessdata python src/reocr_cgt.py

Requiere: pdftoppm (poppler-utils) y tesseract instalados en el sistema.
"""

import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

PDF_CGT = "documentos/CGT_IMSS_BIENESTAR.pdf"
SALIDA = "documentos/CGT_IMSS_BIENESTAR.ocr.txt"
DPI = 300
IDIOMA = "spa"

# Tokens del membrete (logotipo) del CGT. Cada página trae arriba el logo
# "GOBIERNO DE MÉXICO / SERVICIOS DE SALUD / IMSS IMSS-BIENESTAR", que el
# OCR reconoce con basura variable. Estos tres tokens en mayúsculas solo
# aparecen en el logo, NUNCA en el cuerpo: las menciones de cuerpo son
# fragmentos como "del IMSS-BIENESTAR." o "TITULAR DEL IMSS-BIENESTAR", y
# ninguna trae "GOBIERNO DE", "IMSS IMSS" ni un "MÉXICO" suelto. Por eso
# quitar las líneas CORTAS que contienen estos tokens elimina el membrete
# sin tocar el contenido ni los enumeradores de fracciones (I., II., ...).
TOKENS_MEMBRETE = re.compile(r"GOBIERNO\s*DE|IMSS\s+IMSS|M[ÉE]XICO", re.I)


def _es_membrete(linea):
    s = linea.strip()
    return len(s) <= 45 and bool(TOKENS_MEMBRETE.search(s))


def limpiar_texto(texto):
    """Normaliza el texto (ligaduras, caracteres de control), quita las
    líneas del membrete en cualquier posición y colapsa las líneas en
    blanco consecutivas."""
    texto = unicodedata.normalize("NFKC", texto)  # ﬁ->fi, ﬂ->fl, etc.
    texto = "".join(ch for ch in texto if ch >= " " or ch == "\n")
    salida = []
    for linea in texto.splitlines():
        if _es_membrete(linea):
            continue
        # Evitar acumular líneas en blanco seguidas.
        if not linea.strip() and (not salida or not salida[-1].strip()):
            continue
        salida.append(linea)
    return "\n".join(salida).strip()


def ocr_pdf(ruta_pdf, dpi=DPI, idioma=IDIOMA):
    """Devuelve la lista de textos (uno por página) tras render + OCR."""
    if not shutil.which("pdftoppm") or not shutil.which("tesseract"):
        raise RuntimeError(
            "Se requieren pdftoppm (poppler-utils) y tesseract instalados."
        )
    with tempfile.TemporaryDirectory() as tmp:
        prefijo = str(Path(tmp) / "pag")
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", ruta_pdf, prefijo],
            check=True,
        )
        paginas = []
        for png in sorted(Path(tmp).glob("pag-*.png")):
            salida = str(png.with_suffix(""))
            subprocess.run(
                ["tesseract", str(png), salida, "-l", idioma],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            paginas.append(Path(salida + ".txt").read_text(encoding="utf-8"))
        return paginas


def main():
    print(f"Re-OCR de {PDF_CGT} en '{IDIOMA}' a {DPI} DPI...")
    paginas = ocr_pdf(PDF_CGT)
    texto = limpiar_texto("\n\n".join(paginas))
    Path(SALIDA).write_text(texto, encoding="utf-8")
    print(f"Guardado: {SALIDA} ({len(paginas)} páginas, {len(texto)} chars)")


if __name__ == "__main__":
    main()
