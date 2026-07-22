"""
Reporte de calidad y métricas de uso del agente.

Lee logs/metricas.jsonl (generado por la app) y muestra un resumen:
total de preguntas, proporción respondidas vs bloqueadas por safeguard,
tiempo promedio de respuesta, preguntas sin respuesta más frecuentes y
preguntas con calificación negativa.

Uso:
  python src/reporte_calidad.py                 # lee el archivo local
  python src/reporte_calidad.py --fuente nube    # lee desde OCI Object Storage
"""

import argparse
import json
import os
from collections import Counter

ARCHIVO_METRICAS = "logs/metricas.jsonl"


def _parsear_jsonl(texto):
    return [json.loads(linea) for linea in texto.splitlines() if linea.strip()]


def cargar_metricas(ruta=ARCHIVO_METRICAS):
    """Lee las métricas del archivo local."""
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        return _parsear_jsonl(f.read())


def cargar_metricas_nube():
    """Lee las métricas del objeto metricas.jsonl en OCI Object Storage."""
    import monitor

    return _parsear_jsonl(monitor.descargar_metricas())


def porcentaje(parte, total):
    return (parte / total * 100) if total else 0.0


def generar_reporte(metricas):
    total = len(metricas)
    print("=" * 55)
    print("REPORTE DE CALIDAD Y MÉTRICAS DE USO")
    print("=" * 55)

    if total == 0:
        print("Aún no hay preguntas registradas en logs/metricas.jsonl.")
        return

    respondidas = [m for m in metricas if m["tipo_respuesta"] == "respondida"]
    safeguard = [m for m in metricas if m["tipo_respuesta"] == "safeguard"]
    verificacion = [m for m in metricas if m["tipo_respuesta"] == "verificacion"]

    print(f"Total de preguntas recibidas: {total}")
    print()
    print(f"  Respondidas:              {len(respondidas):3d}  "
          f"({porcentaje(len(respondidas), total):.1f}%)")
    print(f"  Bloqueadas por safeguard: {len(safeguard):3d}  "
          f"({porcentaje(len(safeguard), total):.1f}%)")
    print(f"  En verificación:          {len(verificacion):3d}  "
          f"({porcentaje(len(verificacion), total):.1f}%)")

    tiempos = [
        m["tiempo_respuesta"]
        for m in metricas
        if isinstance(m.get("tiempo_respuesta"), (int, float))
    ]
    promedio = sum(tiempos) / len(tiempos) if tiempos else 0.0
    print()
    print(f"Tiempo promedio de respuesta: {promedio:.2f} s")

    print()
    print("Preguntas sin respuesta más frecuentes (safeguard):")
    sin_respuesta = Counter(m["pregunta"] for m in safeguard)
    if sin_respuesta:
        for pregunta, veces in sin_respuesta.most_common(10):
            print(f"  [{veces}x] {pregunta}")
    else:
        print("  (ninguna)")

    print()
    print("Preguntas con calificación negativa (👎):")
    negativas = [m for m in metricas if m.get("calificacion") == "negativo"]
    if negativas:
        for m in negativas:
            print(f"  - {m['pregunta']}")
    else:
        print("  (ninguna)")


def main():
    parser = argparse.ArgumentParser(
        description="Reporte de calidad y métricas de uso del agente."
    )
    parser.add_argument(
        "--fuente",
        choices=["local", "nube"],
        default="local",
        help="De dónde leer las métricas: 'local' (archivo, por defecto) "
        "o 'nube' (OCI Object Storage).",
    )
    args = parser.parse_args()

    if args.fuente == "nube":
        try:
            metricas = cargar_metricas_nube()
        except Exception as e:
            print(f"No se pudieron leer las métricas desde la nube: {e}")
            return
    else:
        metricas = cargar_metricas()

    generar_reporte(metricas)


if __name__ == "__main__":
    main()
