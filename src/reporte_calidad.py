"""
Reporte de calidad y métricas de uso del agente.

Lee logs/metricas.jsonl (generado por la app) y muestra un resumen:
total de preguntas, proporción respondidas vs bloqueadas por safeguard,
tiempo promedio de respuesta, preguntas sin respuesta más frecuentes y
preguntas con calificación negativa.

Uso: python src/reporte_calidad.py
"""

import json
import os
from collections import Counter

ARCHIVO_METRICAS = "logs/metricas.jsonl"


def cargar_metricas(ruta=ARCHIVO_METRICAS):
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


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
    generar_reporte(cargar_metricas())


if __name__ == "__main__":
    main()
