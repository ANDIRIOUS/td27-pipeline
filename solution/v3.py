"""v3: v2 con el peso de las fases del Smart Tagging elegido por la prueba de huecos (A1).

    python solution/v3.py [--out evaluation/results/v3.csv]

En v1 y v2, cada cuadro dentro de una fase con equipo suma PHASE_EVIDENCE = 0.1 nats, como si
cada cuadro fuera una observación independiente. Una fase mediana dura ~22 s, así que una sola
fase sumaba ~54 nats: más que dos cambios de posesión (20 nats cada uno), con una fuente que
coincide con el control claro solo el 72% de los cuadros.

El peso nuevo salió de `huecos.py --barrido`:
  - Se probaron ocho pesos de 0 a 0.1, fijados antes de correr.
  - Cada uno se midió en 20 réplicas de huecos simulados.
  - Se eligió el más bajo dentro de un error estándar del mejor (ver solution/README.md).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import v1  # noqa: E402
import v2  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

# Resultado del barrido (equipo correcto en huecos simulados, 20 réplicas):
#   0 -> 59.4%   0.008 -> 60.5%   0.03 -> 61.7%   0.05 -> 62.0%   0.1 -> 62.3%
# El mejor es 0.1, y 0.05 queda a -0.2 ± 1.7 puntos pareados, dentro de un error estándar (0.4).
# Se toma 0.05 por la regla fijada antes de correr: la prueba puede favorecer a las fases, porque en
# los huecos simulados Metrica sí vio el balón. En los balones muertos seguros (balón fuera,
# jugadores caminando, antes de un reinicio) el peso casi no cambia nada: 96–98% de D con
# cualquier peso.
PHASE_WEIGHT = 0.05


def predict():
    ball, players, team, pitch = load()
    tagging = load_tagging()
    return v1.label(v2.evidence(ball, players, team, pitch, tagging, phase_weight=PHASE_WEIGHT),
                    tagging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v3.csv"))
    args = parser.parse_args()
    labels = predict()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT),
          {k: int((labels == k).sum()) for k in "HADX"})
