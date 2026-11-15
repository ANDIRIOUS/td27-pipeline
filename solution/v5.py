"""v5: v3 + saques iniciales anclados con el juez de reinicios.

    python solution/v5.py [--out evaluation/results/v5.csv]

Hasta la v3, el D explícito antes de cada saque inicial llegaba al tag + 3 s. Ese desfase se había
comprobado a mano mirando los dos saques: una constante sacada de momentos concretos del partido.
La v5 la reemplaza por una regla:
  1. Si el juez (reinicios.kick_frame) ve el golpe del saque, es decir, el primer arranque del
     balón a más de 5 m/s en el punto central, el D llega hasta ahí.
  2. Si no lo ve, se usa la mediana de los desfases de los saques de centro que sí detectó en el
     partido (los dos iniciales y los posteriores a cada gol), calculada aquí, no fijada a mano.
Todo lo demás es la v3.

Ojo: como la v5 usa el juez para sus saques, el juez ya no sirve para evaluar esos dos saques de la
v5. Sigue sirviendo para los córners y los saques de meta, que la v5 no toca.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import reinicios  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402


def kickoff_restarts(ball, pitch, tagging):
    """{start de cada saque inicial: (cuadro del golpe, visto)}, en el formato de v1.label."""
    speed = reinicios.ball_speed(ball, pitch)
    kicks = {}
    for e in tagging:
        kind = (e["tags"].get("Type") or "").strip().lower()
        if e["code"] == "SET PIECES" and kind.startswith("kick off"):
            kicks[e["start_frame"]] = (kind, reinicios.kick_frame(speed, ball, e["start_frame"],
                                                                  kind, pitch))
    delay = int(np.median([k - s for s, (_, k) in kicks.items() if k is not None]))
    return {s: (k, True) if k is not None else (s + delay, False)
            for s, (kind, k) in kicks.items() if kind == "kick off (start)"}


def run(ball, players, team, pitch, tagging):
    score = v2.evidence(ball, players, team, pitch, tagging, phase_weight=v3.PHASE_WEIGHT)
    return v1.label(score, tagging, kickoff_restarts(ball, pitch, tagging))


def predict():
    ball, players, team, pitch = load()
    return run(ball, players, team, pitch, load_tagging())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v5.csv"))
    args = parser.parse_args()
    labels = predict()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT),
          {k: int((labels == k).sum()) for k in "HADX"})
