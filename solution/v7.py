"""v7: la v5 sin la evidencia de balón faltante (candidata segura).

    python solution/v7.py [--out evaluation/results/v7.csv]

Qué cambia respecto a la v5: solo una constante. La v1 a la v5 suman 0.3 nats por cuadro sin balón
hacia D; la v7 suma 0. Todo lo demás es la v5 (`muertos.run` es la v5 con esa constante).

Por qué: cada cuadro sin balón contaba como una observación independiente, pero los huecos del
tracker duran una mediana de 10 s, así que un solo hueco acumulaba ~75 nats hacia D contra 20 que
cuesta entrar a D. La regla fijada antes de correr en `muertos.py` (#30) eligió 0.0: el muerto falso
en la prueba de huecos baja de 16.9% a 11.8% y las guardas de balón muerto seguro no se mueven.

Por qué se llama v7: la v6 (#33) es esto mismo MÁS el arrastre del balón suelto, que está en
disputa. La v7 es la v6 sin arrastre, y no depende de `arrastre.py` ni de `v6.py`.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import muertos  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

# Nats por cuadro sin balón hacia D. No ver el balón no dice nada sobre vivo o muerto: falta en el
# 41% de lo calificado, en huecos de mediana 10 s, y no solo cuando el balón está muerto. Con 0.3, un
# hueco acumulaba ~75 nats hacia D contra 20 que cuesta entrar a D. La eligió la regla pre-registrada
# de `muertos.py` (#30), no un ajuste a mano.
MISSING_EVIDENCE = 0.0
run = muertos.run(MISSING_EVIDENCE)


def predict():
    ball, players, team, pitch = load()
    return run(ball, players, team, pitch, load_tagging())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v7.csv"))
    args = parser.parse_args()
    labels = predict()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT),
          {k: int((labels == k).sum()) for k in "HADX"})
