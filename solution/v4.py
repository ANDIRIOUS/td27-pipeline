"""v4 (experimento, NO es candidata): v3 + reinicio con movimiento sostenido (A2).

    python solution/v4.py [--out evaluation/results/v4.csv]

Resultado, 9 nov: no pasó su criterio. Las posesiones cortas D-x-D bajan de 20 a 18, pero todos
los reinicios se anclan más tarde (mediana de 5.2 a 6.6 s): los jugadores tardan 1–2 s en correr
después de que el balón se pone en juego. Se queda en el repo para que el resultado sea
reproducible; la candidata sigue siendo la v3.

Dos problemas de la v3 con la misma causa: para dar por reanudado el juego bastaba cualquier
movimiento.
  1. Un trote antes del saque, o el balón acomodado antes del silbatazo, declaraba el reinicio y
     abría posesiones falsas de pocos segundos entre dos D.
  2. Los dos saques iniciales no se anclaban: el D explícito llegaba hasta el tag + 3 s, un desfase
     que se había comprobado mirando esos dos saques (zona gris: una constante sacada de momentos
     concretos del partido).

La v4 da por reanudado el juego en el primer movimiento (balón rodando dentro del campo o jugadores
pasando de caminar a correr) tras el cual los jugadores siguen corriendo SUSTAIN segundos. Se aplica
igual a los 59 SET PIECES y a los dos saques. Si en un reinicio no se ve ningún movimiento así, se
usa la mediana de los desfases vistos en el partido, calculada en el código (v1.restart_frames).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

# s: la velocidad del grupo ya viene suavizada a 2 s, así que un trote de menos de eso no alcanza a
# sostenerse; un reinicio real pone a correr a los jugadores durante varios segundos.
SUSTAIN = 2.0


def run(ball, players, team, pitch, tagging):
    score = v2.evidence(ball, players, team, pitch, tagging, phase_weight=v3.PHASE_WEIGHT,
                        sustain=SUSTAIN)
    crowd, _, _, _, live = v1.motion(ball, players, pitch, anchored=True)
    return v1.label(score, tagging, v1.restart_frames(tagging, crowd, live, SUSTAIN))


def predict():
    ball, players, team, pitch = load()
    return run(ball, players, team, pitch, load_tagging())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v4.csv"))
    args = parser.parse_args()
    labels = predict()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT),
          {k: int((labels == k).sum()) for k in "HADX"})
