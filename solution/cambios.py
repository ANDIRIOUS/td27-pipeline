"""Estudio del costo de cambio de equipo (A2c): ¿cuántas recuperaciones del rival nos tragamos y
cuántos cambios inventamos? Sin etiquetas y sobre el partido completo.

    python solution/cambios.py

Por qué hace falta. En el Viterbi, pasar de un equipo al otro cuesta SWITCH[H, A] = 20 nats, y el
control claro aporta ~1 nat por cuadro. Un cambio de una sola vía casi no se nota, porque la nueva
posesión paga el costo con todo lo que sigue. Pero una recuperación corta del rival (el balón va y
regresa) obliga a pagar dos cambios: con 20 nats, el rival necesita más de ~1.6 s de control claro
para que se le reconozca. En la v5 hay 25 tramos de 1 s o más de control claro del rival dentro de
posesiones nuestras.

El criterio sale de la definición del reto (CHALLENGE.md): un cambio de posesión exige control, no
basta un toque, un desvío o un duelo aéreo sin control después; y un indicio de control es que el
jugador conserve el balón. Nuestro indicio es un "episodio de control": EPISODE s seguidos de
control claro (v1.control). Con él se cuentan dos errores opuestos:
  - tragados: episodios de control del rival dentro de una posesión nuestra;
  - inventados: posesiones en las que el balón se ve al menos VISIBLE del tiempo y el equipo no
    tiene ningún episodio de control (si no se ve el balón, no se puede saber).

Regla fijada antes de correr: se elige el costo con la menor suma de tragados e inventados; si dos
empatan, el más alto, que cambia menos la v5. Se exige además que la prueba de huecos y el juez de
reinicios no empeoren más que su ruido.

Solo cambia el costo entre equipos (H <-> A). Los costos hacia y desde el balón muerto quedan
igual que en la v1.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import huecos  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v5  # noqa: E402
from atd import load  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = v1.FPS
# Costo de cambio entre equipos, en nats. Con ~1 nat por cuadro de control claro, una recuperación
# de ida y vuelta pide 2 * costo / 25 segundos de control: 1.6, 1.2, 1.0, 0.8, 0.6 y 0.4 s.
GRID = (20.0, 15.0, 12.5, 10.0, 7.5, 5.0)
EPISODE = 1.0    # s seguidos de control claro: "conservar el balón", en su lectura más corta
VISIBLE = 0.8    # fracción de la posesión con balón visible para poder juzgar si fue inventada


def switch_matrix(cost):
    m = v1.SWITCH.copy()
    m[v1.H, v1.A] = m[v1.A, v1.H] = cost
    return m


def run(cost):
    """La v5 con otro costo de cambio entre equipos: (ball, players, team, pitch, tagging) -> etiquetas."""
    def version(ball, players, team, pitch, tagging):
        score = v2.evidence(ball, players, team, pitch, tagging, phase_weight=v3.PHASE_WEIGHT)
        return v1.label(score, tagging, v5.kickoff_restarts(ball, pitch, tagging), switch_matrix(cost))
    return version


def longest_run(mask):
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return int((edges[1::2] - edges[::2]).max())


def episodes(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return int(((edges[1::2] - edges[::2]) >= EPISODE * FPS).sum())


def audit(labels, clear, seen):
    """Posesiones, cortas (< 2 s), tragados e inventados en lo calificado."""
    spells = short = swallowed = invented = 0
    for _, first, last in half_video_blocks(GAME):
        w = labels[first:last + 1]
        cuts = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
        for i in range(len(cuts) - 1):
            k = w[cuts[i]]
            if k not in "HA":
                continue
            s, e = first + cuts[i], first + cuts[i + 1]
            side = v1.H if k == "H" else v1.A
            spells += 1
            short += e - s < 2 * FPS
            swallowed += episodes(clear[s:e, 1 - side])
            if seen[s:e].mean() >= VISIBLE and longest_run(clear[s:e, side]) < EPISODE * FPS:
                invented += 1
    return spells, short, swallowed, invented


def main():
    ball, players, team, pitch = load()
    tagging = load_tagging()
    clear = v1.control(ball, players, team, pitch)
    seen = ~np.isnan(ball[:, 0])
    print("{:<8} {:>11} {:>8} {:>9} {:>11} {:>6}".format(
        "costo", "posesiones", "< 2 s", "tragados", "inventados", "suma"))
    audits = {}
    for cost in GRID:
        a = audit(run(cost)(ball, players, team, pitch, tagging), clear, seen)
        audits[cost] = a
        print("{:<8g} {:>11} {:>8} {:>9} {:>11} {:>6}".format(cost, *a, a[2] + a[3]))
    chosen = min(GRID, key=lambda c: (audits[c][2] + audits[c][3], -c))
    print("\nelegido por la regla: costo {:g}".format(chosen))

    # Guardas: prueba de huecos (equipo correcto en huecos simulados) con los mismos huecos.
    graded = v2.graded_mask(len(ball))
    crowd = np.nan_to_num(v1.crowd_speed(players), nan=0.0)
    truth = np.where(clear[:, v1.H], 0, np.where(clear[:, v1.A], 1, -1))
    lengths = np.array([e - s for s, e in huecos.live_gaps(ball, crowd, graded)])
    versions = {"costo {:g}".format(c): run(c) for c in GRID}
    results = huecos.simulate(versions, ball, players, team, pitch, tagging, graded, crowd, clear,
                              truth, lengths)
    huecos.table(results, "costo 20")


if __name__ == "__main__":
    main()
