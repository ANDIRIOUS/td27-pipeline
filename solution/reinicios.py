"""Juez de reinicios: ¿cuándo se reanuda de verdad el juego en un balón parado? Sin etiquetas.

    python solution/reinicios.py [evaluation/results/v3.csv ...]

Por qué hace falta. `diagnose.py` cuenta un reinicio como "bien" si hay D cerca del start del tag y
juego vivo entre 5 y 8 s después: supone de entrada que el juego se reanuda ~3 s después del tag.
Con eso no se puede juzgar una regla que mueve el reinicio (A2, #23).

La referencia aquí es física y no supone ningún desfase: el golpe del balón en los SET PIECES
cuyo lugar fija el reglamento (saque de centro, córner, saque de meta).
  1. Desde START_BEFORE s antes del start del tag hasta RESTART_SEARCH s después, se buscan los
     arranques del balón: cuadros donde su velocidad pasa de KICK_SPEED viniendo de menos o de no
     verse.
  2. El golpe es el primer arranque que ocurre en el lugar de la jugada: el centro en un saque de
     centro, una esquina en un córner, el área chica en un saque de meta (posición más cercana en
     ±PLACE_WINDOW s alrededor del arranque).
Quedan fuera los saques de banda (el balón va en las manos) y los tiros libres (pueden ser en
cualquier lugar, así que el lugar no confirma nada).

Primera versión (commit a4aaac7): buscaba un tramo de balón quieto seguido del golpe. No detectó
ninguno de los 33 balones parados, porque el ATD casi nunca ve el balón quieto: mientras está
colocado el tracker lo pierde, o lo ve en movimiento mientras lo acomodan. Se cambió la detección
antes de medir cualquier regla contra el juez.

Con el juez mide el error, en segundos (positivo = tarde), de:
  - las anclas de reinicio de cada regla (v1.restart_frames): cualquier movimiento (v2, v3) y
    movimiento sostenido 2 s (v4), además de dos desfases fijos (start + 3 s y start + la mediana
    de las anclas de v3);
  - el reinicio que sale en las etiquetas de cada CSV: el primer paso de D a H/A en la ventana.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import v1  # noqa: E402
import v4  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = v1.FPS
PLACED = {"corner kick", "goal kick", "kick off (start)", "kick off (after goal)"}
START_BEFORE = 5.0     # s: por si el tag empieza después del golpe
SPEED_SMOOTH = 0.2     # s: mediana corta, para quitar ruido sin correr el golpe más de 0.1 s
KICK_SPEED = 5.0       # m/s: un pase corto ya va más rápido; un toque para acomodarlo no
PLACE_WINDOW = 0.2     # s: a 20 m/s el balón avanza ~4 m en 0.2 s; mirar más lejos confunde lugares
# Golpe "fuerte": el balón llega a STRONG_SPEED en el primer medio segundo. Un arranque que apenas
# cruza KICK_SPEED puede ser el portero acomodando el balón, no el golpe. Se mide aparte.
STRONG_SPEED = 10.0
STRONG_WINDOW = 0.5
# Dónde debe caer el golpe según el tipo, en metros desde el centro del campo.
CENTER_RADIUS = 3.0    # saque de centro: el balón está en el punto central
CORNER_RADIUS = 3.0    # córner: en el cuarto de círculo, ~1 m de la esquina
GOAL_AREA = (7.0, 11.0)  # saque de meta: área chica (5.5 m x 18.3 m) con ~1.5 m de margen


def ball_speed(ball, pitch):
    """Velocidad del balón por cuadro (m/s), con una mediana corta; NaN donde no se ve."""
    length, width = pitch
    speed = np.hypot(np.diff(ball[:, 0] * length, prepend=np.nan),
                     np.diff(ball[:, 1] * width, prepend=np.nan)) * FPS
    return pd.Series(speed).rolling(int(SPEED_SMOOTH * FPS) | 1, center=True).median().to_numpy()


def kick_frame(speed, ball, start, kind, pitch):
    """Primer arranque del balón en el lugar de la jugada, dentro de la ventana del tag, o None."""
    length, width = pitch
    a, b = max(start - int(START_BEFORE * FPS), 1), start + v1.RESTART_SEARCH * FPS
    fast = speed > KICK_SPEED                              # NaN (sin balón) cuenta como no rápido
    onsets = a + np.flatnonzero(fast[a:b] & ~fast[a - 1:b - 1])
    w = int(PLACE_WINDOW * FPS)
    for f in onsets:
        near = ball[f - w:f + w + 1]
        near = near[~np.isnan(near[:, 0])]
        if any(in_place(kind, (x - 0.5) * length, (y - 0.5) * width, pitch) for x, y in near[:, :2]):
            return int(f)
    return None


def in_place(kind, x, y, pitch):
    """¿El punto (x, y), en metros desde el centro, está donde lo pide el tipo de jugada?"""
    length, width = pitch
    if kind.startswith("kick off"):
        return np.hypot(x, y) < CENTER_RADIUS
    if kind == "corner kick":
        return np.hypot(abs(x) - length / 2, abs(y) - width / 2) < CORNER_RADIUS
    return abs(x) > length / 2 - GOAL_AREA[0] and abs(y) < GOAL_AREA[1]   # saque de meta


def label_restart(lab, start):
    """Primer paso de D a H/A en la ventana del tag, según las etiquetas; None si no hay."""
    a, b = max(start - int(START_BEFORE * FPS), 1), start + v1.RESTART_SEARCH * FPS
    w = np.flatnonzero(np.isin(lab[a:b], ["H", "A"]) & (lab[a - 1:b - 1] == "D"))
    return a + int(w[0]) if len(w) else None


def summary(name, errors, total):
    e = np.array([x for x in errors if x is not None]) / FPS
    print("{:<34} {:>3}/{:<3} {:>+8.1f} {:>9.1f} {:>9.0%} {:>9.0%}".format(
        name, len(e), total, np.median(e), np.median(np.abs(e)),
        np.mean(np.abs(e) <= 1), np.mean(np.abs(e) <= 2)))


def main(paths):
    ball, players, team, pitch = load()
    tagging = load_tagging()
    length, width = pitch
    speed = ball_speed(ball, pitch)
    pieces = [e for e in tagging if e["code"] == "SET PIECES"]
    placed = [e for e in pieces if (e["tags"].get("Type") or "").strip().lower() in PLACED]

    judged = {}
    print("golpes detectados por tipo:")
    for kind in sorted({(e["tags"].get("Type") or "").strip().lower() for e in placed}):
        events = [e for e in placed if (e["tags"].get("Type") or "").strip().lower() == kind]
        for e in events:
            k = kick_frame(speed, ball, e["start_frame"], kind, pitch)
            if k is not None:
                judged[e["start_frame"]] = k
        print("  {:<22} {:>2} de {:>2}".format(
            kind, sum(e["start_frame"] in judged for e in events), len(events)))
    strong = {s: k for s, k in judged.items()
              if np.nanmax(speed[k:k + int(STRONG_WINDOW * FPS)]) >= STRONG_SPEED}
    print("  fuertes (>= {:.0f} m/s en {:.1f} s): {} de {}".format(
        STRONG_SPEED, STRONG_WINDOW, len(strong), len(judged)))
    delays = np.array([k - s for s, k in judged.items()]) / FPS
    print("golpe respecto al start del tag: mediana {:+.1f} s, p25 {:+.1f}, p75 {:+.1f}".format(
        np.median(delays), *np.percentile(delays, [25, 75])))

    crowd, _, _, _, live = v1.motion(ball, players, pitch, anchored=True)
    rules = {
        "cualquier movimiento (v2, v3)": v1.restart_frames(tagging, crowd, live, 0.0),
        "sostenido 2 s (v4)": v1.restart_frames(tagging, crowd, live, v4.SUSTAIN),
    }
    median_v3 = int(np.median([f - s for s, (f, _) in rules["cualquier movimiento (v2, v3)"].items()]))
    labels = {os.path.basename(p): pd.read_csv(p).label.str.strip().str.upper().to_numpy()
              for p in paths}
    for title, ref in (("todos los golpes", judged), ("solo golpes fuertes", strong)):
        print("\nerror contra el golpe, en s (positivo = tarde), {}:".format(title))
        print("{:<34} {:>7} {:>8} {:>9} {:>9} {:>9}".format(
            "regla", "n", "mediana", "|mediana|", "<= 1 s", "<= 2 s"))
        summary("start + 3 s fijo", [s + 3 * FPS - k for s, k in ref.items()], len(ref))
        summary("start + {:.1f} s (mediana v3)".format(median_v3 / FPS),
                [s + median_v3 - k for s, k in ref.items()], len(ref))
        for name, r in rules.items():
            summary("ancla: " + name, [r[s][0] - k for s, k in ref.items()], len(ref))
        for name, lab in labels.items():
            errors = [None if f is None else f - k
                      for f, k in ((label_restart(lab, s), k) for s, k in ref.items())]
            summary("etiquetas: " + name, errors, len(ref))


if __name__ == "__main__":
    main(sys.argv[1:])
