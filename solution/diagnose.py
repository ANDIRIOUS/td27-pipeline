"""Diagnóstico sin etiquetas sobre el partido completo: con esto se comparan versiones.

    python solution/diagnose.py evaluation/results/v1.csv [otra.csv ...]

Ninguna cifra usa ground truth. Mide:
  - posesiones: cuántas y cuánto duran (un partido real tiene varios cientos, de ~10-20 s)
  - % de cuadros calificados en D (en LaLiga el balón está vivo ~55% del tiempo)
  - acuerdo de equipo con las fases del Smart Tagging (automático, no humano): de los cuadros
    vivos dentro de una fase con equipo, qué fracción le damos a ese equipo
  - acuerdo-sin-balón: lo mismo, pero solo en cuadros donde el tracker no ve el balón; es donde
    v1 solo podía mantener al último equipo
  - reinicios: fracción de SET PIECES con D en [start-2 s, start+2 s] y vivo en [start+5 s, +8 s]
  - cortas D-x-D: posesiones de menos de 5 s con D antes y después; muchas suelen ser falsos
    arranques (un trote antes de que se reanude el juego)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "starter")]
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402
from v1 import ATTACK_CODES, DEFEND_CODES, FPS, TEAM_OF  # noqa: E402


def report(path, tagging):
    lab = pd.read_csv(path).label.str.strip().str.upper().to_numpy()
    graded = np.zeros(len(lab), dtype=bool)
    spells = []
    short = 0
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
        w = lab[first:last + 1]
        cuts = np.r_[0, np.flatnonzero(w[1:] != w[:-1]) + 1, len(w)]
        spells += [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1) if w[cuts[i]] in "HA"]
        short += sum(1 for i in range(1, len(cuts) - 2) if w[cuts[i]] in "HA"
                     and w[cuts[i] - 1] == "D" and w[cuts[i + 1]] == "D"
                     and cuts[i + 1] - cuts[i] < 5 * FPS)
    spells = np.array(spells) / FPS

    agree = total = agree_nb = total_nb = 0
    ball = np.load(os.path.join(os.path.dirname(HERE), "data", "cache", "atd_tactical.npz"))["ball"]
    missing = np.isnan(ball[:, 0])
    hits = 0
    pieces = [e for e in tagging if e["code"] == "SET PIECES"]
    for e in tagging:
        side = TEAM_OF.get((e["tags"].get("Team") or "").strip().lower())
        if side is None or e["code"] not in ATTACK_CODES | DEFEND_CODES:
            continue
        side = 1 - side if e["code"] in DEFEND_CODES else side
        w = lab[e["start_frame"]:e["end_frame"]]
        alive = (w == "H") | (w == "A")
        agree += int((w[alive] == "HA"[side]).sum())
        total += int(alive.sum())
        nb = alive & missing[e["start_frame"]:e["end_frame"]]
        agree_nb += int((w[nb] == "HA"[side]).sum())
        total_nb += int(nb.sum())
    for e in pieces:
        s = e["start_frame"]
        hits += ("D" in lab[s - 2 * FPS:s + 2 * FPS]) and bool(
            np.isin(lab[s + 5 * FPS:s + 8 * FPS], ["H", "A"]).any())

    print("{:<28} poses {:>4}  mediana {:>5.1f}s  media {:>5.1f}s  D {:>5.1%}  "
          "acuerdo-fases {:>5.1%}  sin-balón {:>5.1%}  reinicios {:>5.1%}  cortas D-x-D {:>3}".format(
              os.path.basename(path), len(spells), np.median(spells), spells.mean(),
              (lab[graded] == "D").mean(), agree / total, agree_nb / total_nb, hits / len(pieces), short))


if __name__ == "__main__":
    tagging = load_tagging()
    for p in sys.argv[1:]:
        report(p, tagging)
