"""Lee el ATD crudo a arreglos numpy y lo cachea en data/cache/ (gitignored, nunca se publica).

    ball:    (n, 3)      x, y normalizados 0-1 y velocidad; NaN si no hay observación
    players: (n, 46, 3)  igual, por slot de track
    team:    (46,)       0 = local (Barça), 1 = visitante (Athletic)

Índice = video frame (frame_id - 1), el mismo que una fila de la entrega.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache", "atd_tactical.npz")


def parse(raw_path, n_frames, n_slots):
    out = np.full((n_frames, n_slots + 1, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            head, rest = line.split(":", 1)
            values = np.array(rest.replace(":", ";").replace(";", ",").split(","), dtype=np.float32)
            out[int(head) - 1] = values.reshape(n_slots + 1, 3)
    return out


def load(game=GAME):
    if not os.path.exists(CACHE):
        meta, raw = require(["atd_meta", "atd_raw"], game)
        team = team_of_each_track(meta)
        allv = parse(raw, source_frames("tactical", game), len(team))
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, players=allv[:, :-1], ball=allv[:, -1], team=team)
    z = np.load(CACHE)
    return z["ball"], z["players"], z["team"], GAMES[game]["pitch"]


if __name__ == "__main__":
    ball, players, team, pitch = load()
    assert ball.shape == (199319, 3) and players.shape == (199319, 46, 3)
    print("ball visible", np.mean(~np.isnan(ball[:, 0])), "home slots", int((team == 0).sum()))
