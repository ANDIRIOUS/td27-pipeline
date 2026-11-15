"""v2: v1 + equipo inferido de la forma de los bloques cuando nadie controla el balón.

    python solution/v2.py [--out evaluation/results/v2.csv]

Tres cambios sobre v1, todos sin etiquetas:

1. Clasificador colectivo. En el 41% de los cuadros sin balón, v1 solo podía mantener al último
   equipo. Aquí una regresión logística aprende, con pseudo-etiquetas, qué forma tienen los dos
   bloques cuando cada equipo tiene el balón: quién está más abierto, hacia dónde se desplaza el
   conjunto, dónde está la línea más retrasada de cada uno. Las pseudo-etiquetas son los cuadros
   con control claro (v1.control) y jugadores en movimiento; ninguna persona decide ninguna. Luego
   se aplica solo donde no hay control claro, que son justo los cuadros con los que no se entrenó.
2. Reinicio anclado: el "vivo" después de un SET PIECES empieza en el primer movimiento real, no a
   un desfase fijo (v1.evidence(anchored=True)).
3. El "pendiente" (salió y no ha vuelto a jugarse) también se apaga cuando los jugadores pasan de
   caminar a correr, aunque el tracker no vea el balón.
"""
import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import v1  # noqa: E402
from atd import load  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

# --- CONSTANTES (a priori) ---------------------------------------------------------------------
FEATURE_SMOOTH = 2.0     # s: la forma de un bloque cambia en segundos, no en cuadros
MAX_STEP_SPEED = 12.0    # m/s: un salto mayor entre cuadros es un cambio de identidad del slot
L2 = 1.0                 # regularización ridge de la logística (features estandarizadas)
NEWTON_STEPS = 25
# Con features suavizadas a 2 s hay ~1 observación independiente por segundo, así que el log-odds
# del clasificador se reparte entre los 25 cuadros de ese segundo.
COLLECTIVE_WEIGHT = 1.0 / v1.FPS
# En validación cruzada por bloques las probabilidades extremas se cumplen ~85%, no 99%:
# un log-odds mayor que ~2 sería exceso de confianza.
LOGIT_CLIP = 2.0


def collective_features(players, team, pitch):
    """(n, f) features de la forma de los bloques, sin usar el balón. Ejes en metros, con +x
    hacia donde ataca el local (que cambia de lado en el descanso)."""
    length, width = pitch
    n = len(players)
    sign = np.ones(n)
    (_, _, last_first), _ = half_video_blocks(GAME)
    sign[last_first + 1:] = -1
    px = (players[:, :, 0] - 0.5) * sign[:, None] * length
    py = (players[:, :, 1] - 0.5) * width
    vx = np.diff(px, axis=0, prepend=np.nan) * v1.FPS
    vx[np.abs(vx) > MAX_STEP_SPEED] = np.nan

    cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # slots vacíos en todo el cuadro
        for k in (v1.H, v1.A):
            m = team == k
            cols += [np.nanmean(px[:, m], 1), np.nanstd(px[:, m], 1), np.nanstd(py[:, m], 1),
                     np.nanmean(vx[:, m], 1), np.nanmean(players[:, m, 2], 1),
                     np.nanmin(px[:, m], 1), np.nanmax(px[:, m], 1)]
    X = np.column_stack([v1.rolling(c, FEATURE_SMOOTH, "mean") for c in cols])
    # Contrastes entre bloques: quién está más abierto y hacia dónde se mueve el conjunto.
    X = np.column_stack([X, X[:, 1] - X[:, 8], X[:, 2] - X[:, 9], X[:, 3] + X[:, 10],
                         X[:, 0] + X[:, 7]])
    return np.where(np.isnan(X), np.nanmean(X, 0), X)


def fit_logistic(Z, y):
    """Logística ridge por Newton-Raphson: determinista, sin semillas ni dependencias."""
    Zb = np.column_stack([Z, np.ones(len(Z))])
    ridge = L2 * np.r_[np.ones(Z.shape[1]), 0.0]
    w = np.zeros(Zb.shape[1])
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-Zb @ w))
        grad = Zb.T @ (p - y) + ridge * w
        hess = (Zb * (p * (1 - p))[:, None]).T @ Zb + np.diag(ridge)
        w -= np.linalg.solve(hess, grad)
    return w


def collective_logit(ball, players, team, pitch, graded):
    """Log-odds por cuadro de que A tenga el balón según la forma de los bloques, y la máscara de
    cuadros con control claro (las pseudo-etiquetas)."""
    clear = v1.control(ball, players, team, pitch)
    moving = np.nan_to_num(v1.crowd_speed(players), nan=0.0) > v1.WALK_SPEED
    train = graded & clear.any(1) & moving
    X = collective_features(players, team, pitch)
    mu, sd = X[graded].mean(0), X[graded].std(0)
    Z = (X - mu) / sd
    w = fit_logistic(Z[train], clear[train, v1.A].astype(float))
    return np.clip(np.column_stack([Z, np.ones(len(Z))]) @ w, -LOGIT_CLIP, LOGIT_CLIP), clear


def graded_mask(n):
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    return graded


def evidence(ball, players, team, pitch, tagging, phase_weight=v1.PHASE_EVIDENCE, sustain=0.0):
    """(n, 3) puntajes de la v2: v1 anclada + clasificador colectivo donde no hay control claro."""
    score = v1.evidence(ball, players, team, pitch, tagging, anchored=True, phase_weight=phase_weight,
                        sustain=sustain)
    logit, clear = collective_logit(ball, players, team, pitch, graded_mask(len(ball)))
    loose = ~clear.any(1)                       # solo donde v1 no tenía evidencia de equipo
    score[loose, v1.A] += 0.5 * COLLECTIVE_WEIGHT * logit[loose]
    score[loose, v1.H] -= 0.5 * COLLECTIVE_WEIGHT * logit[loose]
    return score


def predict():
    ball, players, team, pitch = load()
    tagging = load_tagging()
    return v1.label(evidence(ball, players, team, pitch, tagging), tagging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v2.csv"))
    args = parser.parse_args()
    labels = predict()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT),
          {k: int((labels == k).sum()) for k in "HADX"})
