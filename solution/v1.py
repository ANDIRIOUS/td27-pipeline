"""v1: evidencia por cuadro + Viterbi de tres estados (H, A, D).

    python solution/v1.py [--out evaluation/results/v1.csv]

Idea en una línea: cada cuadro aporta log-odds a favor de "balón muerto" y a favor de cada equipo;
un Viterbi con costos de cambio asimétricos elige la secuencia de estados que mejor explica toda la
mitad. Los costos de cambio son la histéresis: cambiar de equipo exige control sostenido, pasar a
muerto exige varios segundos de señal muerta, y salir de muerto es casi gratis porque todo
reinicio tiene un poseedor claro.

Nada aquí se entrenó ni se ajustó contra etiquetas: todas las constantes están fijadas a priori con
una justificación física (ver CONSTANTES). La ventana pública solo se usa para verificar.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
from atd import load  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2

# --- CONSTANTES (a priori, no ajustadas contra la ventana pública) ---------------------------
CONTROL_RADIUS = 3.0     # m: a más de esto del balón nadie lo controla (balón en el aire, rebote)
CONTROL_MARGIN = 1.0     # m: el rival más cercano debe estar al menos esto más lejos
TEAM_EVIDENCE = 1.0      # nats por cuadro de control claro
WALK_SPEED = 1.3         # m/s: mediana colectiva por debajo de esto ~ jugadores caminando
SLOW_GAIN = 3.0          # nats por m/s de diferencia con WALK_SPEED, recortado a +-3
OUT_MARGIN = (0.005, 0.01)  # fracción del campo (~0.5 m y ~0.7 m) para decir "fuera" sin ruido
OUT_EVIDENCE = 4.0       # balón visto fuera del campo: muerto casi seguro
PENDING_EVIDENCE = 1.5   # salió y todavía no lo vimos rodar dentro: sigue muerto
STILL_SPEED = 0.7        # m/s: balón "quieto" (colocado para un reinicio)
STILL_EVIDENCE = 1.0
LIVE_SPEED = 3.0         # m/s: balón rodando dentro del campo = vivo (cierra el "pendiente")
FAST_SPEED = 8.0         # m/s: balón golpeado dentro del campo
FAST_EVIDENCE = -0.5
MISSING_EVIDENCE = 0.3   # sin balón: leve indicio de muerto (el tracker lo pierde en reinicios)
RESTART_BEFORE = (-4.0, 3.0)  # s respecto al start de un SET PIECE: muerto
RESTART_AFTER = (4.0, 7.0)    # s: ya se reanudó
RESTART_EVIDENCE = 1.0
RESTART_SEARCH = 20      # s (v2): un reinicio con barrera o cambio puede tardar, no más que esto
PHASE_EVIDENCE = 0.1     # nats por cuadro dentro de una fase del Smart Tagging con equipo
# Costos de cambio (nats). Con ~1 nat/cuadro de evidencia, 20 nats ~ 0.8 s de control claro.
SWITCH = np.array([
    #  a H   a A   a D
    [0.0, 20.0, 20.0],   # desde H
    [20.0, 0.0, 20.0],   # desde A
    [2.0, 2.0, 0.0],     # desde D: el reinicio siempre tiene poseedor
])

# Fases del Smart Tagging cuyo Team es el que ataca vs el que defiende (comprobado en el XML:
# PROGRESSION Barcelona coincide con DEFENDING IN MIDDLE THIRD Atletic Club en el mismo intervalo).
ATTACK_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "BALL IN THE BOX",
                "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL", "PLAYERS IN THE BOX"}
DEFEND_CODES = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}
TEAM_OF = {"barcelona": H, "atletic club": A}


def rolling(x, seconds, how="median"):
    """Ventana centrada; NaN si la ventana no tiene ningún dato."""
    r = pd.Series(x).rolling(int(seconds * FPS) | 1, center=True, min_periods=1)
    return getattr(r, how)().to_numpy()


def control(ball, players, team, pitch):
    """(n, 2) bool: control claro de H / de A. El más cercano a < CONTROL_RADIUS y el rival
    más cercano al menos CONTROL_MARGIN más lejos."""
    length, width = pitch
    bx, by = ball[:, 0], ball[:, 1]
    dist = np.hypot((players[:, :, 0] - bx[:, None]) * length, (players[:, :, 1] - by[:, None]) * width)
    dist = np.where(np.isnan(dist), np.inf, dist)
    near = np.stack([dist[:, team == k].min(1) for k in (H, A)], 1)
    return np.stack([(near[:, k] < CONTROL_RADIUS) & (near[:, 1 - k] > near[:, k] + CONTROL_MARGIN)
                     for k in (H, A)], 1)


def crowd_speed(players):
    """Mediana de velocidad de los jugadores en cada cuadro, suavizada 2 s (m/s)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return rolling(np.nanmedian(players[:, :, 2], 1), 2.0, "mean")


def motion(ball, players, pitch, anchored=False):
    """Señales de movimiento por cuadro: velocidad del grupo (m/s), balón fuera, balón dentro,
    velocidad del balón (m/s) y "vivo" (balón rodando dentro; con anchored, también los jugadores
    pasando de caminar a correr, aunque el tracker no vea el balón)."""
    length, width = pitch
    bx, by = ball[:, 0], ball[:, 1]
    seen = ~np.isnan(bx)
    crowd = crowd_speed(players)
    mx, my = OUT_MARGIN
    out = seen & ((bx < -mx) | (bx > 1 + mx) | (by < -my) | (by > 1 + my))
    speed = np.hypot(np.diff(bx * length, prepend=np.nan), np.diff(by * width, prepend=np.nan)) * FPS
    speed = rolling(speed, 1.0)
    inside = seen & ~out
    live = inside & (speed > LIVE_SPEED)
    if anchored:
        walk = np.nan_to_num(crowd, nan=0.0) >= WALK_SPEED
        live = live | (walk & ~np.r_[True, walk[:-1]])
    return crowd, out, inside, speed, live


def restart_frames(tagging, crowd, live, sustain=0.0):
    """{start de cada SET PIECES: (cuadro del reinicio, visto)}, con el reinicio anclado al primer
    movimiento dentro de RESTART_SEARCH s.

    sustain=0 (v2, v3): cualquier movimiento basta; si no se ve ninguno, start + RESTART_BEFORE[1].
    sustain>0 (v4): el movimiento cuenta solo si los jugadores siguen corriendo `sustain` s, y si no
    se ve ninguno se usa la mediana de los desfases vistos en el mismo partido (calculada aquí, no
    fijada a mano)."""
    ok = live
    if sustain:
        walk = np.nan_to_num(crowd, nan=0.0) >= WALK_SPEED
        k = int(sustain * FPS)
        c = np.cumsum(np.r_[0, walk])
        i = np.arange(len(walk))
        j = np.minimum(i + k, len(walk))
        ok = live & (c[j] - c[i] == j - i)
    found = {}
    for event in tagging:
        if (event["code"] or "").strip().upper() == "SET PIECES":
            start = event["start_frame"]
            moved = np.flatnonzero(ok[start:start + RESTART_SEARCH * FPS])
            found[start] = start + int(moved[0]) if len(moved) else None
    seen = [f - s for s, f in found.items() if f is not None]
    delay = int(np.median(seen)) if sustain and seen else int(RESTART_BEFORE[1] * FPS)
    return {s: (f, True) if f is not None else (s + delay, False) for s, f in found.items()}


def evidence(ball, players, team, pitch, tagging, anchored=False, phase_weight=PHASE_EVIDENCE,
             sustain=0.0):
    """(n, 3) puntajes por cuadro en nats: columnas H, A, D. Solo importan las diferencias.

    anchored=True (v2): el "vivo" tras un reinicio se ancla al primer movimiento real en vez de a
    un desfase fijo, y el "pendiente" también se apaga cuando los jugadores vuelven a correr.
    phase_weight: nats por cuadro de las fases con equipo (v3 usa 0.05).
    sustain: segundos de movimiento sostenido para dar por reanudado un SET PIECES (v4).
    """
    n = len(ball)
    score = np.zeros((n, 3))
    seen = ~np.isnan(ball[:, 0])

    # 1. Control claro del balón.
    score[:, :2] += TEAM_EVIDENCE * control(ball, players, team, pitch)

    # 2. Muerto por dinámica colectiva: mediana de velocidad de los jugadores, suavizada 2 s.
    crowd, out, inside, speed, live = motion(ball, players, pitch, anchored)
    dead = np.clip(SLOW_GAIN * (WALK_SPEED - np.nan_to_num(crowd, nan=0.0)), -3, 3)

    # 3. Muerto por geometría y por el balón mismo.
    # "Pendiente": la última vez que vimos el balón fuera es posterior a la última vez que lo
    # vimos rodar dentro, o sea, salió y no ha vuelto a jugarse.
    last_out = np.maximum.accumulate(np.where(out, np.arange(n), -1))
    last_live = np.maximum.accumulate(np.where(live, np.arange(n), -1))
    pending = (last_out >= 0) & (last_out > last_live) & ~out
    dead += OUT_EVIDENCE * out + PENDING_EVIDENCE * pending
    dead += STILL_EVIDENCE * (inside & (speed < STILL_SPEED))
    dead += FAST_EVIDENCE * (inside & (speed > FAST_SPEED))
    dead += MISSING_EVIDENCE * ~seen

    # 4. Smart Tagging: reinicios (muerto antes, vivo después) y fases con equipo (leve).
    restarts = restart_frames(tagging, crowd, live, sustain) if anchored else {}
    for event in tagging:
        code = (event["code"] or "").strip().upper()
        start = event["start_frame"]
        if code == "SET PIECES":
            before = start + int(RESTART_BEFORE[0] * FPS)
            restart = start + int(RESTART_BEFORE[1] * FPS)
            a, b = (start + int(s * FPS) for s in RESTART_AFTER)
            if anchored:
                # El reinicio es el primer movimiento real tras el start (restart_frames). Si no se
                # ve ninguno, solo queda el muerto previo.
                restart, moved = restarts[start]
                a, b = (restart, restart + b - a) if moved else (0, 0)
            dead[max(before, 0):restart] += RESTART_EVIDENCE
            dead[a:b] -= RESTART_EVIDENCE
            continue
        side = TEAM_OF.get((event["tags"].get("Team") or "").strip().lower())
        if side is None:
            continue
        if code in DEFEND_CODES:
            side = 1 - side
        elif code not in ATTACK_CODES:
            continue
        score[start:event["end_frame"], side] += phase_weight

    score[:, D] += dead
    return score


def viterbi(score, switch=SWITCH):
    """Camino de máximo puntaje: sum(score[t, s_t]) - sum(switch[s_{t-1}, s_t])."""
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    total = score[0].copy()
    for t in range(1, n):
        cand = total[:, None] - switch          # cand[desde, a]
        back[t] = cand.argmax(0)
        total = cand.max(0) + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = total.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict():
    ball, players, team, pitch = load()
    tagging = load_tagging()
    return label(evidence(ball, players, team, pitch, tagging), tagging)


def label(score, tagging, restarts=None, switch=SWITCH):
    """Viterbi por mitad, con D explícito antes de cada saque inicial y X fuera de lo calificado.

    restarts (v4, v5): {start del saque: (cuadro, visto)}; el saque inicial se toma de ahí en vez de
    sumarle un desfase fijo al Smart Tagging. switch: costos de cambio del Viterbi (cambios.py)."""
    labels = np.full(len(score), "X", dtype="<U1")
    kickoffs = sorted(e["start_frame"] for e in tagging
                      if (e["tags"].get("Type") or "").strip().lower() == "kick off (start)")
    for (_, first, last), kickoff in zip(half_video_blocks(GAME), kickoffs):
        # Antes del saque inicial de cada mitad no hay juego: D explícito (si dejáramos X, el
        # scorer rellenaría hacia atrás con el equipo que saca y la primera posesión empezaría
        # minutos antes). El saque se toma del Smart Tagging + el desfase del reinicio, o del
        # reinicio anclado si viene en restarts.
        restart = restarts[kickoff][0] if restarts else kickoff + int(RESTART_BEFORE[1] * FPS)
        restart = max(first, restart)
        labels[first:restart] = "D"
        path = viterbi(score[restart:last + 1], switch)
        labels[restart:last + 1] = np.array(["H", "A", "D"])[path]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "v1.csv"))
    args = parser.parse_args()
    labels = predict()
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT),
          {k: int((labels == k).sum()) for k in "HADX"})
