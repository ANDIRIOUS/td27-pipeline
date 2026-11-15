"""B4 (#10): sincroniza el tracking del broadcast con el táctico, sin tocar el video, y mide si su
balón rellena los huecos del táctico. Experimento: no se integra al modelo.

    docker run --rm -v "$PWD":/r -v /ruta/a/data:/r/data mtah-v1:1 python solution/broadcast_sync.py
    docker run --rm -v "$PWD":/r -v /ruta/a/data:/r/data mtah-v1:1 python solution/broadcast_sync.py --autotest

Escribe `data/cache/atd_broadcast.npz` (ATD del broadcast parseado) y `data/cache/broadcast_sync.npz`
(tabla de mapeo: cuadro táctico de cada cuadro del broadcast, o -1). Ambos están en .gitignore y nunca
se imprimen: el script solo reporta agregados.

Método
------
Los dos ATD están en coordenadas de campo normalizadas 0-1 y ambos XML dicen FrameRate 25.0, así que
dentro de un tramo continuo del broadcast el cuadro táctico es `b + d`, con un desfase `d` constante.

1. Tramos. Un cuadro del broadcast es usable si su homografía es válida y ve al menos MIN_PLAYERS
   jugadores. Se corta donde un cuadro no es usable o donde las posiciones saltan: la mediana, sobre
   los jugadores del cuadro, de la distancia al vecino más cercano del mismo equipo en el cuadro
   anterior supera MAX_JUMP. Se descartan tramos de menos de MIN_SEG cuadros.
2. Desfase. Medida sin identidades: para cada jugador del broadcast, distancia al jugador más cercano
   del mismo equipo en el táctico (cuadro b + d), recortada a CLIP; el balón cuenta como un objeto más
   cuando ambos lo ven. El costo es la media sobre N_SAMPLES cuadros del tramo. Búsqueda global sobre
   todos los desfases posibles a paso STRIDE y luego refinamiento a paso 1 alrededor del mínimo.
3. Aceptación ("mínimo nítido"): costo mínimo <= MAX_COST, y <= RATIO veces el mejor costo de
   cualquier desfase a más de EXCL cuadros del mínimo. En tramos de al menos DRIFT_MIN_SEG cuadros se
   busca el desfase por separado en el primer y el último tercio: si difieren más de DRIFT_MAX cuadros,
   el desfase no es constante (repetición en cámara lenta o frecuencias distintas) y se descarta.
4. Medición sobre los cuadros calificados del táctico (ver main()).

Constantes (fijadas a priori; ninguna se ajustó viendo resultados)
------------------------------------------------------------------
Distancias en metros sobre un campo de 105.3 x 68.0 m (`games.py`, medido del tracking corregido; el
XML dice 105 x 68, la diferencia es de 30 cm y no cambia nada aquí).

Resultado (12 nov 2026, ~3 min)
-------------------------------
El mapeo se sostiene (129 de 174 tramos aceptados, 55.7% de los cuadros del broadcast, error de
jugadores p50 0.30 m), pero el balón casi no aporta: los dos trackers pierden el balón en los mismos
momentos. Solo el 8.7% de los cuadros calificados sin balón táctico recibe balón del broadcast, y en
los huecos de más de 10 s, el 5.2%. Error del balón donde ambos lo ven: p50 0.78 m, p90 5.65 m.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import atd  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402

CACHE_B = os.path.join(ROOT, "data", "cache", "atd_broadcast.npz")
OUT = os.path.join(ROOT, "data", "cache", "broadcast_sync.npz")
FPS = 25

# Con menos de 6 puntos casi cualquier momento del partido empata la nube; además el histograma de
# jugadores visibles del broadcast tiene un valle entre 3 y 10 (tomas cerradas vs. plano general).
MIN_PLAYERS = 6
# Un jugador corre a lo sumo ~10 m/s = 0.4 m por cuadro; el ruido del tracking es de decímetros. Si la
# MEDIANA de los desplazamientos pasa de 1 m, se movió la toma entera: es un corte.
MAX_JUMP = 1.0
# 2 s: en ese tiempo un jugador al trote (3 m/s) se mueve 6 m, muy por encima del error de tracking,
# así que la configuración del tramo ya es característica de un momento.
MIN_SEG = 2 * FPS
# Paso grueso de 0.4 s: fallar por 5 cuadros (0.2 s) mueve a un jugador al trote ~0.6 m, poco frente
# a los 3-5 m de un desfase equivocado. El refinamiento a paso 1 recupera el cuadro exacto.
STRIDE = 10
N_SAMPLES = 8
# Más allá de 5 m ya no es el mismo jugador; recortar evita que un fantasma del tracker o un jugador
# que el táctico no ve domine el costo.
CLIP = 5.0
# Dos trackers automáticos independientes: ~0.5 m de error cada uno, más la homografía del broadcast
# en tomas con zoom. Un empate real queda en ~1 m de media; 2 m deja margen sin aceptar basura.
MAX_COST = 2.0
# Mínimo nítido: el desfase correcto debe tener a lo sumo la mitad del error del mejor desfase
# alternativo a más de 2 s (EXCL), donde todos ya se movieron varios metros.
RATIO = 0.5
EXCL = 2 * FPS
# Chequeo de deriva: tramos de 10 s o más; mismas frecuencias implican el mismo desfase en el primer y
# el último tercio, salvo +-1 cuadro de ruido del argmin.
DRIFT_MIN_SEG = 10 * FPS
DRIFT_MAX = 2
FAR = 1e4  # posición de relleno para slots vacíos: nunca gana un mínimo


def load_broadcast():
    if not os.path.exists(CACHE_B):
        meta, raw, hom = require(["atd_meta", "atd_raw", "homography"], source="broadcast")
        team = team_of_each_track(meta)
        allv = atd.parse(raw, source_frames("broadcast"), len(team))
        h = pd.read_csv(hom, usecols=["frame", "valid"]).sort_values("frame")
        assert np.array_equal(h["frame"].to_numpy(), np.arange(len(allv)))
        np.savez(CACHE_B, players=allv[:, :-1], ball=allv[:, -1], team=team,
                 valid=h["valid"].to_numpy().astype(bool))
    z = np.load(CACHE_B)
    return z["ball"], z["players"], z["team"], z["valid"]


def to_metres(players, ball, team, pitch):
    """(home, away, ball) en metros; NaN donde no hay observación."""
    s = np.array(pitch, dtype=np.float32)
    return players[:, team == 0, :2] * s, players[:, team == 1, :2] * s, ball[:, :2] * s


def tactical_arrays(home, away, ball):
    """El táctico con los slots vacíos lejos, para que min() los ignore sin nanmin."""
    return np.nan_to_num(home, nan=FAR), np.nan_to_num(away, nan=FAR), ball


def samples(home, away, ball, rows, labels):
    """Puntos visibles de cada cuadro muestreado: (cuadro broadcast, home, away, balón o None)."""
    out = []
    for r, b in zip(rows, labels):
        h, a = home[r], away[r]
        out.append((b, h[~np.isnan(h[:, 0])], a[~np.isnan(a[:, 0])],
                    None if np.isnan(ball[r, 0]) else ball[r]))
    return out


def frame_cost(bh, ba, bb, tac, idx):
    """Suma de distancias recortadas y número de objetos, para un cuadro del broadcast contra los
    cuadros tácticos idx."""
    th, ta, tb = tac
    num = np.zeros(len(idx), np.float64)
    cnt = np.zeros(len(idx), np.float64)
    for pts, t in ((bh, th), (ba, ta)):
        if len(pts):
            d = np.sqrt(((t[idx][:, None] - pts[None, :, None]) ** 2).sum(-1)).min(2)
            num += np.minimum(d, CLIP).sum(1)
            cnt += len(pts)
    if bb is not None:
        d = np.sqrt(((tb[idx] - bb) ** 2).sum(-1))
        ok = ~np.isnan(d)
        num[ok] += np.minimum(d[ok], CLIP)
        cnt[ok] += 1
    return num, cnt


def cost_offsets(smp, tac, offsets):
    n_tac = len(tac[0])
    num = np.zeros(len(offsets))
    cnt = np.zeros(len(offsets))
    bad = np.zeros(len(offsets), bool)
    for b, bh, ba, bb in smp:
        idx = b + offsets
        ok = (idx >= 0) & (idx < n_tac)
        n_, c_ = frame_cost(bh, ba, bb, tac, idx[ok])
        num[ok] += n_
        cnt[ok] += c_
        bad |= ~ok
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = num / cnt
    cost[bad | (cnt == 0)] = np.inf
    return cost


def refine(smp, tac, center, half, forbid=None):
    d = np.arange(center - half, center + half + 1)
    if forbid is not None:
        d = d[np.abs(d - forbid) > EXCL]
    if not len(d):
        return center, np.inf
    c = cost_offsets(smp, tac, d)
    i = int(np.argmin(c))
    return int(d[i]), float(c[i])


def search(smp, tac):
    """(desfase, costo mínimo, costo del mejor desfase a más de EXCL cuadros)."""
    n_tac = len(tac[0])
    bs = [s[0] for s in smp]
    coarse_d = np.arange(-min(bs), n_tac - max(bs), STRIDE)
    c = cost_offsets(smp, tac, coarse_d)
    if not np.isfinite(c).any():
        return 0, np.inf, np.inf
    d1, c1 = refine(smp, tac, int(coarse_d[np.argmin(c)]), STRIDE)
    far = np.abs(coarse_d - d1) > EXCL
    if not far.any():
        return d1, c1, np.inf
    _, c2 = refine(smp, tac, int(coarse_d[far][np.argmin(c[far])]), STRIDE, forbid=d1)
    return d1, c1, min(c2, float(c[far].min()))


def judge(home, away, ball, b0, b1, tac, labels_of=lambda r: r):
    """Busca y juzga un tramo [b0, b1]. Devuelve (desfase, c1, c2, deriva, aceptado)."""
    rows = np.unique(np.linspace(b0, b1, N_SAMPLES).round().astype(int))
    smp = samples(home, away, ball, rows, [labels_of(r) for r in rows])
    d, c1, c2 = search(smp, tac)
    drift = 0
    if b1 - b0 + 1 >= DRIFT_MIN_SEG and np.isfinite(c1):
        third = (b1 - b0) // 3
        ends = []
        for lo, hi in ((b0, b0 + third), (b1 - third, b1)):
            r = np.unique(np.linspace(lo, hi, N_SAMPLES).round().astype(int))
            ends.append(refine(samples(home, away, ball, r, [labels_of(x) for x in r]), tac, d,
                               2 * STRIDE)[0])
        drift = ends[1] - ends[0]
    ok = c1 <= MAX_COST and c1 <= RATIO * c2 and abs(drift) <= DRIFT_MAX
    return d, c1, c2, drift, bool(ok)


def segments(home, away, valid):
    """Tramos continuos [(b0, b1)] del broadcast."""
    nvis = (~np.isnan(home[..., 0])).sum(1) + (~np.isnan(away[..., 0])).sum(1)
    usable = valid & (nvis >= MIN_PLAYERS)
    jump = np.full(len(home), np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for lo in range(1, len(home), 20000):
            hi = min(lo + 20000, len(home))
            parts = []
            for p in (home, away):
                d = np.sqrt(((p[lo:hi, :, None] - p[lo - 1:hi - 1, None, :]) ** 2).sum(-1))
                parts.append(np.nanmin(d, axis=2))
            jump[lo:hi] = np.nanmedian(np.concatenate(parts, axis=1), axis=1)
    cont = usable & np.r_[False, usable[:-1]] & (jump <= MAX_JUMP)  # sigue el tramo del anterior
    starts = np.flatnonzero(usable & ~cont)
    out = []
    for s in starts:
        e = s
        while e + 1 < len(home) and cont[e + 1]:
            e += 1
        out.append((int(s), int(e)))
    return out, usable


def nn_errors(home, away, tac, b0, b1, d, step=5):
    """Distancia sin recortar de cada jugador del broadcast a su vecino del mismo equipo (metros)."""
    th, ta, _ = tac
    out = []
    for b in range(b0, b1 + 1, step):
        for p, t in ((home[b], th[b + d]), (away[b], ta[b + d])):
            p = p[~np.isnan(p[:, 0])]
            if len(p):
                out.append(np.sqrt(((p[:, None] - t[None]) ** 2).sum(-1)).min(1))
    return np.concatenate(out) if out else np.empty(0)


def q(x, ps=(50, 90)):
    return " ".join("p{}={:.2f}".format(p, np.percentile(x, p)) for p in ps) if len(x) else "n/a"


def autotest(tac_h, tac_a, tac_b, tac, n_trials=10):
    """Recorta trozos del táctico como si fueran tramos del broadcast (60% de los slots, ruido de
    0.3 m, desfase conocido) y verifica que se recupera el desfase; la copia espejada en x debe
    rechazarse."""
    rng = np.random.default_rng(10)
    pitch_len = 105.3
    blocks = half_video_blocks(GAME)
    hits = rejected_flips = 0
    for trial in range(n_trials):
        _, first, last = blocks[trial % 2]
        L = int(rng.integers(MIN_SEG, 15 * FPS))
        t0 = int(rng.integers(first, last - L))
        d_true = int(rng.integers(-40000, 40000))
        keep_h = rng.random(tac_h.shape[1]) < 0.6
        keep_a = rng.random(tac_a.shape[1]) < 0.6
        noise = lambda x: x + rng.normal(0, 0.3, x.shape).astype(np.float32)  # noqa: E731
        h = noise(tac_h[t0:t0 + L][:, keep_h])
        a = noise(tac_a[t0:t0 + L][:, keep_a])
        bl = noise(tac_b[t0:t0 + L])
        lab = lambda r: t0 - d_true + r  # noqa: E731
        d, c1, c2, drift, ok = judge(h, a, bl, 0, L - 1, tac, lab)
        hits += ok and abs(d - d_true) <= 1
        fl = lambda x: np.stack([pitch_len - x[..., 0], x[..., 1]], -1)  # noqa: E731
        _, _, _, _, ok_f = judge(fl(h), fl(a), fl(bl), 0, L - 1, tac, lab)
        rejected_flips += not ok_f
    print("autotest: desfase recuperado (+-1 cuadro) y aceptado en {}/{}; copia espejada rechazada "
          "en {}/{}".format(hits, n_trials, rejected_flips, n_trials))
    return hits == n_trials and rejected_flips == n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotest", action="store_true")
    args = ap.parse_args()
    t_start = time.time()

    ball_t, players_t, team_t, pitch = atd.load()
    tac_h, tac_a, tac_b = to_metres(players_t, ball_t, team_t, pitch)
    tac = tactical_arrays(tac_h, tac_a, tac_b)
    if args.autotest:
        ok = autotest(tac_h, tac_a, tac_b, tac)
        print("tiempo {:.0f} s".format(time.time() - t_start))
        sys.exit(0 if ok else 1)

    ball_b, players_b, team_b, valid = load_broadcast()
    bh, ba, bb = to_metres(players_b, ball_b, team_b, pitch)
    n_b, n_t = len(bh), len(tac_h)
    segs, usable = segments(bh, ba, valid)
    long_segs = [s for s in segs if s[1] - s[0] + 1 >= MIN_SEG]
    print("broadcast: {} cuadros, {:.1%} usables; {} tramos, {} de al menos {} cuadros "
          "({:.1%} de los cuadros)".format(n_b, usable.mean(), len(segs), len(long_segs), MIN_SEG,
                                          sum(e - s + 1 for s, e in long_segs) / n_b))

    # Chequeo de sistema de coordenadas en los 10 tramos más largos: la identidad debe ganar.
    top = sorted(long_segs, key=lambda s: s[0] - s[1])[:10]
    L, W = pitch
    transforms = {
        "identidad": lambda h, a, b: (h, a, b),
        "espejo x": lambda h, a, b: tuple(np.stack([L - x[..., 0], x[..., 1]], -1) for x in (h, a, b)),
        "espejo y": lambda h, a, b: tuple(np.stack([x[..., 0], W - x[..., 1]], -1) for x in (h, a, b)),
        "espejo xy": lambda h, a, b: tuple(np.stack([L - x[..., 0], W - x[..., 1]], -1) for x in (h, a, b)),
        "equipos cruzados": lambda h, a, b: (a, h, b),
    }
    best = {k: [] for k in transforms}
    for s, e in top:
        sl = slice(s, e + 1)
        for k, f in transforms.items():
            h, a, b_ = f(bh[sl], ba[sl], bb[sl])
            rows = np.unique(np.linspace(0, e - s, N_SAMPLES).round().astype(int))
            best[k].append(search(samples(h, a, b_, rows, rows + s), tac)[1])
    c = np.array([best[k] for k in transforms])
    print("simetrías (10 tramos más largos), costo mínimo mediano en m: " + ", ".join(
        "{} {:.2f}".format(k, np.median(best[k])) for k in transforms)
        + "; la identidad gana en {}/10".format(int((c.argmin(0) == 0).sum())))

    # Búsqueda por tramo.
    tac_of_b = np.full(n_b, -1, np.int32)
    res = []
    for i, (s, e) in enumerate(long_segs):
        d, c1, c2, drift, ok = judge(bh, ba, bb, s, e, tac)
        res.append((s, e, d, c1, c2, drift, ok))
        if ok:
            tac_of_b[s:e + 1] = np.arange(s, e + 1) + d
    r = np.array(res, dtype=float)
    acc = r[:, 6].astype(bool)
    np.savez(OUT, tac_of_b=tac_of_b, segments=r)
    reasons = {
        "costo > MAX_COST": r[:, 3] > MAX_COST,
        "mínimo no nítido": (r[:, 3] <= MAX_COST) & (r[:, 3] > RATIO * r[:, 4]),
        "deriva": (r[:, 3] <= MAX_COST) & (r[:, 3] <= RATIO * r[:, 4]) & (np.abs(r[:, 5]) > DRIFT_MAX),
    }
    lens = r[:, 1] - r[:, 0] + 1
    print("tramos aceptados {} / descartados {} ({}); duración mediana aceptados {:.1f} s".format(
        acc.sum(), (~acc).sum(), ", ".join("{} {}".format(k, int(v.sum())) for k, v in reasons.items()),
        np.median(lens[acc]) / FPS if acc.any() else 0))
    print("costo mínimo (m) aceptados: {}; descartados: {}".format(q(r[acc, 3]), q(r[~acc & np.isfinite(r[:, 3]), 3])))
    print("razón c1/c2 aceptados: {}".format(q(r[acc, 3] / r[acc, 4])))
    dl = r[(lens >= DRIFT_MIN_SEG) & np.isfinite(r[:, 3]) & (r[:, 3] <= MAX_COST), 5]
    print("deriva primer vs último tercio (tramos >= 10 s con costo <= MAX_COST, n={}): "
          "|deriva|=0 {:.0%}, <=2 {:.0%}, >2 {:.0%}".format(
              len(dl), np.mean(dl == 0), np.mean(np.abs(dl) <= 2), np.mean(np.abs(dl) > 2)))
    order = r[acc][np.argsort(r[acc, 0])]
    chrono = np.mean(order[1:, 0] + order[1:, 2] >= order[:-1, 1] + order[:-1, 2] - FPS) if len(order) > 1 else 0
    print("tramos aceptados en orden cronológico respecto al anterior: {:.0%} "
          "(el resto son repeticiones a velocidad real)".format(chrono))

    # 1. Cuadros del broadcast mapeados.
    mapped = tac_of_b >= 0
    print("\n[1] cuadros del broadcast mapeados: {:.1%} ({} de {})".format(mapped.mean(), mapped.sum(), n_b))

    # 2. Relleno del balón en cuadros tácticos calificados sin balón.
    graded = np.zeros(n_t, bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    b_ball = mapped & ~np.isnan(bb[:, 0])
    filled = np.zeros(n_t, bool)
    filled[tac_of_b[b_ball]] = True
    covered = np.zeros(n_t, bool)
    covered[tac_of_b[mapped]] = True
    missing = graded & np.isnan(tac_b[:, 0])
    print("[2] calificados con algún cuadro broadcast mapeado: {:.1%}; calificados sin balón: {} "
          "({:.1%}); de ésos reciben balón del broadcast: {} ({:.1%})".format(
              covered[graded].mean(), missing.sum(), missing.sum() / graded.sum(),
              (missing & filled).sum(), (missing & filled).sum() / missing.sum()))
    # ¿Ven el balón en momentos distintos? Si los huecos fueran independientes, el broadcast lo vería
    # igual de seguido con o sin balón táctico.
    print("    con cuadro broadcast mapeado, el broadcast ve el balón en {:.1%} de los calificados sin "
          "balón táctico y en {:.1%} de los que sí lo tienen".format(
              filled[missing & covered].mean(), filled[graded & covered & ~missing].mean()))
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(np.int8), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    run_len = np.array([e - s for s, e in runs])
    run_fill = np.array([filled[s:e].mean() for s, e in runs])
    for name, sel in (("< 2 s", run_len < 2 * FPS), ("2-10 s", (run_len >= 2 * FPS) & (run_len <= 10 * FPS)),
                      ("> 10 s", run_len > 10 * FPS)):
        fr = run_len[sel].sum()
        print("    huecos {:7s}: {:5d} huecos, {:6d} cuadros, rellenos {:.1%}".format(
            name, sel.sum(), fr, (run_len[sel] * run_fill[sel]).sum() / fr if fr else 0))
    lg = run_len > 10 * FPS
    print("    huecos > 10 s cubiertos al menos a la mitad: {} de {}".format(int((run_fill[lg] >= 0.5).sum()), lg.sum()))

    # 3. Error de posición del balón donde ambos lo ven.
    bidx = np.flatnonzero(b_ball)
    tidx = tac_of_b[bidx]
    both = graded[tidx] & ~np.isnan(tac_b[tidx, 0])
    err = np.sqrt(((bb[bidx[both]] - tac_b[tidx[both]]) ** 2).sum(1))
    print("[3] error del balón (m, campo 105.3 x 68.0 de games.py) en {} cuadros con ambos: {}; "
          "> 5 m: {:.1%}".format(len(err), q(err), np.mean(err > 5) if len(err) else 0))

    # Control de calidad: error de jugadores en tramos aceptados.
    pe = np.concatenate([nn_errors(bh, ba, tac, int(s), int(e), int(d)) for s, e, d, *_ in r[acc]]) if acc.any() else np.empty(0)
    print("QC jugadores (m, vecino más cercano del mismo equipo, tramos aceptados): {}".format(q(pe, (50, 90))))
    print("tiempo {:.0f} s".format(time.time() - t_start))


if __name__ == "__main__":
    main()
