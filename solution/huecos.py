"""Prueba de huecos simulados: ¿qué tan bien adivina cada versión el equipo cuando el tracker
pierde el balón en juego vivo? Sin etiquetas y sobre el partido completo.

    python solution/huecos.py             # v1 y v2, con y sin fases
    python solution/huecos.py --barrido   # peso de las fases en la v2 (A1)

Por qué hace falta. `diagnose.py` mide el acuerdo con las fases del Smart Tagging, y las versiones
usan esas mismas fases como evidencia: si se les baja el peso, la métrica baja por construcción.
Para comparar pesos de fases (o versiones que las usan distinto) hace falta una referencia que no
dependa de ellas. Aquí la referencia es el propio control claro (v1.control), que el modelo no ve
porque se le esconde el balón.

Cómo funciona:
  1. Se miden los huecos reales del tracker en juego vivo (balón faltante con los jugadores
     corriendo) y de esa distribución se sacan los largos de los huecos simulados.
  2. En cada réplica se eligen al azar tramos de juego vivo con control claro, sin traslaparse, hasta
     cubrir MASK_FRACTION de los cuadros calificados, y ahí se pone el balón como faltante. Todo lo
     que depende del balón reacciona como en un hueco real, y el clasificador colectivo ya no puede
     entrenar con esos cuadros.
  3. Se corre cada versión completa (evidencia, clasificador, Viterbi) sobre los datos con huecos.
  4. En los cuadros que tenían control claro, sin contar EDGE segundos en cada borde del tramo, se
     mide qué fracción recibe el equipo correcto y qué fracción queda como D (muerto falso).

Además revisa si el Smart Tagging tenía información en los huecos reales: Metrica lo infiere del
mismo video y tracking, así que en los huecos simulados las fases sí "vieron" el balón. Si en los
huecos reales las fases cambian de equipo mucho menos que en juego con balón visible, entonces ahí
estaban a ciegas y esta prueba le da a las fases más crédito del que merecen.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import v1  # noqa: E402
import v2  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = v1.FPS
SEED = 2027
REPLICAS = 20
MASK_FRACTION = 0.12   # de los cuadros calificados: deja ~88% para entrenar el clasificador
EDGE = 2.0             # s: las features se suavizan a 2 s, así que los bordes ven datos sin hueco
MIN_GAP = 5.0          # s: con EDGE en cada lado, un hueco más corto no deja nada que medir
MIN_CONTROL = 1.0      # s de control claro dentro de la parte medida del tramo
SEPARATION = 2.0       # s mínimos entre dos tramos simulados

VERSIONS = {
    "v1 sin fases": lambda *d: v1.evidence(*d, phase_weight=0.0),
    "v1": lambda *d: v1.evidence(*d),
    "v2 sin fases": lambda *d: v2.evidence(*d, phase_weight=0.0),
    "v2": lambda *d: v2.evidence(*d),
}


# Barrido del peso de las fases (A1), fijado antes de correrlo. Puntos de referencia:
#   0.1    el peso de v1 y v2 (cada cuadro de una fase cuenta como observación independiente);
#   0.03   una observación por segundo: logit(68.6%) = 0.78 nats entre 25 cuadros, como el
#          clasificador colectivo;
#   0.008  una observación cada 3.7 s, el tramo medio de acuerdo/desacuerdo de las fases con el
#          control claro;
#   0.002  una observación por fase: logit(75.4%) = 1.12 nats entre los ~22 s de una fase mediana.
# Los demás llenan los huecos en escala aproximadamente logarítmica.
PHASE_GRID = (0.0, 0.002, 0.004, 0.008, 0.015, 0.03, 0.05, 0.1)
# Regla de elección: el peso más bajo cuya diferencia pareada contra el mejor no sea peor que un
# error estándar. Se prefiere el más bajo porque la prueba puede darle a las fases más crédito del
# que merecen (ver leak_check).


def runs(mask):
    """Inicios y finales (exclusivos) de cada tramo True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return edges[::2], edges[1::2]


def live_gaps(ball, crowd, graded):
    """Huecos reales en juego vivo: balón faltante, calificado y jugadores corriendo en promedio."""
    starts, ends = runs(np.isnan(ball[:, 0]) & graded)
    return [(s, e) for s, e in zip(starts, ends)
            if e - s >= MIN_GAP * FPS and crowd[s:e].mean() >= v1.WALK_SPEED]


def sample_blocks(rng, lengths, clear, crowd, graded):
    """Tramos simulados sin traslaparse, hasta cubrir MASK_FRACTION de lo calificado."""
    edge, sep = int(EDGE * FPS), int(SEPARATION * FPS)
    taken = np.zeros(len(clear), dtype=bool)
    target, covered, blocks = MASK_FRACTION * graded.sum(), 0, []
    candidates = np.flatnonzero(graded)
    while covered < target:
        length = int(rng.choice(lengths))
        s = int(rng.choice(candidates))
        e = s + length
        if not graded[s:e].all() or taken[max(s - sep, 0):e + sep].any():
            continue
        if crowd[s:e].mean() < v1.WALK_SPEED or clear[s + edge:e - edge].sum() < MIN_CONTROL * FPS:
            continue
        taken[s:e] = True
        covered += length
        blocks.append((s, e))
    return blocks


def phase_team(tagging, n):
    """Equipo con el balón según las fases del Smart Tagging, -1 fuera de ellas."""
    side_of = np.full(n, -1, dtype=np.int8)
    for e in tagging:
        side = v1.TEAM_OF.get((e["tags"].get("Team") or "").strip().lower())
        if side is None or e["code"] not in v1.ATTACK_CODES | v1.DEFEND_CODES:
            continue
        side_of[e["start_frame"]:e["end_frame"]] = 1 - side if e["code"] in v1.DEFEND_CODES else side
    return side_of


def leak_check(gaps, tagging, crowd, ball, graded):
    """Cambios de equipo de las fases por minuto: dentro de huecos reales vs con balón visible."""
    side = phase_team(tagging, len(ball))
    switch = np.r_[False, (side[1:] != side[:-1]) & (side[1:] >= 0) & (side[:-1] >= 0)]
    in_gap = np.zeros(len(ball), dtype=bool)
    for s, e in gaps:
        in_gap[s:e] = True
    visible = graded & ~np.isnan(ball[:, 0]) & (crowd >= v1.WALK_SPEED)
    for name, m in (("en huecos reales", in_gap), ("con balón visible", visible)):
        print("  fases {:<18} {:>5.2f} cambios de equipo por minuto ({:.0f} min)".format(
            name, switch[m].sum() / (m.sum() / FPS / 60), m.sum() / FPS / 60))


def dead_side(ball, players, team, pitch, tagging, graded, crowd):
    """Para cada peso del barrido, sin esconder nada: fracción en D de los cuadros donde el balón
    muerto es seguro (balón visto fuera, jugadores a < 1 m/s, 4 s antes de un SET PIECES) y número
    de posesiones. Revisa que más peso de fases no tape muertos reales."""
    bx, by = ball[:, 0], ball[:, 1]
    mx, my = v1.OUT_MARGIN
    out = graded & ~np.isnan(bx) & ((bx < -mx) | (bx > 1 + mx) | (by < -my) | (by > 1 + my))
    walking = graded & (crowd < 1.0)
    before = np.zeros(len(ball), dtype=bool)
    for e in tagging:
        if e["code"] == "SET PIECES":
            before[max(e["start_frame"] - 4 * FPS, 0):e["start_frame"]] = True
    before &= graded
    print("\n{:<8} {:>12} {:>12} {:>14} {:>11}".format(
        "peso", "D fuera", "D caminando", "D pre-reinicio", "posesiones"))
    for w in PHASE_GRID:
        lab = v1.label(v2.evidence(ball, players, team, pitch, tagging, phase_weight=w), tagging)
        spells = 0
        for _, first, last in v1.half_video_blocks(v1.GAME):
            x = lab[first:last + 1]
            starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
            spells += int(np.isin(x[starts], ["H", "A"]).sum())
        dead = lab == "D"
        print("{:<8g} {:>12.1%} {:>12.1%} {:>14.1%} {:>11d}".format(
            w, dead[out].mean(), dead[walking].mean(), dead[before].mean(), spells))


def main():
    ball, players, team, pitch = load()
    tagging = load_tagging()
    graded = v2.graded_mask(len(ball))
    crowd = np.nan_to_num(v1.crowd_speed(players), nan=0.0)
    clear = v1.control(ball, players, team, pitch)
    truth = np.where(clear[:, v1.H], 0, np.where(clear[:, v1.A], 1, -1))
    gaps = live_gaps(ball, crowd, graded)
    lengths = np.array([e - s for s, e in gaps])
    print("huecos reales en juego vivo >= {:.0f} s: {}, mediana {:.1f} s, p90 {:.1f} s".format(
        MIN_GAP, len(gaps), np.median(lengths) / FPS, np.percentile(lengths, 90) / FPS))
    leak_check(gaps, tagging, crowd, ball, graded)

    if "--barrido" in sys.argv:
        versions = {"v2 fases {:g}".format(w): (lambda w: lambda *d: v2.evidence(*d, phase_weight=w))(w)
                    for w in PHASE_GRID}
    else:
        versions = VERSIONS
    results = simulate(versions, ball, players, team, pitch, tagging, graded, crowd, clear,
                       truth, lengths)
    ref = "v2" if "v2" in results else max(results, key=lambda k: results[k][:, 0].mean())
    table(results, ref)
    if "--barrido" in sys.argv:
        best = results[ref][:, 0]
        for w, name in zip(PHASE_GRID, results):
            d = results[name][:, 0] - best
            if d.mean() >= -d.std() / np.sqrt(REPLICAS):
                print("\nelegido: peso {:g} (el más bajo dentro de un error estándar de {})".format(w, ref))
                break
        dead_side(ball, players, team, pitch, tagging, graded, crowd)


def simulate(versions, ball, players, team, pitch, tagging, graded, crowd, clear, truth, lengths):
    """(réplicas, 2) por versión: fracción de equipo correcto y de muerto falso."""
    rng = np.random.default_rng(SEED)
    results = {name: [] for name in versions}
    edge = int(EDGE * FPS)
    for _ in range(REPLICAS):
        blocks = sample_blocks(rng, lengths, clear.any(1), crowd, graded)
        hidden = ball.copy()
        scored = np.zeros(len(ball), dtype=bool)
        for s, e in blocks:
            hidden[s:e] = np.nan
            scored[s + edge:e - edge] = True
        scored &= truth >= 0
        for name, version in versions.items():
            # Cada versión devuelve puntajes (n, 3), que se etiquetan como en la v2, o etiquetas ya
            # hechas (n,), para las que eligen su propio saque o costo de cambio (cambios.py).
            out = version(hidden, players, team, pitch, tagging)
            labels = v1.label(out, tagging) if out.ndim == 2 else out
            got = labels[scored]
            results[name].append(((got == np.array(["H", "A"])[truth[scored]]).mean(),
                                  (got == "D").mean()))
    print("\n{} réplicas, ~{:.0%} de lo calificado escondido por réplica, semilla {}".format(
        REPLICAS, MASK_FRACTION, SEED))
    return {name: np.array(r) for name, r in results.items()}


def table(results, ref):
    # Todas las versiones ven los mismos huecos en cada réplica, así que la diferencia contra la
    # referencia se mide réplica por réplica: su dispersión es mucho menor que la de cada una sola.
    print("{:<16} {:>18} {:>18} {:>22}".format("versión", "equipo correcto", "muerto falso",
                                               "equipo vs " + ref + " (pareado)"))
    for name, r in results.items():
        d = r[:, 0] - results[ref][:, 0]
        print("{:<16} {:>10.1%} ± {:>4.1%} {:>10.1%} ± {:>4.1%} {:>+13.1%} ± {:>4.1%}".format(
            name, r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std(), d.mean(), d.std()))

if __name__ == "__main__":
    main()
