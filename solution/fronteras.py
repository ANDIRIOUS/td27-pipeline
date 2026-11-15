"""Auditoría de fronteras: ¿dónde corta el modelo cada cambio de posesión? Sin etiquetas y sobre el
partido completo. Solo mide; no arregla nada.

    python solution/fronteras.py

Por qué hace falta. El ranking usa spellF1, que empareja posesiones por IoU temporal. Según
EVALUATION.md, una entrega perfecta pero 8 cuadros tarde baja a 0.8933 y 1 s tarde a 0.6539. El reto
dice que con el balón disputado o suelto la posesión se queda con el último equipo que la tuvo, y
que cambia cuando el rival gana posesión controlada. En el Viterbi, durante un tramo suelto nadie
suma evidencia de control, así que el corte entre equipos cae donde lo empujen señales débiles
(fases, clasificador colectivo) o un desempate arbitrario. Nunca hemos medido dónde cae.

Referencia (un indicio nuestro, la lectura más corta de "conservar el balón"; NO es la verdad del
reto). Mismo indicio de control que cambios.py: un episodio es un tramo de al menos EPISODE s
seguidos de control claro (v1.control).
  Parte A, relevos H<->A. Cada par de episodios consecutivos de equipos distintos (viejo, nuevo) es
  un relevo. Su frontera de referencia `o` es el inicio del episodio nuevo. `u` es el último cuadro
  con control claro del equipo viejo en [fin del episodio viejo - 1, o); el tramo suelto es
  [u + 1, o). Se descarta el relevo si (a) hubo una pausa en medio (balón visto fuera en
  [fin del episodio viejo, o) o el start de un SET PIECES en [fin del episodio viejo - 4 s, o]), o
  (b) el tramo suelto no está vacío y el balón se ve en menos de VISIBLE de él. Cada relevo
  auditable se empareja con el cambio del modelo de la misma dirección más cercano a `o` dentro de
  WINDOW s; offset = t - o (positivo: el modelo corta tarde; negativo: le da el balón al equipo
  nuevo antes de que lo controle).
  Parte B, salidas del balón. Una salida vista es un cuadro con el balón fuera cuyo cuadro anterior
  estaba dentro. Se empareja con el paso del modelo de equipo a D más cercano dentro de WINDOW s;
  offset = d - t.

Umbrales, fijados antes de correr: vale la pena diseñar un arreglo para los relevos (parte A) solo
si, sobre los emparejados, la mediana del |offset| supera NEAR cuadros o más del 25% tiene
|offset| > FAR cuadros. Lo mismo, por separado, para las salidas (parte B). Si no se supera ningún
umbral, las fronteras se quedan como están.

Resultado (v7, la versión entregada; partido completo, una corrida):
  Parte A: 193 relevos de referencia; 33 descartados por pausa, 16 por balón no visto; 144
  auditables, 74 emparejados y 70 sin cambio del modelo cerca (48.6% de los auditables: el modelo
  se tragó el relevo o lo puso a más de 5 s). Sobre los 74 emparejados: mediana del offset -0.88 s
  (H->A -0.76, A->H -1.44), mediana del |offset| 0.88 s, 32% con |offset| <= 8, 46% con
  |offset| > 25, 68% temprano y 0% tarde. Mediana del tramo suelto 1.20 s. Posición del corte
  (n = 58): p25 0.00, mediana 0.10, p75 0.44; 9% < 0, 91% en [0, 1], 0% > 1.
  Parte B: 41 salidas vistas; 33 emparejadas, 7 ya eran D, 1 sin D. Mediana del offset 0.00 s,
  67% con |offset| <= 8, 21% con |offset| > 25.
  Con la v5 (la versión que auditó este estudio en #31) daba 75 emparejados y 69 sin cambio, con la
  misma mediana del offset, y 18% de salidas con |offset| > 25.
  Veredicto A: vale la pena un arreglo (mediana |offset| 22 > 8 cuadros; 46% con |offset| > 25).
  Veredicto B: las fronteras se quedan como están.
  Lectura: el modelo corta temprano, pegado al final del control viejo (mediana 0.10 del tramo
  suelto), nunca tarde. La referencia es un indicio nuestro, no la verdad del reto.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import cambios  # noqa: E402
import huecos  # noqa: E402
import v1  # noqa: E402
import v7  # noqa: E402
from atd import load  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = v1.FPS
EPISODE = cambios.EPISODE   # 1.0 s seguidos de control claro: el mismo indicio que cambios.py
VISIBLE = cambios.VISIBLE   # 0.8: fracción del tramo suelto con balón visible para poder juzgarlo
WINDOW = 5.0                # s: un cambio del modelo más lejos que esto ya no es el mismo relevo
NEAR, FAR = 8, 25           # cuadros: en EVALUATION.md, 8 tarde = 0.89 de spellF1 y 1 s = 0.65
LONG_FAR = 0.25             # fracción con |offset| > FAR a partir de la cual vale la pena arreglar
PAUSE_BEFORE = 4.0          # s antes del fin del episodio viejo en que un SET PIECES cuenta como pausa
MIN_LOOSE = 0.5             # s: tramo suelto mínimo para medir la posición relativa del corte


def nearest(frames, target):
    """Elemento de `frames` más cercano a target con |diferencia| <= WINDOW s, o None."""
    if len(frames) == 0:
        return None
    best = frames[np.abs(frames - target).argmin()]
    return int(best) if abs(best - target) <= WINDOW * FPS else None


def relays(labels, clear, seen, out, set_starts, first, last):
    """Relevos de un solo bloque (mitad): lista de (dirección, estado, offset, suelto, posición,
    o_abs). dirección 0 = H->A, 1 = A->H. estado: pausa | no visto | sin cambio | ok. o_abs es el
    cuadro absoluto de la frontera de referencia. Función pura."""
    sl = slice(first, last + 1)
    lab, clear, seen, out = labels[sl], clear[sl], seen[sl], out[sl]
    starts = np.array(set_starts, dtype=int) - first
    eps = []
    for k in (v1.H, v1.A):
        s, e = huecos.runs(clear[:, k])
        eps += [(int(a), int(b), k) for a, b in zip(s, e) if b - a >= EPISODE * FPS]
    eps.sort()
    change = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])
                            & np.isin(lab[:-1], ["H", "A"])) + 1
    rows = []
    for (_, end, old), (o, _, new) in zip(eps[:-1], eps[1:]):
        if old == new:
            continue
        direction = old                       # H(0)->A = 0, A(1)->H = 1
        u = end - 1 + np.flatnonzero(clear[end - 1:o, old])[-1]
        loose = o - (u + 1)
        pause = out[end:o].any() or ((starts >= end - PAUSE_BEFORE * FPS) & (starts <= o)).any()
        if pause:
            rows.append((direction, "pausa", None, loose, None, first + o))
            continue
        if loose and seen[u + 1:o].mean() < VISIBLE:
            rows.append((direction, "no visto", None, loose, None, first + o))
            continue
        same = change[(lab[change - 1] == "HA"[old]) & (lab[change] == "HA"[new])]
        t = nearest(same, o)
        if t is None:
            rows.append((direction, "sin cambio", None, loose, None, first + o))
        else:
            pos = (t - (u + 1)) / loose if loose >= MIN_LOOSE * FPS else None
            rows.append((direction, "ok", t - o, loose, pos, first + o))
    return rows


def exits(labels, inside, out, first, last):
    """Salidas vistas de un bloque: lista de (estado, offset). estado: ok | ya D | sin D."""
    sl = slice(first, last + 1)
    lab, inside, out = labels[sl], inside[sl], out[sl]
    t_out = np.flatnonzero(out[1:] & inside[:-1]) + 1
    d_all = np.flatnonzero(np.isin(lab[:-1], ["H", "A"]) & (lab[1:] == "D")) + 1
    rows = []
    for t in t_out:
        d = nearest(d_all, t)
        rows.append(("ok", d - t) if d is not None else
                    ("ya D", None) if lab[t] == "D" else ("sin D", None))
    return rows


def stats(offsets):
    """(mediana, mediana |.|, %<=NEAR, %>FAR, %temprano, %tarde); offsets en cuadros -> medianas en s."""
    o = np.asarray(offsets, dtype=float)
    if len(o) == 0:
        return (np.nan,) * 6
    a = np.abs(o)
    return (np.median(o) / FPS, np.median(a) / FPS, 100 * (a <= NEAR).mean(), 100 * (a > FAR).mean(),
            100 * (o < -NEAR).mean(), 100 * (o > NEAR).mean())


def verdict(offsets):
    o = np.abs(np.asarray(offsets, dtype=float))
    if len(o) == 0:
        return "sin datos"
    why = []
    if np.median(o) > NEAR:
        why.append("mediana |offset| {:.0f} > {} cuadros".format(np.median(o), NEAR))
    if (o > FAR).mean() > LONG_FAR:
        why.append("{:.0%} con |offset| > {} cuadros".format((o > FAR).mean(), FAR))
    return "vale la pena un arreglo (" + "; ".join(why) + ")" if why else \
        "las fronteras se quedan como están"


def print_stats(columns):
    """columns: {nombre: lista de offsets}."""
    names = ("mediana offset (s)", "mediana |offset| (s)", "% |offset| <= {}".format(NEAR),
             "% |offset| > {}".format(FAR), "% temprano (< -{})".format(NEAR),
             "% tarde (> +{})".format(NEAR))
    table = {n: stats(v) for n, v in columns.items()}
    for i, name in enumerate(names):
        print("{:<26}".format(name) + "".join("{:>10.2f}".format(table[n][i]) for n in columns))


def _check():
    """Autochequeo sintético: H controla 2 s, 1 s suelto con balón visible, A controla 2 s, y el
    modelo cambia 10 cuadros después de o."""
    n, o = 125, 75
    clear = np.zeros((n, 2), dtype=bool)
    clear[:50, v1.H] = True
    clear[o:, v1.A] = True
    labels = np.array(["H"] * (o + 10) + ["A"] * (n - o - 10))
    seen, out = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    rows = relays(labels, clear, seen, out, [], 0, n - 1)
    assert rows == [(0, "ok", 10, 25, (85 - 50) / 25, o)], rows
    out[60] = True   # una pausa en medio: se descarta
    assert relays(labels, clear, seen, out, [], 0, n - 1)[0][1] == "pausa"
    out[60], seen[55:75] = False, False   # balón no visto en el tramo suelto
    assert relays(labels, clear, seen, out, [], 0, n - 1)[0][1] == "no visto"
    inside = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    out[30:] = True
    inside[30:] = False
    inside[29] = True
    labels = np.array(["H"] * 35 + ["D"] * (n - 35))
    assert exits(labels, inside, out, 0, n - 1) == [("ok", 5)]


def main():
    _check()
    ball, players, team, pitch = load()
    tagging = load_tagging()
    labels = v7.run(ball, players, team, pitch, tagging)
    clear = v1.control(ball, players, team, pitch)
    seen = ~np.isnan(ball[:, 0])
    _, out, inside, _, _ = v1.motion(ball, players, pitch, anchored=True)
    set_starts = [e["start_frame"] for e in tagging if e["code"] == "SET PIECES"]

    rows, outs = [], []
    for _, first, last in half_video_blocks(GAME):
        rows += relays(labels, clear, seen, out, set_starts, first, last)
        outs += exits(labels, inside, out, first, last)

    print("PARTE A: relevos entre equipos (referencia: inicio del episodio de control nuevo)\n")
    cols = {"H->A": [r for r in rows if r[0] == 0], "A->H": [r for r in rows if r[0] == 1],
            "total": rows}
    print("{:<26}".format("") + "".join("{:>10}".format(n) for n in cols))
    for label, state in (("relevos de referencia", None), ("descartados por pausa", "pausa"),
                         ("descartados no visto", "no visto")):
        print("{:<26}".format(label) + "".join(
            "{:>10d}".format(sum(state is None or r[1] == state for r in c)) for c in cols.values()))
    aud = {n: [r for r in c if r[1] in ("sin cambio", "ok")] for n, c in cols.items()}
    ok = {n: [r for r in c if r[1] == "ok"] for n, c in cols.items()}
    print("{:<26}".format("auditables") + "".join("{:>10d}".format(len(v)) for v in aud.values()))
    print("{:<26}".format("emparejados") + "".join("{:>10d}".format(len(v)) for v in ok.values()))
    print("{:<26}".format("sin cambio del modelo") + "".join(
        "{:>10d}".format(len(aud[n]) - len(ok[n])) for n in cols))
    print_stats({n: [r[2] for r in v] for n, v in ok.items()})
    print("{:<26}".format("mediana tramo suelto (s)") + "".join(
        "{:>10.2f}".format(np.median([r[3] for r in v]) / FPS if v else np.nan) for v in aud.values()))
    print("veredicto A:", verdict([r[2] for r in ok["total"]]))

    pos = np.array([r[4] for r in rows if r[4] is not None])
    print("\nPosición del corte en el tramo suelto (>= {} s; 0 = fin del control viejo, 1 = inicio "
          "del nuevo), n = {}".format(MIN_LOOSE, len(pos)))
    if len(pos):
        print("p25 {:.2f}  mediana {:.2f}  p75 {:.2f}  | < 0: {:.0f}%  en [0, 1]: {:.0f}%  > 1: {:.0f}%".format(
            *np.percentile(pos, [25, 50, 75]), 100 * (pos < 0).mean(),
            100 * ((pos >= 0) & (pos <= 1)).mean(), 100 * (pos > 1).mean()))

    print("\nPARTE B: salidas del balón (equipo -> D)\n")
    matched = [o for s, o in outs if s == "ok"]
    print("salidas vistas {}  emparejadas {}  ya era D {}  sin D {}".format(
        len(outs), len(matched), sum(s == "ya D" for s, _ in outs), sum(s == "sin D" for s, _ in outs)))
    print_stats({"salidas": matched})
    print("veredicto B:", verdict(matched))


if __name__ == "__main__":
    main()
