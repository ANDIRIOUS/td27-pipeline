"""Arrastre en balón suelto (#32): ¿la posesión se queda con el último equipo que la controló?
Sin etiquetas y sobre el partido completo.

    python solution/arrastre.py

Por qué hace falta. La auditoría de fronteras (#31, fronteras.py) midió que en los relevos entre
equipos la v5 corta temprano: mediana del offset -0.88 s, 67% temprano, 0% tarde, y el corte cae
pegado al final del control viejo (mediana 0.10 del tramo suelto). El reto (CHALLENGE.md) dice:
"In phases where the ball is contested or loose, possession stays with the last team that had it",
y un toque sin control no es un cambio. En el Viterbi, durante un tramo suelto nadie suma evidencia
de control, así que el corte lo deciden señales débiles (fases 0.05 + colectivo hasta 2/25 = 0.08,
suma 0.13 nats por cuadro) o un desempate arbitrario.

Mecanismo. Un cuadro "suelto" es uno con el balón visto dentro del campo y sin control claro de
nadie. A esos cuadros se les suma EVIDENCIA nats al equipo del último control claro, si fue hace
<= MAX_CARRY s y el balón no se vio fuera después. Variante "puente": solo en tramos sueltos entre
dos controles del MISMO equipo (el balón va y vuelve, o un desvío sin control), sin tocar los
relevos reales.

Rejilla (fijada a priori):
  EVIDENCIA 0.25 y 0.5 nats por cuadro. Debe ganarle a las señales débiles que hoy deciden el corte
  (0.13) y quedar muy por debajo de las señales de balón muerto (STILL 1.0, PENDING 1.5, OUT 4.0) y
  del control claro (1.0). Con 0.25, un tramo suelto de 1.2 s (la mediana) acumula ~7.5 nats; con
  0.5, ~15, contra 20 de costo de cambio.
  MAX_CARRY 4 s: un balón largo tarda hasta ~4 s; más allá ya no sabemos de quién es.
  BRIDGE 0.5 nats para el puente.

Medidas (v5 y las tres variantes, partido completo): cobertura del arrastre; auditoría de
cambios.py (< 2 s, no marcadas, inventadas); relevos y salidas de fronteras.py (más el desglose de
los relevos "sin cambio"); % de D en muerto seguro; detector de reinicios; y la prueba de huecos
pareada contra v5.

Ojo: la referencia de fronteras es un indicio nuestro (lectura de la regla del reto), no la verdad
del reto, y por construcción el arrastre empuja el corte hacia esa referencia. Por eso la decisión
se apoya también en guardas independientes de ella (muerto seguro, reinicios, huecos, salidas,
auditoría de posesiones).

Regla (pre-registrada, aplicada en `verdict`):
  Pasa las guardas si, contra v5: la suma no marcadas + inventadas no sube; el equipo correcto
  pareado no baja más de 1 EE; el muerto falso pareado no sube más de 1 EE; ninguna columna de
  muerto seguro baja más de 1.0 punto; los saques a <= 1 s no bajan en más de uno; en salidas, la
  mediana |offset| sigue <= 8 cuadros y las de |offset| > 25 no suben en más de 2.
  Mejora si pasa las guardas y además: los relevos buenos (emparejados con |offset| <= 8) suben al
  menos GOOD_GAIN y los "sin cambio" no suben en más de 3; o la suma no marcadas + inventadas baja
  al menos 5.
  Si ninguna mejora: no se agrega nada (complejidad innecesaria). Si varias mejoran: la de más
  relevos buenos; empate: el puente (el cambio más conservador, y solo gana si empata con el
  arrastre), y luego la menor evidencia.

Resultado (partido completo, una corrida; v5 / puente 0.5 / arrastre 0.25 / arrastre 0.5):
  Cobertura: 11.5% / 22.0% / 22.0% de lo calificado (puente / arrastre).
  Auditoría (no marcadas + inventadas): 43 / 50 / 47 / 60; suben las inventadas (16 / 29 / 25 / 43).
  Relevos buenos (|offset| <= 8): 25 / 24 / 41 / 50; sin cambio: 69 / 57 / 60 / 41; mediana del
  offset -0.88 / -1.00 / -0.30 / -0.12 s; temprano 67% / 71% / 50% / 49%; tarde 0% / 1% / 1% / 3%.
  Salidas, muerto seguro y reinicios: sin cambios relevantes (saques a <= 1 s: 11 / 12 / 12 / 12).
  Huecos pareados (contra v5): equipo correcto +0.25 / +0.13 / +0.08 pt (EE 0.11 / 0.17 / 0.21);
  muerto falso +0.05 / +0.07 / +0.07 pt (EE 0.04 / 0.05 / 0.06), más de 1 EE hacia arriba.
  Veredicto: ninguna variante pasa las guardas (la suma no marcadas + inventadas sube en las tres y
  el muerto falso pareado sube más de 1 EE). El arrastre sí acerca los relevos a la referencia,
  pero la regla pre-registrada dice que no se agrega nada: la v5 se queda.

Segunda vuelta (pre-registrada). Hipótesis de por qué suben las inventadas: el arrastre va al equipo
del último CUADRO con control claro, así que un toque breve del rival (despeje, desvío) seguido de
hasta 4 s de balón suelto junta hasta 25-50 nats y le fabrica una posesión a quien solo tocó el
balón. El reto dice que un toque sin control no es cambio de posesión y que el balón suelto se
queda con el último equipo que LO TUVO. Mecanismo: variante "episodio", el arrastre va al último
equipo con un episodio de control (>= EPISODE s seguidos de control claro, el mismo indicio de
cambios.py); "suelto" sigue definido con el control claro completo. Mismas evidencias (0.25 y 0.5).
Cambio de regla, declarado antes de correr: una guarda de huecos falla solo si el cambio pareado
supera 1 EE Y además MATERIAL (0.5 pt), porque en la primera vuelta 1 EE eran ~0.05 pt, sin
significado práctico. Todo lo demás de la regla queda idéntico. Con la guarda nueva las tres
variantes de la primera vuelta siguen sin pasar (falla la suma no marcadas + inventadas). Esta es la
última vuelta: si ninguna variante pasa, el arrastre se descarta y la frontera temprana queda
documentada como limitación.

Resultado de la segunda vuelta (v5 / episodio 0.25 / episodio 0.5; las cuatro versiones de la primera
vuelta dieron exactamente las mismas cifras):
  Cobertura: 16.2% de lo calificado (ambas).
  Auditoría (no marcadas + inventadas): 43 / 37 / 31; inventadas 16 / 13 / 14.
  Relevos buenos: 25 / 45 / 60; sin cambio 69 / 63 / 50; lejanos 34 / 24 / 20; mediana del offset
  -0.88 / 0.00 / 0.00 s; temprano 67% / 44% / 35%; tarde 0% / 0% / 1%.
  Salidas, muerto seguro y reinicios: sin cambios relevantes (saques a <= 1 s: 11 / 11 / 11).
  Huecos pareados: equipo correcto +0.02 / +0.04 pt; muerto falso -0.00 / -0.03 pt.
  Veredicto: ambas pasan las guardas y mejoran (buenos +20 / +35, sin cambio -6 / -19, suma -6 / -12).
  Elegida por la regla: episodio 0.5 (más relevos buenos).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import cambios  # noqa: E402
import fronteras  # noqa: E402
import huecos  # noqa: E402
import reinicios  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v5  # noqa: E402
from atd import load  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = v1.FPS
# Evidencia de arrastre en nats por cuadro. Debe ganarle a las señales débiles que hoy deciden el corte
# en un tramo suelto (fases 0.05 + colectivo hasta 2/25 = 0.08, suma 0.13) y quedar muy por debajo de
# las señales de balón muerto (STILL_EVIDENCE 1.0, PENDING 1.5, OUT 4.0) y del control claro (1.0).
GRID = (0.25, 0.5)
MAX_CARRY = 4.0   # s: un balón largo tarda hasta ~4 s; más allá ya no sabemos de quién es
EPISODE = cambios.EPISODE  # 1.0 s seguidos de control claro: "conservar el balón" en su lectura más corta
# Segunda vuelta: una guarda de huecos falla solo si el cambio pareado supera 1 EE Y 0.5 puntos. En la
# primera vuelta la desviación pareada fue ~0.2 pt, así que 1 EE eran 0.05 pt, una diferencia sin
# significado práctico; 0.5 pt es la cuarta parte de los 2 pt que se exigieron para llamar mejora en
# el barrido de muertos falsos (#30).
MATERIAL = 0.005
BRIDGE = 0.5      # variante puente (solo tramos sueltos entre dos controles del MISMO equipo)
GOOD_GAIN = 10    # relevos: el ruido de spellF1 es ~0.04, unas 10 de 239 posesiones


def episodes(clear, min_frames):
    """(n, 2) bool: `clear` restringido a los tramos de al menos min_frames cuadros seguidos."""
    keep = np.zeros_like(clear)
    for k in (v1.H, v1.A):
        for a, b in zip(*huecos.runs(clear[:, k])):
            if b - a >= min_frames:
                keep[a:b, k] = True
    return keep


def carry(clear, inside, out, max_frames, bridge_only=False, episode=0):
    """(n,) int8: equipo (0 = H, 1 = A) que recibe arrastre en cada cuadro; -1 = ninguno.
    Cuadro suelto = balón visto dentro del campo (inside) y sin control claro de nadie.
    El arrastre va al equipo del último cuadro con control claro, si ese control fue hace
    <= max_frames y el balón no se vio fuera (out) después de él.
    bridge_only: además exige que el siguiente control claro sea del mismo equipo y que entre ambos
    controles haya <= max_frames (solo se usa con episode=0).
    episode > 0 (cuadros): el dueño del arrastre sale solo de los tramos de control claro de al
    menos `episode` cuadros; "suelto" se sigue definiendo con el control claro completo."""
    n = len(clear)
    idx = np.arange(n)
    held = clear.any(1)                                   # con control claro de alguien no es suelto
    owner = episodes(clear, episode) if episode else clear
    has = owner.any(1)
    prev_c = np.maximum.accumulate(np.where(has, idx, -1))
    prev_out = np.maximum.accumulate(np.where(out, idx, -1))
    who = np.where(prev_c >= 0, owner[np.maximum(prev_c, 0), 1], 0)   # 0 = H, 1 = A
    ok = inside & ~held & (prev_c >= 0) & (idx - prev_c <= max_frames) & (prev_out <= prev_c)
    if bridge_only:
        # Lo mismo sobre el arreglo invertido: siguiente control y siguiente salida.
        rev = slice(None, None, -1)
        rc = np.maximum.accumulate(np.where(held[rev], idx, -1))[rev]
        ro = np.maximum.accumulate(np.where(out[rev], idx, -1))[rev]
        next_c, next_out = n - 1 - rc, n - 1 - ro
        has_c, has_out = rc >= 0, ro >= 0
        same = clear[np.minimum(next_c, n - 1), 1] == who.astype(bool)
        ok &= has_c & same & (next_c - prev_c <= max_frames) & ~(has_out & (next_out < next_c))
    return np.where(ok, who, -1).astype(np.int8)


def _check():
    """Autochequeo sintético de `carry`."""
    def build(n, h, a, loose_out=(), unseen=()):
        clear = np.zeros((n, 2), dtype=bool)
        for s, e in h:
            clear[s:e, v1.H] = True
        for s, e in a:
            clear[s:e, v1.A] = True
        inside, out = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
        for i in loose_out:
            out[i], inside[i] = True, False
        for i in unseen:
            inside[i] = False
        return clear, inside, out

    # (a) H controla, 25 sueltos, A controla: arrastre de H; con puente, ninguno.
    clear, inside, out = build(75, [(0, 25)], [(50, 75)])
    got = carry(clear, inside, out, 100)
    assert (got[25:50] == v1.H).all() and (got[:25] == -1).all() and (got[50:] == -1).all(), got
    assert (carry(clear, inside, out, 100, True) == -1).all()
    # (b) H controla, 25 sueltos, H controla: arrastre de H en ambos modos.
    clear, inside, out = build(75, [(0, 25), (50, 75)], [])
    for mode in (False, True):
        got = carry(clear, inside, out, 100, mode)
        assert (got[25:50] == v1.H).all() and (got[:25] == -1).all() and (got[50:] == -1).all(), (mode, got)
    # (b2) lo mismo con A: el arrastre es de A (1), no de H.
    clear, inside, out = build(75, [], [(0, 25), (50, 75)])
    assert (carry(clear, inside, out, 100, True)[25:50] == v1.A).all()
    # (c) tramo suelto más largo que max_frames: solo los primeros max_frames cuadros.
    clear, inside, out = build(60, [(0, 10)], [])
    got = carry(clear, inside, out, 20)
    assert (got[10:30] == v1.H).all() and (got[30:] == -1).all(), got
    # (c2) puente: controles del mismo equipo separados por más de max_frames: ninguno.
    clear, inside, out = build(60, [(0, 10), (50, 60)], [])
    assert (carry(clear, inside, out, 20, True) == -1).all()
    # (d) balón visto fuera en medio: sin arrastre después de la salida (y con puente, ninguno).
    clear, inside, out = build(75, [(0, 25)], [(50, 75)], loose_out=[35])
    got = carry(clear, inside, out, 100)
    assert (got[25:35] == v1.H).all() and (got[35:50] == -1).all(), got
    clear, inside, out = build(75, [(0, 25), (50, 75)], [], loose_out=[35])
    assert (carry(clear, inside, out, 100, True) == -1).all()
    # (e) cuadros con balón no visto no reciben arrastre.
    clear, inside, out = build(75, [(0, 25)], [(50, 75)], unseen=range(30, 40))
    got = carry(clear, inside, out, 100)
    assert (got[30:40] == -1).all() and (got[25:30] == v1.H).all() and (got[40:50] == v1.H).all(), got
    # (f) H controla 50 (episodio), 10 sueltos, A toca 5, 30 sueltos: con episode=25 todo suelto es de H.
    clear, inside, out = build(95, [(0, 50)], [(60, 65)])
    got = carry(clear, inside, out, 100, episode=25)
    assert (got[50:60] == v1.H).all() and (got[65:] == v1.H).all() and (got[60:65] == -1).all(), got
    got = carry(clear, inside, out, 100)
    assert (got[65:] == v1.A).all() and (got[50:60] == v1.H).all(), got
    # (g) H controla 10 cuadros (no es episodio) y luego suelto: con episode=25 nadie recibe arrastre.
    clear, inside, out = build(60, [(0, 10)], [])
    assert (carry(clear, inside, out, 100, episode=25) == -1).all()


def run(evidence, bridge_only=False, episode=0):
    """La v5 con arrastre: (ball, players, team, pitch, tagging) -> etiquetas. Autocontenida sobre
    el `ball` que reciba, porque la prueba de huecos le pasa el balón con huecos simulados."""
    def version(ball, players, team, pitch, tagging):
        score = v2.evidence(ball, players, team, pitch, tagging, phase_weight=v3.PHASE_WEIGHT)
        clear = v1.control(ball, players, team, pitch)
        _, out, inside, _, _ = v1.motion(ball, players, pitch, anchored=True)
        who = carry(clear, inside, out, int(MAX_CARRY * FPS), bridge_only, episode)
        for k in (v1.H, v1.A):
            score[who == k, k] += evidence
        return v1.label(score, tagging, v5.kickoff_restarts(ball, pitch, tagging))
    return version


def sure_dead(ball, tagging, graded, crowd):
    """Copia de muertos.sure_dead: tres máscaras de muerto seguro (fuera, caminando, 4 s antes de
    un SET PIECES)."""
    bx, by = ball[:, 0], ball[:, 1]
    mx, my = v1.OUT_MARGIN
    out = graded & ~np.isnan(bx) & ((bx < -mx) | (bx > 1 + mx) | (by < -my) | (by > 1 + my))
    walking = graded & (crowd < 1.0)
    before = np.zeros(len(ball), dtype=bool)
    for e in tagging:
        if e["code"] == "SET PIECES":
            before[max(e["start_frame"] - 4 * FPS, 0):e["start_frame"]] = True
    return out, walking, before & graded


def restarts(labels, ball, tagging, pitch):
    """(saques detectados, con reinicio en las etiquetas, mediana |error| s, a <= 1 s), como reinicios.py."""
    speed = reinicios.ball_speed(ball, pitch)
    errors, found, near = [], 0, 0
    for e in tagging:
        kind = (e["tags"].get("Type") or "").strip().lower()
        if e["code"] != "SET PIECES" or kind not in reinicios.PLACED:
            continue
        k = reinicios.kick_frame(speed, ball, e["start_frame"], kind, pitch)
        if k is None:
            continue
        found += 1
        f = reinicios.label_restart(labels, e["start_frame"])
        if f is not None:
            errors.append((f - k) / FPS)
            near += abs(f - k) <= FPS
    return found, len(errors), np.median(np.abs(errors)), int(near)


def relay_measures(labels, clear, seen, out, set_starts):
    """Medidas de relevos de fronteras.py sobre el partido completo y desglose de los sin cambio."""
    rows = []
    for _, first, last in half_video_blocks(GAME):
        rows += fronteras.relays(labels, clear, seen, out, set_starts, first, last)
    ok = np.array([r[2] for r in rows if r[1] == "ok"], dtype=float)
    none = [r for r in rows if r[1] == "sin cambio"]
    split = dict.fromkeys(("pasa por D", "se queda con el viejo", "ya era del nuevo", "otro"), 0)
    for direction, _, _, loose, _, o_abs in none:
        old, new = "HA"[direction], "HA"[1 - direction]
        u, fin = o_abs - loose - 1, o_abs + FPS - 1
        key = ("pasa por D" if labels[u] == old and labels[fin] == new else
               "se queda con el viejo" if labels[fin] == old else
               "ya era del nuevo" if labels[u] == new else "otro")
        split[key] += 1
    s = fronteras.stats(ok)
    return {"auditables": len(ok) + len(none), "emparejados": len(ok), "sin cambio": len(none),
            "buenos": int((np.abs(ok) <= fronteras.NEAR).sum()),
            "lejanos": int((np.abs(ok) > fronteras.FAR).sum()),
            "mediana offset (s)": s[0], "mediana |offset| (cuadros)": float(np.median(np.abs(ok))) if len(ok) else np.nan,
            "% temprano": s[4], "% tarde": s[5], "split": split}


def exit_measures(labels, inside, out):
    outs = []
    for _, first, last in half_video_blocks(GAME):
        outs += fronteras.exits(labels, inside, out, first, last)
    off = np.abs([o for s, o in outs if s == "ok"])
    return {"emparejadas": len(off), "mediana |offset| (cuadros)": float(np.median(off)),
            "lejanas": int((off > fronteras.FAR).sum()), "ya era D": sum(s == "ya D" for s, _ in outs),
            "sin D": sum(s == "sin D" for s, _ in outs)}


def verdict(m, names, evidence, paired, se):
    """Aplica la regla a cada versión contra v5. m: {nombre: medidas}. Devuelve (ganadora o None, líneas)."""
    base, lines, better = m["v5"], [], []
    for n in names:
        x = m[n]
        guards = {
            "no marcadas+inventadas no sube": x["suma"] <= base["suma"],
            "equipo correcto: no baja > 1 EE y > 0.5 pt": not (
                paired[n][:, 0].mean() < -se[n][0] and paired[n][:, 0].mean() < -MATERIAL),
            "muerto falso: no sube > 1 EE y > 0.5 pt": not (
                paired[n][:, 1].mean() > se[n][1] and paired[n][:, 1].mean() > MATERIAL),
            "muerto seguro >= -1.0 pt": all(a - b >= -0.01 for a, b in zip(x["dead"], base["dead"])),
            "saques <= 1 s no bajan > 1": x["hits"] >= base["hits"] - 1,
            "salidas: mediana <= 8 y lejanas no suben > 2": (
                x["exits"]["mediana |offset| (cuadros)"] <= fronteras.NEAR
                and x["exits"]["lejanas"] <= base["exits"]["lejanas"] + 2),
        }
        passes = all(guards.values())
        gain_rel = x["rel"]["buenos"] - base["rel"]["buenos"]
        gain_none = x["rel"]["sin cambio"] - base["rel"]["sin cambio"]
        improves = passes and ((gain_rel >= GOOD_GAIN and gain_none <= 3) or base["suma"] - x["suma"] >= 5)
        failed = [g for g, v in guards.items() if not v]
        lines.append("{:<14} guardas: {} | buenos {:+d}, sin cambio {:+d}, suma {:+d} -> {}".format(
            n, "ok" if passes else "NO (" + "; ".join(failed) + ")", gain_rel, gain_none,
            x["suma"] - base["suma"], "MEJORA" if improves else "no mejora"))
        if improves:
            better.append(n)
    if not better:
        lines.append("ninguna mejora: no se agrega nada (v5 se queda)")
        return None, lines
    win = min(better, key=lambda n: (-m[n]["rel"]["buenos"], not n.startswith("puente"), evidence[n]))
    lines.append("elegida por la regla: " + win)
    return win, lines


def main():
    _check()
    ball, players, team, pitch = load()
    tagging = load_tagging()
    base_labels = v5.run(ball, players, team, pitch, tagging)
    assert np.array_equal(run(0.0)(ball, players, team, pitch, tagging), base_labels), \
        "el truco aditivo no reproduce la v5"

    graded = v2.graded_mask(len(ball))
    crowd = np.nan_to_num(v1.crowd_speed(players), nan=0.0)
    clear = v1.control(ball, players, team, pitch)
    seen = ~np.isnan(ball[:, 0])
    _, out, inside, _, _ = v1.motion(ball, players, pitch, anchored=True)
    set_starts = [e["start_frame"] for e in tagging if e["code"] == "SET PIECES"]
    masks = sure_dead(ball, tagging, graded, crowd)

    versions = {"v5": v5.run, "puente 0.5": run(BRIDGE, True)}
    versions.update({"arrastre {:g}".format(g): run(g) for g in GRID})
    versions.update({"episodio {:g}".format(g): run(g, episode=int(EPISODE * FPS)) for g in GRID})
    evidence = {"v5": 0.0, "puente 0.5": BRIDGE, **{"arrastre {:g}".format(g): g for g in GRID},
                **{"episodio {:g}".format(g): g for g in GRID}}
    modes = {"puente 0.5": (True, 0), "arrastre 0.25": (False, 0), "arrastre 0.5": (False, 0),
             "episodio 0.25": (False, int(EPISODE * FPS)), "episodio 0.5": (False, int(EPISODE * FPS))}
    m = {}
    for name, version in versions.items():
        labels = base_labels if name == "v5" else version(ball, players, team, pitch, tagging)
        a, r = cambios.audit(labels, clear, seen), restarts(labels, ball, tagging, pitch)
        m[name] = {"audit": a, "suma": a[2] + a[3], "rel": relay_measures(labels, clear, seen, out, set_starts),
                   "exits": exit_measures(labels, inside, out),
                   "dead": [(labels == "D")[k].mean() for k in masks],
                   "hits": r[3], "restarts": r}

    print("== 1. Cobertura del arrastre (cuadros calificados)")
    for name, (bridge, ep) in modes.items():
        who = carry(clear, inside, out, int(MAX_CARRY * FPS), bridge, ep)
        print("{:<14} {:>6.2f}%".format(name, 100 * (graded & (who >= 0)).sum() / graded.sum()))

    print("\n== 2. Auditoría de posesiones (cambios.audit)")
    print("{:<14} {:>11} {:>6} {:>11} {:>10} {:>6}".format("versión", "posesiones", "< 2 s", "no marcadas",
                                                         "inventadas", "suma"))
    for name in versions:
        print("{:<14} {:>11} {:>6} {:>11} {:>10} {:>6}".format(name, *m[name]["audit"], m[name]["suma"]))

    print("\n== 3. Relevos (fronteras.relays)")
    keys = [k for k in m["v5"]["rel"] if k != "split"]
    print("{:<30}".format("") + "".join("{:>14}".format(n) for n in versions))
    for k in keys:
        print("{:<30}".format(k) + "".join("{:>14.2f}".format(m[n]["rel"][k]) if isinstance(m[n]["rel"][k], float)
                                           else "{:>14d}".format(m[n]["rel"][k]) for n in versions))
    for k in m["v5"]["rel"]["split"]:
        print("{:<30}".format("  sin cambio: " + k) + "".join("{:>14d}".format(m[n]["rel"]["split"][k]) for n in versions))

    print("\n== 4. Salidas del balón (fronteras.exits)")
    print("{:<30}".format("") + "".join("{:>14}".format(n) for n in versions))
    for k in m["v5"]["exits"]:
        print("{:<30}".format(k) + "".join("{:>14.1f}".format(m[n]["exits"][k]) if isinstance(m[n]["exits"][k], float)
                                           else "{:>14d}".format(m[n]["exits"][k]) for n in versions))

    print("\n== 5. Muerto seguro: % de D")
    print("{:<14} {:>11} {:>12} {:>14}".format("versión", "D fuera", "D caminando", "D pre-reinicio"))
    for name in versions:
        print("{:<14} {:>11.1%} {:>12.1%} {:>14.1%}".format(name, *m[name]["dead"]))

    print("\n== 6. Detector de reinicios")
    print("{:<14} {:>9} {:>13} {:>11} {:>8}".format("versión", "saques", "con reinicio", "|mediana|", "<= 1 s"))
    for name in versions:
        print("{:<14} {:>9} {:>13} {:>10.1f}s {:>8}".format(name, *m[name]["restarts"]))

    print("\n== 7. Prueba de huecos")
    truth = np.where(clear[:, v1.H], 0, np.where(clear[:, v1.A], 1, -1))
    lengths = np.array([e - s for s, e in huecos.live_gaps(ball, crowd, graded)])
    results = huecos.simulate(versions, ball, players, team, pitch, tagging, graded, crowd, clear,
                              truth, lengths)
    huecos.table(results, "v5")
    paired = {n: results[n] - results["v5"] for n in versions}
    se = {n: p.std(0) / np.sqrt(huecos.REPLICAS) for n, p in paired.items()}
    print("\npareado contra v5 (por réplica)")
    print("{:<14} {:>20} {:>20}".format("versión", "equipo correcto", "muerto falso"))
    print("{:<14} {:>20} {:>20}".format("", "media / std / EE", "media / std / EE"))
    for n in versions:
        print("{:<14} {:>+7.2%} {:>5.2%} {:>5.2%}   {:>+7.2%} {:>5.2%} {:>5.2%}".format(
            n, paired[n][:, 0].mean(), paired[n][:, 0].std(), se[n][0],
            paired[n][:, 1].mean(), paired[n][:, 1].std(), se[n][1]))

    print("\n== 8. Veredicto")
    names = [n for n in versions if n != "v5"]
    print("\n".join(verdict(m, names, evidence, paired, se)[1]))


if __name__ == "__main__":
    main()
