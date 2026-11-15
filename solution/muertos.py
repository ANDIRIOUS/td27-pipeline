"""Muertos falsos en huecos (A3): ¿cuánto pesa no ver el balón? Sin etiquetas y sobre el partido
completo.

    python solution/muertos.py

Por qué hace falta. En `v1.evidence` hay `dead += MISSING_EVIDENCE * ~seen`, con 0.3 nats por cuadro
sin balón hacia D. Cada cuadro cuenta como una observación independiente, pero los huecos del
tracker duran una mediana de 10 s: un solo hueco acumula ~75 nats hacia D, contra 20 nats que cuesta
entrar a D. En la prueba de huecos la v5 deja 16.9% ± 3.8% de juego vivo como D ("muerto falso"). Un
D falso dentro de una posesión la parte en dos, y la métrica del reto (spellF1) califica posesiones
completas. Se quiere saber si bajar la constante reduce los muertos falsos SIN tapar balones muertos
reales. Solo cambia esa constante; todo lo demás es la v5.

Cómo se mide, con los mismos huecos simulados para todos los valores:
  1. Referencia a priori (solo informativa): fracción de cuadros sin balón cuando los jugadores
     caminan, justo antes de un SET PIECES y cuando corren, con sus log-cocientes.
  2. Guardas sobre el partido completo: % de D en los cuadros de muerto seguro (balón visto fuera,
     jugadores caminando, 4 s antes de un SET PIECES), y la auditoría de posesiones de cambios.py.
  3. Detector de reinicios (reinicios.py): error del reinicio de las etiquetas contra el saque.
  4. Prueba de huecos (huecos.py): equipo correcto y muerto falso, y su diferencia pareada contra 0.3.

Regla fijada antes de correr:
  - Un valor PASA LAS GUARDAS si: (1) ninguna de las tres columnas de muerto seguro baja más de 1.0
    punto porcentual respecto a 0.3; (2) el equipo correcto pareado contra 0.3 no es peor que -1
    error estándar; (3) en el detector de reinicios, el número de saques a <= 1 s no baja en más de
    uno respecto a 0.3.
  - Un valor MEJORA si pasa las guardas y su muerto falso pareado contra 0.3 baja al menos 2 puntos
    porcentuales y más de 2 errores estándar.
  - Si ninguno mejora: se queda 0.3 y no hay versión nueva. Si varios mejoran: se elige el más alto
    (el que menos cambia la v5) entre los que quedan a menos de 1 error estándar pareado del mejor
    muerto falso.

Resultado (20 réplicas, semilla 2027; las guardas del partido completo casi no se mueven: D fuera
96.6%, D caminando 97.2%, D pre-reinicio 83.1% -> 82.6%; saques a <= 1 s: 11 de 17 en todos):

    falta    equipo correcto    muerto falso    muerto falso pareado vs 0.3
    0.3          62.0%             16.9%            --
    0.15         63.7%             14.5%            -2.4% (EE 0.21%)
    0.05         64.5%             12.8%            -4.1% (EE 0.28%)
    0.012        65.3%             11.9%            -5.0% (EE 0.34%)
    0            65.4%             11.8%            -5.1% (EE 0.34%)

Veredicto de la regla: los cuatro valores pasan las guardas y mejoran; el mejor muerto falso es 0 y
es el único a < 1 EE pareado de sí mismo, así que la regla elige 0.0 (no mira el balón faltante).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "starter")]
import cambios  # noqa: E402
import huecos  # noqa: E402
import reinicios  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v5  # noqa: E402
from atd import load  # noqa: E402
from load_smart_tagging import load as load_tagging  # noqa: E402

FPS = v1.FPS
BASE = 0.3
# Nats por cuadro sin balón hacia D.
#   0.3    el valor de v1 a v5: cada cuadro sin balón cuenta como una observación independiente;
#   0.012  = 0.3 / 25: una observación por segundo, el mismo criterio que COLLECTIVE_WEIGHT (v2) y
#          que el barrido del peso de las fases (v3);
#   0.0    no ver el balón no dice nada sobre vivo o muerto (falta en el 41% de lo calificado, y no
#          solo cuando el balón está muerto);
#   0.15 y 0.05 llenan el hueco en escala aproximadamente logarítmica.
GRID = (0.3, 0.15, 0.05, 0.012, 0.0)


def run(missing):
    """La v5 con otra evidencia de balón faltante: (ball, players, team, pitch, tagging) -> etiquetas."""
    def version(ball, players, team, pitch, tagging):
        score = v2.evidence(ball, players, team, pitch, tagging, phase_weight=v3.PHASE_WEIGHT)
        score[:, v1.D] += (missing - v1.MISSING_EVIDENCE) * np.isnan(ball[:, 0])
        return v1.label(score, tagging, v5.kickoff_restarts(ball, pitch, tagging))
    return version


def sure_dead(ball, tagging, graded, crowd):
    """Copia de huecos.dead_side; se unifica al integrar. Tres máscaras de muerto seguro."""
    bx, by = ball[:, 0], ball[:, 1]
    mx, my = v1.OUT_MARGIN
    out = graded & ~np.isnan(bx) & ((bx < -mx) | (bx > 1 + mx) | (by < -my) | (by > 1 + my))
    walking = graded & (crowd < 1.0)
    before = np.zeros(len(ball), dtype=bool)
    for e in tagging:
        if e["code"] == "SET PIECES":
            before[max(e["start_frame"] - 4 * FPS, 0):e["start_frame"]] = True
    return out, walking, before & graded


def prior(ball, tagging, graded, crowd):
    """Fracción de cuadros sin balón en tres contextos, y log-cocientes contra el de jugadores corriendo."""
    _, walking, before = sure_dead(ball, tagging, graded, crowd)
    running = graded & (crowd >= v1.WALK_SPEED)
    gone = np.isnan(ball[:, 0])
    p = {name: gone[m].mean() for name, m in (("caminando", walking), ("pre-reinicio", before),
                                              ("corriendo", running))}
    print("sin balón: caminando {:.1%}, 4 s antes de un SET PIECES {:.1%}, corriendo {:.1%}".format(
        p["caminando"], p["pre-reinicio"], p["corriendo"]))
    print("log(P(caminando)/P(corriendo)) = {:+.2f}   log(P(pre-reinicio)/P(corriendo)) = {:+.2f}".format(
        np.log(p["caminando"] / p["corriendo"]), np.log(p["pre-reinicio"] / p["corriendo"])))


def restarts(labels, ball, tagging, pitch):
    """(saques detectados, con reinicio en las etiquetas, mediana error s, mediana |error| s, a <= 1 s)."""
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
    return found, len(errors), np.median(errors), np.median(np.abs(errors)), int(near)


def main():
    ball, players, team, pitch = load()
    tagging = load_tagging()
    graded = v2.graded_mask(len(ball))
    crowd = np.nan_to_num(v1.crowd_speed(players), nan=0.0)
    clear = v1.control(ball, players, team, pitch)
    seen = ~np.isnan(ball[:, 0])
    full = {m: run(m)(ball, players, team, pitch, tagging) for m in GRID}
    assert np.array_equal(full[BASE], v5.run(ball, players, team, pitch, tagging)), "el truco aditivo no reproduce la v5"

    print("== Referencia a priori (solo informativa)")
    prior(ball, tagging, graded, crowd)

    print("\n== Guardas sobre el partido completo")
    masks = sure_dead(ball, tagging, graded, crowd)
    guard = {}
    print("{:<8} {:>11} {:>12} {:>14} | {:>10} {:>6} {:>10} {:>11}".format(
        "falta", "D fuera", "D caminando", "D pre-reinicio", "posesiones", "< 2 s", "no marcadas", "inventadas"))
    for m in GRID:
        guard[m] = [(full[m] == "D")[k].mean() for k in masks]
        print("{:<8g} {:>11.1%} {:>12.1%} {:>14.1%} | {:>10} {:>6} {:>10} {:>11}".format(
            m, *guard[m], *cambios.audit(full[m], clear, seen)))

    print("\n== Detector de reinicios")
    hits = {}
    print("{:<8} {:>9} {:>11} {:>11} {:>11} {:>8}".format(
        "falta", "saques", "con reinicio", "mediana (s)", "|mediana|", "<= 1 s"))
    for m in GRID:
        found, with_restart, med, abs_med, hits[m] = restarts(full[m], ball, tagging, pitch)
        print("{:<8g} {:>9} {:>11} {:>+11.1f} {:>11.1f} {:>8}".format(
            m, found, with_restart, med, abs_med, hits[m]))

    print("\n== Prueba de huecos")
    truth = np.where(clear[:, v1.H], 0, np.where(clear[:, v1.A], 1, -1))
    lengths = np.array([e - s for s, e in huecos.live_gaps(ball, crowd, graded)])
    versions = {"falta {:g}".format(m): run(m) for m in GRID}
    results = huecos.simulate(versions, ball, players, team, pitch, tagging, graded, crowd, clear,
                              truth, lengths)
    huecos.table(results, "falta 0.3")
    ref = results["falta 0.3"]
    paired = {m: results["falta {:g}".format(m)] - ref for m in GRID}
    se = {m: p.std(0) / np.sqrt(huecos.REPLICAS) for m, p in paired.items()}
    print("\nmuerto falso pareado contra 0.3 (por réplica)")
    print("{:<8} {:>10} {:>8} {:>16}".format("falta", "media", "std", "error estándar"))
    for m in GRID:
        p = paired[m][:, 1]
        print("{:<8g} {:>+10.1%} {:>8.1%} {:>16.2%}".format(m, p.mean(), p.std(), se[m][1]))

    print("\n== Veredicto")
    better = []
    for m in GRID[1:]:
        g1 = all(a - b >= -0.01 for a, b in zip(guard[m], guard[BASE]))
        g2 = paired[m][:, 0].mean() >= -se[m][0]
        g3 = hits[m] >= hits[BASE] - 1
        fp, fp_se = paired[m][:, 1].mean(), se[m][1]
        ok = g1 and g2 and g3 and fp <= -0.02 and -fp > 2 * fp_se
        print("falta {:<6g} guardas: seguro {} equipo {} saques {}; muerto falso {:+.1%} ({:+.1f} EE) -> {}".format(
            m, "ok" if g1 else "NO", "ok" if g2 else "NO", "ok" if g3 else "NO", fp, fp / fp_se,
            "MEJORA" if ok else "no mejora"))
        if ok:
            better.append(m)
    if not better:
        print("ninguno mejora: se queda 0.3 y no hay versión nueva")
        return
    best = min(better, key=lambda m: results["falta {:g}".format(m)][:, 1].mean())
    close = []
    for m in better:
        d = results["falta {:g}".format(m)][:, 1] - results["falta {:g}".format(best)][:, 1]
        if d.mean() <= d.std() / np.sqrt(huecos.REPLICAS):  # el mejor (d = 0) siempre entra
            close.append(m)
    print("mejor muerto falso: {:g}; elegido por la regla (el más alto a < 1 EE del mejor): {:g}".format(
        best, max(close)))


if __name__ == "__main__":
    main()
