"""Figuras del documento técnico (docs/Tiki-Data.md).

    docker run --rm -v "$PWD":/r -w /r tiki-data/dev:1 python docs/figuras/figuras.py

No lee `data/` ni ningún cuadro del partido: todas las cifras son agregados ya publicados en
`solution/README.md` y se copian aquí a mano, con la sección de donde salen.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, H, A, D = "#1f2933", "#6b7280", "#2563eb", "#dc2626", "#9ca3af"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, x, y, w, h, title, body, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", weight="bold", color=color, fontsize=8.5)
    ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", color=INK, fontsize=7,
            linespacing=1.35)


def arrow(ax, x0, x1, y):
    ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))


def metodo():
    """Figura 1: esquema del método. Sin datos."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.set(xlim=(0, 12.6), ylim=(0, 3.4)), ax.axis("off")
    w, h, y = 2.85, 2.6, 0.4
    xs = [0.05, 3.25, 6.45, 9.65]
    box(ax, xs[0], y, w, h, "Entradas",
        "ATD automático:\nx, y y velocidad de\njugadores y balón\n\nSmart Tagging:\nSET PIECES y fases")
    box(ax, xs[1], y, w, h, "Pistas por cuadro",
        "nats hacia H, A o D\n\ncontrol claro (3 m / 1 m)\njugadores caminando\nbalón fuera, quieto o\ngolpeado; "
        "reinicios y fases\nforma de los bloques")
    box(ax, xs[2], y, w, h, "Viterbi, 3 estados",
        "secuencia óptima\nde cada mitad\n\ncosto por cambio (nats):\nequipo→equipo: 20\nequipo→D: 20\nD→equipo: 2")
    box(ax, xs[3], y, w, h, "Etiquetas",
        "H / A / D en cada\ncuadro calificado\n\nD explícito antes\ndel saque de cada mitad\n"
        "X solo fuera de\nlo calificado")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w + 0.03, b - 0.03, y + h / 2)
    for i, (c, lab) in enumerate([(H, "H"), (A, "A"), (D, "D")]):
        ax.add_patch(plt.Rectangle((xs[3] + 0.55 + i * 0.5, y + 0.12), 0.42, 0.22, color=c))
        ax.text(xs[3] + 0.76 + i * 0.5, y + 0.23, lab, ha="center", va="center", color="white",
                fontsize=7, weight="bold")
    fig.savefig(os.path.join(OUT, "fig1-metodo.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def ablaciones():
    """Figura 2: prueba de huecos simulados (20 réplicas, semilla 2027)."""
    # solution/README.md, "Prueba de huecos simulados" (v1/v2 con y sin fases) y "v7" (v5 y v7).
    vers = ["v1\nsin fases", "v2\nsin fases", "v1", "v2", "v5", "v7"]
    equipo = [53.7, 59.4, 60.5, 62.3, 62.0, 65.4]
    muerto = [20.4, 19.4, 15.7, 15.1, 16.9, 11.8]
    # solution/README.md, "Peso de las fases (A1)": equipo correcto en huecos por peso.
    pesos = [0, 0.002, 0.004, 0.008, 0.015, 0.03, 0.05, 0.1]
    barrido = [59.4, 59.7, 60.3, 60.5, 61.1, 61.7, 62.0, 62.3]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    x = range(len(vers))
    a1.plot(x, equipo, "o-", color=H, label="equipo correcto")
    a1.plot(x, muerto, "s--", color=MUTED, label="juego vivo marcado D")
    for i, (e, m) in enumerate(zip(equipo, muerto)):
        a1.annotate(f"{e:.1f}", (i, e), (0, 6), textcoords="offset points", ha="center", fontsize=7)
        a1.annotate(f"{m:.1f}", (i, m), (0, -12), textcoords="offset points", ha="center",
                    fontsize=7, color=MUTED)
    a1.set(xticks=list(x), xticklabels=vers, ylim=(0, 75), ylabel="% (control escondido)",
           title="a) Versiones y ablaciones")
    a1.legend(frameon=False, fontsize=7, loc="center right")

    a2.plot(range(len(pesos)), barrido, "o-", color=H)
    a2.plot([6], [62.0], "o", ms=10, mfc="none", mec=INK)
    a2.annotate("elegido: 0.05", (6, 62.0), (-58, 4), textcoords="offset points", fontsize=7)
    a2.set(xticks=range(len(pesos)), xticklabels=[str(p) for p in pesos], ylim=(58, 63),
           xlabel="nats por cuadro de fase", ylabel="equipo correcto (%)", title="b) Peso de las fases")
    a2.tick_params(axis="x", labelsize=6.5, rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2-ablaciones.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    metodo()
    ablaciones()
