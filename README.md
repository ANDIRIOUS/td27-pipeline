# Tiki-Data: posesión cuadro a cuadro con evidencia y Viterbi

> Team Tiki-Data's submission to the More Than a Hack 2027 Initial Challenge: one `H`/`A`/`D`
> label per frame of the tactical video, built only from automatic tracking and Smart Tagging.
> To reproduce it, put the challenge data in `data/` and run `docker compose run --rm pipeline`;
> the output's SHA-256 must match the one below. Documentation is in Spanish.

## Qué es

Es la entrega del equipo Tiki-Data al Initial Challenge de More Than a Hack 2027: una etiqueta
(`H`, `A` o `D`) para cada uno de los 199,319 cuadros del video táctico del Barça vs Athletic.

El modelo no ve el video ni usa etiquetas humanas. Lee el tracking automático (ATD) y el Smart
Tagging, y convierte cada cuadro en pistas a favor del Barça, del Athletic o del balón muerto.
Después, un Viterbi de tres estados elige la secuencia de toda la mitad que mejor explica esas
pistas, con un costo por cada cambio. Todo es numpy y pandas, corre en un contenedor fijado y
regenera el mismo archivo byte a byte. La versión entregada es la **v7**.

## Cómo reproducir

**1. Datos.** Los datos del partido no vienen en este repo. Se descargan al inscribirse en el
reto. Crea la carpeta `data/` en la raíz y descomprime ahí el paquete, con los nombres con los que
llega (`data/tactical/` y `data/broadcast/`), como indica el `DATA.md` del repo oficial:
https://github.com/metrica-sports/more-than-a-hack-2027

La entrega solo lee el ATD y el Smart Tagging de `data/tactical/`. `data/broadcast/` solo lo usa el
estudio `solution/broadcast_sync.py`. Si tus datos están en otra ruta, apunta a ellos con
`TIKI_DATA=/ruta/a/data` (Docker Compose) o `MTAH_DATA_DIR=/ruta/a/data` (sin Docker).

**2. Correr.** Con Docker Compose (recomendado):

```bash
docker compose run --rm pipeline
```

Sin Docker (Python 3.10 o más reciente):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r solution/requirements.txt
TEAM=Tiki-Data sh solution/run.sh
```

`solution/run.sh` genera la entrega, la valida, imprime su hash, la puntúa en la ventana pública
de 5 minutos y corre un diagnóstico sin etiquetas sobre el partido completo. La primera corrida
parsea el ATD y lo guarda en `data/cache/`.

**3. Resultado.** Sale `evaluation/results/Tiki-Data.csv`. Su huella SHA-256 debe ser:

```
aff8290262161f5fb4bb96897e320eacebab03e0ec43420173af2052ef235ab9
```

Se verificó en Docker arm64 y x86_64, desde un caché vacío.

## Dónde está cada cosa

| ruta | qué hay |
|---|---|
| `docs/Tiki-Data.md` | documento técnico: enfoque, decisiones y resultados |
| `solution/README.md` | detalle de cada versión, sus hashes y los estudios hechos |
| `solution/` | código del modelo, scripts de estudio y entorno fijado (`Dockerfile`, `requirements.txt`) |
| `solution/fronteras.py`, `solution/arrastre.py` | estudios de análisis; no forman parte de la entrega |
| `docs/figuras/` | figuras del documento técnico y el script que las genera |
| `evaluation/` | evaluador oficial, validador de la entrega y ventana pública de 5 minutos |

## Qué no hay aquí

Los datos del partido (videos, tracking, Smart Tagging y homografía). Están licenciados solo para
esta competencia y no se pueden redistribuir. Tampoco el caché `data/cache/` ni el CSV de la
entrega: se regenera con el comando de arriba.

## Créditos y licencia

El evaluador, el starter, los baselines y la ventana pública son de Metrica Sports, bajo licencia
MIT. Ver `LICENSE`, que también aclara que los datos del partido no están cubiertos por ella.
