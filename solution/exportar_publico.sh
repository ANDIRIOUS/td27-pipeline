#!/bin/sh
# Arma el repo público de la entrega (SUBMISSION.md, entregable 2) en una carpeta nueva, SIN
# historial: copia solo la lista explícita de abajo, verifica que no se cuele nada del partido y
# hace un único commit. El historial privado nunca sale de aquí.
#
# Uso, desde cualquier lugar del repo privado:
#   sh solution/exportar_publico.sh /ruta/destino            # destino inexistente o vacío
#   sh solution/exportar_publico.sh --verificar /ruta/clon   # solo la verificación, p. ej. sobre un clon
#
# Por qué una lista y no "todo menos X": publicar datos del partido descalifica y la licencia no
# permite redistribuirlos. Con una lista, un archivo nuevo en el repo privado no sale hasta que
# alguien lo agrega aquí a propósito.
#
# El README.md de la raíz del privado es el del organizador y no sale. El README.md del público se
# copia de docs/README-publico.md, que no se exporta con su nombre original.
#
# Después (manual, lo hace el equipo; este script no toca GitHub):
#   1. Clonar el destino en otra carpeta y, dentro del clon, regenerar la entrega con los datos
#      montados en solo lectura (sin run.sh, que además puntúa la ventana pública):
#        docker build -f solution/Dockerfile -t tiki-data-export-test:1 .
#        docker run --rm -v "$PWD":/r -v /ruta/a/data:/r/data:ro tiki-data-export-test:1 sh -c \
#          'python solution/v7.py --out evaluation/results/Tiki-Data.csv &&
#           python evaluation/validate_submission.py evaluation/results/Tiki-Data.csv'
#        shasum -a 256 evaluation/results/Tiki-Data.csv
#      y comparar con el sha256 de solution/README.md. Con :ro, si falta data/cache/ falla en vez
#      de escribirlo; sin caché, quitar el :ro.
#   2. Crear en GitHub un repo público vacío (sin README ni licencia), y desde el destino:
#        git remote add origin git@github.com:<cuenta>/<repo>.git && git push -u origin main
#   3. Poner la URL en el formulario de entrega.
set -eu

# Rutas relativas a la raíz del repo, iguales en el público para que run.sh y Docker funcionen igual.
ARCHIVOS='
.dockerignore
.gitattributes
.gitignore
LICENSE
compose.yaml
games.py
starter/baseline.py
starter/load_smart_tagging.py
solution/Dockerfile
solution/Dockerfile.dev
solution/README.md
solution/requirements.txt
solution/requirements-dev.txt
solution/run.sh
solution/exportar_publico.sh
solution/atd.py
solution/v1.py
solution/v2.py
solution/v3.py
solution/v4.py
solution/v5.py
solution/v7.py
solution/reinicios.py
solution/muertos.py
docs/figuras/fig1-metodo.png
docs/figuras/fig2-ablaciones.png
docs/figuras/figuras.py
solution/arrastre.py
solution/diagnose.py
solution/fronteras.py
solution/huecos.py
solution/cambios.py
solution/broadcast_sync.py
evaluation/possession_metrics.py
evaluation/run_evaluation.py
evaluation/validate_submission.py
evaluation/results/README.md
evaluation/gt/BAR-ATH_possession_gt_public.csv
evaluation/gt/BAR-ATH_possession_gt_public_meta.json
evaluation/results/baseline_raw.csv
evaluation/results/baseline_smooth25.csv
evaluation/results/baseline_smooth50.csv
evaluation/results/baseline_smooth125.csv
docs/Tiki-Data.md
'

# Material del organizador bajo MIT (LICENSE cubre evaluation/gt/ explícitamente). Son los únicos
# .csv y los únicos archivos de más de 1 MB permitidos: run_evaluation.py necesita la ventana
# pública y diagnose.py usa baseline_smooth50 como referencia.
PERMITIDOS_GRANDES='
evaluation/gt/BAR-ATH_possession_gt_public.csv
evaluation/results/baseline_raw.csv
evaluation/results/baseline_smooth25.csv
evaluation/results/baseline_smooth50.csv
evaluation/results/baseline_smooth125.csv
'

falla() { echo "ERROR: $*" >&2; exit 1; }

# Aborta si en $1 aparece algo que no debe publicarse. Ignora .git/.
revisar_uno() {  # $1 carpeta, $2 ruta relativa; imprime el motivo si no puede publicarse
    grande=no
    case "$PERMITIDOS_GRANDES" in *"
$2
"*) grande=si ;; esac
    case "$2" in
        data/*) echo "$2  (ruta bajo data/)"; return ;;
        *.mp4|*.xml|*.atd|*.npy|*.npz|*.parquet|*.pkl) echo "$2  (extensión prohibida)"; return ;;
        *.csv) [ "$grande" = si ] || { echo "$2  (.csv fuera de los permitidos)"; return; } ;;
    esac
    if [ "$grande" = no ] && [ -n "$(find "$1/$2" -size +1024k)" ]; then
        echo "$2  (más de 1 MB)"
    fi
}

verificar() {
    malos=$( (cd "$1" && find . -path ./.git -prune -o -type f -print) | sed 's|^\./||' |
        while read -r f; do revisar_uno "$1" "$f"; done)
    [ -z "$malos" ] || falla "esto no puede publicarse, no se hizo nada más:
$malos"
    echo "verificación de seguridad: OK"
}

if [ "${1:-}" = "--verificar" ]; then
    [ -d "${2:-}" ] || falla "uso: $0 --verificar <carpeta>"
    verificar "$2"
    exit 0
fi

DEST=${1:-}
[ -n "$DEST" ] || falla "uso: $0 <carpeta-destino>"
if [ -e "$DEST" ]; then
    [ -d "$DEST" ] && [ -z "$(ls -A "$DEST")" ] || falla "$DEST existe y no es una carpeta vacía"
fi
ORIGEN=$(cd "$(dirname "$0")/.." && pwd)

# Primero comprueba la lista completa, para no dejar un destino a medias.
for f in $ARCHIVOS docs/README-publico.md; do
    [ -f "$ORIGEN/$f" ] || falla "falta $f en $ORIGEN"
done

mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
LISTA=$(mktemp)
trap 'rm -f "$LISTA"' EXIT
for f in $ARCHIVOS; do
    mkdir -p "$DEST/$(dirname "$f")"
    cp -p "$ORIGEN/$f" "$DEST/$f"
    echo "$f" >> "$LISTA"
done
cp -p "$ORIGEN/docs/README-publico.md" "$DEST/README.md"
echo README.md >> "$LISTA"
sort -o "$LISTA" "$LISTA"

verificar "$DEST"

# -f porque .gitignore ignora *.pdf (por si el documento llega en PDF); la lista ya es explícita.
# El autor del commit es la identidad git de quien corre el script.
cd "$DEST"
git init -q -b main
xargs git add -f -- < "$LISTA"
git commit -q -m "Tiki-Data: entrega del Initial Challenge de More Than a Hack 2027"

# Chequeo final: lo commiteado es exactamente la lista, ni más ni menos.
git ls-files | sort | diff "$LISTA" - || falla "lo commiteado no coincide con la lista"
echo "exportados ($(wc -l < "$LISTA" | tr -d ' ') archivos, commit $(git rev-parse --short HEAD)):"
sed 's/^/  /' "$LISTA"
