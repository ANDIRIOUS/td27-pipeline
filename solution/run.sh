#!/bin/sh
# Pipeline completo, igual dentro y fuera de Docker:
#   ATD -> entrega -> validación -> hash -> puntaje en la ventana pública -> diagnóstico sin etiquetas
# TEAM fija el nombre del archivo (SUBMISSION.md pide <team_name>.csv); VERSION elige la versión;
# por defecto es la congelada, v7.
set -e
cd "$(dirname "$0")/.."
VERSION=${VERSION:-v7}
TEAM=${TEAM:-$VERSION}
OUT=evaluation/results/$TEAM.csv
python solution/$VERSION.py --out "$OUT"
python evaluation/validate_submission.py "$OUT"
python -c "import hashlib,sys; print('sha256', hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(), sys.argv[1])" "$OUT"
python evaluation/run_evaluation.py --bootstrap 1000 --no-detail
python solution/diagnose.py evaluation/results/baseline_smooth50.csv "$OUT"
