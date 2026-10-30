#!/usr/bin/env bash
# Prueba de recuperación en carpeta y entorno nuevos (sin cachés de la ejecución original).
#
#   bash scripts/recuperacion.sh [CARPETA_NUEVA] [REVISION_GIT]
#
# 1) clona el código en la revisión elegida e instala dependencias fijadas en un venv nuevo
# 2) recupera con DVC los datos originales y preparados desde el remoto por defecto y verifica huellas
# 3-4) descarga v1 y v2 de Hugging Face por SHA (caché nueva) y predice las mismas entradas fijas
# 5) compara con las salidas de referencia
set -euo pipefail

WORK="${1:-$HOME/recuperacion-defiende-servidor}"
REV="${2:-main}"
REPO_URL="https://github.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15.git"
PYTHON="${PYTHON:-python3.14}"

rm -rf "$WORK" && mkdir -p "$WORK"
export HF_HOME="$WORK/hf-cache"          # caché de Hugging Face nueva
export XDG_CACHE_HOME="$WORK/xdg-cache"  # evita reutilizar cachés globales (pip, dvc site cache)
export PIP_NO_CACHE_DIR=1

echo "## $(date -Iseconds) · máquina: $(uname -srm) · python: $($PYTHON --version)"
echo "## 1. código en revisión $REV"
git clone --quiet "$REPO_URL" "$WORK/repo"
cd "$WORK/repo"
git checkout --quiet "$REV"
echo "commit: $(git rev-parse HEAD)"
$PYTHON -m venv .venv
.venv/bin/pip install --quiet -r requirements-lock.txt
.venv/bin/pip freeze | grep -iE '^(scikit-learn|numpy|pandas|dvc|huggingface.hub|mlflow)=='

echo "## 2. datos con DVC desde el remoto (caché vacía: $(ls .dvc/cache 2>/dev/null | wc -l) entradas)"
if [ -n "${DAGSHUB_USER:-}" ]; then   # para leer de DagsHub (DVC_REMOTE=origin), que exige credenciales
  .venv/bin/dvc remote modify --local origin auth basic
  .venv/bin/dvc remote modify --local origin user "$DAGSHUB_USER"
  .venv/bin/dvc remote modify --local origin password "$DAGSHUB_TOKEN"
fi
.venv/bin/dvc pull ${DVC_REMOTE:+-r "$DVC_REMOTE"}
.venv/bin/dvc status
echo "SHA-256 originales contra MANIFEST.json:"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from src.data.download import sha256
m = json.loads(Path("data/raw/MANIFEST.json").read_text())
a = json.loads(Path("evidencias/auditoria.json").read_text())
ok = True
for name in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
    good = sha256(Path("data/raw") / name) == m[name]["sha256"]; ok &= good
    print(f"  {'✓' if good else '✗'} raw/{name}")
for name, h in a["split"]["file_sha256"].items():
    good = sha256(Path("data/prepared") / name) == h; ok &= good
    print(f"  {'✓' if good else '✗'} prepared/{name}")
raise SystemExit(0 if ok else 1)
PY

echo "## 3-4. modelos desde Hugging Face por SHA y predicción sobre las entradas fijas"
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import pandas as pd
entrada = pd.read_csv("evidencias/recuperacion/entrada.csv")
val = pd.read_csv("data/prepared/val.csv").set_index("row_id")
same = val.loc[entrada.row_id].reset_index()[entrada.columns].equals(entrada)
print(f"  {'✓' if same else '✗'} entrada.csv coincide con las filas de val recuperadas por DVC")
raise SystemExit(0 if same else 1)
PY
OUT="$WORK/salidas"
mkdir -p "$OUT"
for v in v1 v2; do
  SHA=$(.venv/bin/python -c "import json;print(json.load(open('evidencias/seleccion.json'))['versions']['$v']['huggingface']['revision_sha'])")
  echo "$v → revisión HF $SHA"
  .venv/bin/python -m src.predict --repo Isxcesp/defiende-servidor-unsw-nb15 --revision "$SHA" \
    --input evidencias/recuperacion/entrada.csv --out "$OUT/despues_$v.csv"
done
sha256sum "$OUT"/despues_*.csv evidencias/recuperacion/antes_*.csv evidencias/recuperacion/entrada.csv
find "$HF_HOME" -name model.joblib -exec sha256sum {} \;

echo "## 5. comparación con las salidas de referencia"
cp evidencias/recuperacion/antes_*.csv "$OUT/"
.venv/bin/python -m src.recovery check "$OUT"
echo "## fin $(date -Iseconds)"
