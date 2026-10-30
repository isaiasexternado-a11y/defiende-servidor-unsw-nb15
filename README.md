# Defiende el servidor · UNSW-NB15

Proyecto del curso de MLOps (Universidad Externado de Colombia, 2026-II): clasificar conexiones de red como **normales (0) o ataque (1)** con UNSW-NB15, con código, datos, experimentos y modelos versionados y recuperables.

| Pieza | Dónde |
|---|---|
| Código | GitHub `isaiasexternado-a11y/defiende-servidor-unsw-nb15` (tags `v1`, `v2`) |
| Datos (originales y preparados) | DVC → remoto DagsHub `https://dagshub.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15.dvc` |
| Experimentos | MLflow local, exportado en [`evidencias/mlflow/`](evidencias/mlflow) y [`evidencias/experimentos.csv`](evidencias/experimentos.csv) |
| Modelos | Hugging Face [`Isxcesp/defiende-servidor-unsw-nb15`](https://huggingface.co/Isxcesp/defiende-servidor-unsw-nb15) (tags `v1`, `v2`) |

Índice de evidencias y referencias de cada versión: [`entrega.md`](entrega.md). Datos y atribución: [`docs/DATASET.md`](docs/DATASET.md).

## Requisitos

- **Python 3.14** (probado con 3.14.4 en Linux/WSL2) y Git.
- Dependencias exactas en [`requirements-lock.txt`](requirements-lock.txt); las directas en [`requirements.txt`](requirements.txt).
- ~1 GB libre para los datos y 3 GB de RAM bastan; no se necesita GPU.

## 1. Instalación

```bash
git clone https://github.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15.git
cd defiende-servidor-unsw-nb15
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## 2. Recuperar los datos con DVC

```bash
dvc pull            # trae data/raw (archivos recibidos) y data/prepared (splits) desde el remoto por defecto (hf, sin credenciales)
dvc status          # debe decir "Data and pipelines are up to date."
sha256sum data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv
# bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa  (training)
# 734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559  (testing)
```

Hay dos remotos con el mismo contenido (`dvc remote list`):

- `hf` (por defecto, solo lectura): espejo en el dataset público https://huggingface.co/datasets/Isxcesp/defiende-servidor-unsw-nb15-dvc, servido por HTTP. No pide cuenta ni token.
- `origin`: almacenamiento DVC de DagsHub, donde se hace `dvc push`. Exige autenticación incluso con el repo público; las credenciales van solo en local, nunca en Git:

```bash
dvc remote modify --local origin auth basic
dvc remote modify --local origin user <usuario_dagshub>
dvc remote modify --local origin password <token_dagshub>
dvc push -r origin   # el espejo hf se actualiza aparte, subiendo .dvc/cache al dataset con `hf upload`
```

Alternativa sin DVC: `python -m src.data.download --full` descarga los CSV desde la fuente oficial de UNSW y verifica sus SHA-256.

## 3. Preparar datos y ejecutar un experimento

```bash
python -m src.data.prepare      # auditoría → evidencias/auditoria.json; splits → data/prepared/ (determinista, semilla 42)
python -m src.train configs/experiments/e4_hist_gradient_boosting.yaml
```

`src.train` exige un árbol Git sin cambios para ligar el run a un commit (`--allow-dirty` lo omite y marca el commit como `-dirty`). Registra en `mlflow.db` (SQLite local); para verlo: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Las cuatro configuraciones están en [`configs/experiments/`](configs/experiments). Después de entrenar:

```bash
python -m src.export_mlflow     # evidencias/experimentos.csv y evidencias/mlflow/<run_id>/
python -m src.select            # v1/v2 y umbral con validación → evidencias/seleccion.json
```

## 4. Descargar un modelo y predecir

Cada versión se descarga por su **SHA completo** de Hugging Face; `src.predict` verifica el SHA-256 de `model.joblib` antes de cargarlo.

```bash
# v1 · regresión logística · umbral 0.5
python -m src.predict --repo Isxcesp/defiende-servidor-unsw-nb15 \
  --revision 98c0228f88069b0a48ff041388bd1a4eb76a309b \
  --input evidencias/recuperacion/entrada.csv --out salidas/pred_v1.csv

# v2 · HistGradientBoosting · umbral 0.65
python -m src.predict --repo Isxcesp/defiende-servidor-unsw-nb15 \
  --revision 9da5454fe51b13cbb471a6b266d25ad681861504 \
  --input evidencias/recuperacion/entrada.csv --out salidas/pred_v2.csv
```

La entrada es cualquier CSV con `row_id` y las 42 variables de UNSW-NB15 (por ejemplo `data/prepared/val.csv`). La salida tiene `row_id, [y_true,] score_attack, y_pred`, con la regla `y_pred = 1 si score_attack >= umbral`.

## 5. Prueba de recuperación completa

```bash
bash scripts/recuperacion.sh "$HOME/recuperacion-defiende-servidor" main
```

Clona el repo en una carpeta nueva, crea un venv nuevo, hace `dvc pull` con caché vacía, verifica huellas, descarga v1 y v2 por SHA con una caché de Hugging Face nueva y compara las predicciones con [`evidencias/recuperacion/`](evidencias/recuperacion). Ejecución registrada: [`evidencias/recuperacion/entorno-nuevo/log.txt`](evidencias/recuperacion/entorno-nuevo/log.txt).

## Estructura

```
configs/                 data.yaml, selection.yaml, experiments/e1..e4
data/raw.dvc             archivos recibidos (DVC)
data/prepared.dvc        train/val/test preparados y splits.csv (DVC)
src/data/                download.py, prepare.py
src/                     model.py, metrics.py, train.py, export_mlflow.py, select.py,
                         publish_hf.py, evaluate_test.py, card_test.py, predict.py, recovery.py
scripts/recuperacion.sh  prueba de recuperación en entorno nuevo
evidencias/              auditoría, experimentos, selección, test, MLflow exportado, recuperación
```
