# Entrega · MLOps — Defiende el servidor — Grupo A — Isaias Cespedes Novoa

## Integrantes y repositorio

- **Integrante:** Isaias Cespedes Novoa · Grupo A · entrega individual
- **Código (GitHub):** https://github.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15
- **Commit final de la entrega:** el que se indica en el mensaje de entrega (HEAD de `main`); este archivo se actualizó por última vez en ese commit.
- **Modelos (Hugging Face):** https://huggingface.co/Isxcesp/defiende-servidor-unsw-nb15
- **Almacenamiento DVC (DagsHub):** https://dagshub.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15 · remoto `origin` = `https://dagshub.com/isaiasexternado-a11y/defiende-servidor-unsw-nb15.dvc`

## Versiones

| | v1 | v2 |
|---|---|---|
| Configuración | `e1_logreg_base` (regresión logística) | `e4_hist_gradient_boosting` |
| Tag Git | `v1` → `3fd054d7be2c6130d5f9696d859652b24889e647` | `v2` → `94131ed859c9125b88a993dc1f828c552e40ce5a` (selección congelada) |
| Commit que entrenó el modelo | `3fd054d7be2c6130d5f9696d859652b24889e647` | `3fd054d7be2c6130d5f9696d859652b24889e647` |
| Datos DVC (`data/prepared.dvc`) | `3f6784bc3efe6154e32ef7baca8247a9.dir` | `3f6784bc3efe6154e32ef7baca8247a9.dir` |
| Datos DVC (`data/raw.dvc`) | `02b04edb2e25a537a7aa1cbf75c15435.dir` | `02b04edb2e25a537a7aa1cbf75c15435.dir` |
| Run ID MLflow | `9682854f9eed4516b187ad6d713e9833` | `216748533c514fc5b4625ceed0aeee59` |
| Revisión Hugging Face | `98c0228f88069b0a48ff041388bd1a4eb76a309b` (tag `v1`) | `9da5454fe51b13cbb471a6b266d25ad681861504` (tag `v2`) |
| SHA-256 `model.joblib` | `4a9e3d854a328082157a4f136a3ea1a2943d1f025c0144cd35a2300848907658` | `4e173642c483816c4f073fe6e003394f0ca4ed40069ef864e97cc06d978b6ab2` |
| Umbral | 0.5 | 0.65 |
| Val: recall / precision / FPR / macro-F1 | 0.989 / 0.923 / 0.177 / 0.923 | 0.952 / 0.976 / 0.050 / 0.945 |

Las dos versiones se entrenaron en el mismo commit (el de los cuatro experimentos). v2 difiere en el modelo y el umbral; la decisión quedó registrada en `evidencias/seleccion.json` y congelada en el commit `94131ed` antes de tocar el test. La model card de v2 con el resultado de test está en la revisión `cfd5b24f8496c172b2a4e641bb600a97700313e4`, que solo cambia `README.md` (ver `evidencias/model-card-test.json`).

## Ubicación de las evidencias

| Evidencia | Archivo |
|---|---|
| Auditoría de datos y splits | `evidencias/auditoria.json` |
| Tabla de experimentos | `evidencias/experimentos.csv` (conclusiones en `evidencias/conclusiones.json`) |
| MLflow exportado (4 runs) | `evidencias/mlflow/<run_id>/` (`run.json`, `config.json`, `val_metrics.json`, `val_scores.csv`) |
| Selección y umbral | `evidencias/seleccion.json`, barrido de umbral en `evidencias/umbral_v2.csv` |
| Métricas de test | `evidencias/metricas-test.json` |
| Predicciones de test | `evidencias/predicciones-test.csv` (82.332 filas: `row_id, y_true, score_attack, y_pred`) |
| Recuperación | `evidencias/recuperacion/` (entradas, salidas antes) y `evidencias/recuperacion/entorno-nuevo/` (log, comandos, salidas después, comparación) |
| Informe | `informe/informe.pdf` (también adjunto al mensaje de entrega); se genera con `python informe/build.py` a partir de los archivos de evidencia |

`row_id` = `<archivo>:<fila>` con fila 0-based sin encabezado; `train:` es `UNSW_NB15_training-set.csv` y `test:` es `UNSW_NB15_testing-set.csv`.

## Acceso

- GitHub y Hugging Face son públicos: `git clone` y la descarga de modelos no necesitan credenciales.
- DagsHub también es público, pero su almacenamiento DVC exige autenticación incluso en repos públicos (probado: `dvc pull` anónimo responde 401). Se solicitó al docente su usuario de DagsHub para invitarlo como colaborador con rol de lectura; con esa invitación y un token propio basta configurar las credenciales en `.dvc/config.local` (ver README, sección 2).
- No hay tokens ni contraseñas en el repositorio; las credenciales de DVC se guardan solo en `.dvc/config.local` (ignorado por Git).

## Comandos

Instalación, recuperación, entrenamiento y predicción: ver [`README.md`](README.md). Resumen:

```bash
pip install -r requirements-lock.txt
dvc pull
python -m src.data.prepare
python -m src.train configs/experiments/e4_hist_gradient_boosting.yaml
python -m src.predict --repo Isxcesp/defiende-servidor-unsw-nb15 --revision 9da5454fe51b13cbb471a6b266d25ad681861504 --input evidencias/recuperacion/entrada.csv --out salidas/pred_v2.csv
bash scripts/recuperacion.sh "$HOME/recuperacion-defiende-servidor" main
```

## Tiempo y recursos medidos

Equipo: Windows + WSL2 (Linux 6.18), 12 núcleos lógicos, ~3 GB de RAM asignados a WSL, sin GPU, Python 3.14.4.

| Paso | Tiempo |
|---|---|
| `src.data.prepare` (auditoría + splits) | 25 s |
| Entrenamiento e1 / e2 / e3 / e4 | 23 s / 19 s / 27 s / 16 s |
| `dvc push` inicial (720 MB) | 6 min |
| Prueba de recuperación completa (clon, venv, `dvc pull`, 2 modelos) | 4 min |

Tamaño de los modelos: v1 11 KB, v2 1,1 MB (el bosque de e3 pesa 58 MB y no se publicó).

## Limitaciones conocidas

- UNSW-NB15 es tráfico de laboratorio de 2015: no demuestra eficacia sobre una red actual.
- En test el FPR de v2 sube a 0.181 (0.050 en validación): el umbral elegido en validación no se transfiere a la distribución del test oficial. Se conserva sin reajustar.
- 42 % de las filas de entrenamiento son duplicados por entradas y ~10 % de las filas de test tienen entradas ya vistas en train/val.
- Se excluyeron de train/val 940 filas de grupos con etiquetas contradictorias.
- `data/raw` incluye también los CSV completos de UNSW (no usados por el modelo); por eso el remoto DVC pesa ~720 MB.
- MLflow se usó en local (SQLite); la evidencia consultable es la exportación en `evidencias/mlflow/`.

## Uso de IA, código externo y contribución

- **Contribución:** trabajo individual de Isaias Cespedes Novoa.
- **Asistente de IA:** Claude Code (modelo Claude Opus 5.5, Anthropic), usado desde VS Code para escribir la mayor parte del código de `src/` y `scripts/`, las configuraciones, las model cards y los borradores de README y `entrega.md`, y para ejecutar la auditoría, los experimentos, la publicación y la prueba de recuperación. Las decisiones de diseño (política de duplicados y contradicciones, familias de modelos, política de umbral FPR ≤ 5 %, publicación pública) se discutieron y aprobaron durante la sesión.
- **Qué se verificó:** huellas SHA-256 de los datos contra el anexo del taller; determinismo del split (dos ejecuciones idénticas); que ningún grupo de entradas quede repartido entre train y val; que los modelos descargados de Hugging Face por SHA reproduzcan exactamente los scores de los runs de MLflow; recálculo de métricas de test desde `predicciones-test.csv`; recuperación completa en entorno nuevo.
- **Código externo:** no se copió código de terceros. Se usan librerías de código abierto (scikit-learn BSD-3, pandas BSD-3, MLflow Apache-2.0, DVC Apache-2.0, huggingface_hub Apache-2.0).
- **Datos:** UNSW-NB15, uso académico; citas en `docs/DATASET.md`.
