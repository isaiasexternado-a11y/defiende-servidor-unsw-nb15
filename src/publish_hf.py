"""Publica v1 o v2 en un repositorio de modelos de Hugging Face Hub.

    hf auth login                                   # una vez, con token de escritura
    python -m src.publish_hf v1 --repo USUARIO/defiende-servidor-unsw-nb15
    python -m src.publish_hf v2 --repo USUARIO/defiende-servidor-unsw-nb15

Cada revisión lleva el model.joblib exacto del run de MLflow (se verifica su SHA-256),
config.json con el umbral elegido, requirements.txt y la model card. El SHA completo
del commit de HF y el tag se guardan en evidencias/seleccion.json.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version as pkg_version
from pathlib import Path

import mlflow
from huggingface_hub import HfApi

from src.train import TRACKING_URI, file_sha256

RUNTIME_PACKAGES = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "huggingface_hub"]


def fmt(m: dict) -> str:
    cm = m["confusion_matrix"]
    return (f"| {m['recall_attack']:.4f} | {m['precision_attack']:.4f} | {m['fpr']:.4f} | "
            f"{m['macro_f1']:.4f} | {m['accuracy']:.4f} | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |")


def model_card(version: str, v: dict, selection: dict, audit: dict, repo: str, code_url: str) -> str:
    other = selection["versions"]["v1" if version == "v2" else "v2"]
    sizes = audit["split"]["sizes"]
    rows = [f"| {version} (esta) {fmt(v['val_metrics'])}"]
    if version == "v2":
        rows.append(f"| v1 {fmt(other['val_metrics'])}")
    return f"""---
license: other  # datos UNSW-NB15: uso académico
library_name: sklearn
tags: [tabular-classification, intrusion-detection, unsw-nb15, mlops-course]
datasets: [UNSW-NB15]
---

# Defiende el servidor · {version} · {v['config_id']}

Clasificador binario de conexiones de red (UNSW-NB15): **0 = normal, 1 = ataque**.
Proyecto académico del curso de MLOps (Universidad Externado de Colombia, 2026-II).

## Propósito

Practicar el versionado trazable de código, datos, experimentos y modelos. No es un IDS
para producción.

## Qué contiene esta revisión

| Archivo | Contenido |
|---|---|
| `model.joblib` | `sklearn.pipeline.Pipeline` completo: imputación + escalado + one-hot de `proto/service/state` + {v['family']} |
| `config.json` | Umbral, regla de decisión, variables de entrada, SHA-256 del modelo y referencias |
| `requirements.txt` | Versiones exactas para cargar el modelo |

- **Familia:** {v['family']}
- **Umbral:** {v['threshold']} — regla: `ataque si score_attack >= {v['threshold']}`
- **SHA-256 de `model.joblib`:** `{v['model_sha256']}`
- **Razón de esta versión:** {v['reason']}
{"- **Cambio frente a v1:** de regresión logística (umbral 0.5) a HistGradientBoosting, con umbral elegido en validación para limitar falsas alarmas a FPR <= 5 %." if version == "v2" else ""}

## Trazabilidad

| Pieza | Referencia |
|---|---|
| Código (commit Git del entrenamiento) | [`{v['git_commit_trained']}`]({code_url}/tree/{v['git_commit_trained']}) |
| Datos preparados (DVC `data/prepared.dvc`, md5) | `{v['dvc_prepared_md5']}` |
| Experimento (MLflow run ID) | `{v['mlflow_run_id']}` |
| Selección | `evidencias/seleccion.json` en el repo de código |

## Datos

UNSW-NB15, partición oficial de entrenamiento (175.341 filas). Validación separada del mismo
archivo con `StratifiedGroupKFold` (semilla {audit['split']['seed']}): los grupos de entradas idénticas no se
reparten entre train ({sizes['train']}) y val ({sizes['val']}). Se excluyeron {sizes['excluded_contradictory']} filas de grupos con
etiquetas contradictorias. Variables excluidas: `id` (orden del archivo, filtra la clase),
`attack_cat` (es la respuesta) y `label`.

## Métricas de validación (ataque = clase positiva)

| Versión | Recall ataque | Precision ataque | FPR | Macro-F1 | Accuracy | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Denominadores cero: la métrica vale 0 y se reporta en `zero_division` (no ocurrió aquí).

## Uso

```python
import json, joblib, pandas as pd
from huggingface_hub import hf_hub_download

REV = "<SHA completo de esta revisión>"
path = lambda f: hf_hub_download("{repo}", f, revision=REV)
model = joblib.load(path("model.joblib"))
cfg = json.load(open(path("config.json")))

X = pd.read_csv("conexiones.csv")[cfg["features"]]
score = model.predict_proba(X)[:, 1]
y_pred = (score >= cfg["threshold"]).astype(int)
```

O desde el repositorio de código: `python -m src.predict --repo {repo} --revision <SHA> --input ... --out ...`.
`joblib.load` ejecuta pickle: carga solo revisiones de las que conozcas el SHA.

## Limitaciones

- Tráfico generado en laboratorio (IXIA PerfectStorm, 2015): no demuestra eficacia en una red actual.
- Alta proporción de duplicados (42 % de las filas de entrenamiento); el test oficial comparte entradas con train/val en ~10 % de sus filas.
- Clasificación binaria: no distingue la familia del ataque.
- Probabilidades no calibradas; el umbral se eligió para la distribución de validación.

## Referencia

Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS 2015.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", choices=["v1", "v2"])
    parser.add_argument("--repo", required=True, help="USUARIO/nombre del repo de modelos")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--evidence", type=Path, default=Path("evidencias"))
    args = parser.parse_args()

    selection_path = args.evidence / "seleccion.json"
    selection = json.loads(selection_path.read_text())
    audit = json.loads((args.evidence / "auditoria.json").read_text())
    v = selection["versions"][args.version]
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    code_url = remote.removesuffix(".git")

    mlflow.set_tracking_uri(TRACKING_URI)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle = Path(mlflow.MlflowClient().download_artifacts(v["mlflow_run_id"], "bundle", str(tmp)))
        if file_sha256(bundle / "model.joblib") != v["model_sha256"]:
            sys.exit("✗ el model.joblib del run no coincide con el SHA-256 registrado")

        out = tmp / "hf"
        out.mkdir()
        (out / "model.joblib").write_bytes((bundle / "model.joblib").read_bytes())
        run_cfg = json.loads((bundle / "config.json").read_text())
        cfg = {**run_cfg, "version": args.version, "threshold": v["threshold"],
               "threshold_rule": v["threshold_rule"], "dvc_prepared_md5": v["dvc_prepared_md5"],
               "val_metrics": v["val_metrics"], "code_repo": code_url}
        (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        (out / "requirements.txt").write_text(
            "".join(f"{p}=={pkg_version(p)}\n" for p in RUNTIME_PACKAGES) + "# Python 3.14\n")
        (out / "README.md").write_text(model_card(args.version, v, selection, audit, args.repo, code_url))

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=args.repo, folder_path=str(out),
                                   commit_message=f"{args.version}: {v['config_id']} (run {v['mlflow_run_id']})")
        api.create_tag(args.repo, tag=args.version, revision=commit.oid, exist_ok=False)

    v["huggingface"] = {"repo": args.repo, "revision_sha": commit.oid, "tag": args.version,
                        "url": f"https://huggingface.co/{args.repo}/tree/{commit.oid}"}
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    print(f"✓ {args.version} publicada: {args.repo}@{commit.oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
