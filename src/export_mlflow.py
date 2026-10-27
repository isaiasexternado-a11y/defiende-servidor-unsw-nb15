"""Exporta los runs de MLflow a evidencias/ para revisarlos sin el servidor local.

    python -m src.export_mlflow

Escribe evidencias/experimentos.csv (una fila por configuración) y, por run,
evidencias/mlflow/<run_id>/ con run.json (parámetros, métricas, tags) y los artefactos
livianos del bundle. El modelo (model.joblib) no se copia: se publica en Hugging Face
y aquí queda su SHA-256.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd

from src.train import EXPERIMENT, TRACKING_URI

LIGHT_ARTIFACTS = ["config.json", "val_metrics.json", "val_scores.csv"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("evidencias"))
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time ASC"])

    rows = []
    mlflow_dir = args.out / "mlflow"
    for run in runs:
        info, data = run.info, run.data
        run_dir = mlflow_dir / info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": info.run_id,
            "run_name": info.run_name,
            "experiment": EXPERIMENT,
            "start_time_ms": info.start_time,
            "end_time_ms": info.end_time,
            "params": data.params,
            "metrics": data.metrics,
            "tags": {k: v for k, v in data.tags.items() if not k.startswith("mlflow.log-model")},
        }, indent=2, ensure_ascii=False) + "\n")
        local = Path(client.download_artifacts(info.run_id, "bundle"))
        for name in LIGHT_ARTIFACTS:
            shutil.copy(local / name, run_dir / name)

        t, m, p = data.tags, data.metrics, data.params
        rows.append({
            "config_id": t["config_id"], "hypothesis": t["hypothesis"], "change": t["change"],
            "family": t["family"], "config_path": p["config_path"],
            "params": json.dumps({k: v for k, v in p.items() if k.startswith(("model.", "pre."))}),
            "seed": p["seed"], "threshold": p["threshold"],
            "git_commit": t["git_commit"], "dvc_prepared_md5": t["dvc_prepared_md5"],
            "mlflow_run_id": info.run_id, "model_sha256": t["model_sha256"],
            "recall_attack": m["recall_attack"], "precision_attack": m["precision_attack"],
            "fpr": m["fpr"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "tn": int(m["cm_tn"]), "fp": int(m["cm_fp"]), "fn": int(m["cm_fn"]), "tp": int(m["cm_tp"]),
            "zero_division": t["zero_division"], "fit_seconds": round(m["fit_seconds"], 1),
            "conclusion": "",
        })

    table = pd.DataFrame(rows)
    conclusions = args.out / "conclusiones.json"
    if conclusions.exists():
        table["conclusion"] = table.config_id.map(json.loads(conclusions.read_text())).fillna("")
    table.to_csv(args.out / "experimentos.csv", index=False)
    print(f"✓ {len(table)} runs exportados a {mlflow_dir} y {args.out / 'experimentos.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
