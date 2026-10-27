"""Entrena una configuración sobre train, la evalúa en val y la registra en MLflow.

    python -m src.train configs/experiments/e1_logreg_base.yaml

Registra hipótesis, parámetros, semilla, commit Git, versión DVC de los datos,
métricas de validación, scores por fila de val y el modelo completo (joblib + MLflow).
El test no se toca aquí.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from src.metrics import binary_metrics, predict_label
from src.model import build_pipeline

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "defiende-servidor-unsw-nb15"


def git_commit(allow_dirty: bool) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty and not allow_dirty:
        sys.exit("✗ hay cambios sin commit: el run no quedaría ligado a un commit. Haz commit o usa --allow-dirty.")
    return commit + ("-dirty" if dirty else "")


def dvc_ref(dvc_file: Path) -> str:
    return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = yaml.safe_load(args.data_config.read_text())
    seed = cfg.get("seed", data_cfg["seed"])
    prepared = Path(data_cfg["out_dir"])
    target, categorical = data_cfg["target"], data_cfg["categorical"]

    train = pd.read_csv(prepared / "train.csv")
    val = pd.read_csv(prepared / "val.csv")
    features = [c for c in train.columns if c not in ["row_id", *data_cfg["excluded"]]]

    commit = git_commit(args.allow_dirty)
    pipe = build_pipeline(cfg, features, categorical, seed)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=cfg["id"]) as run:
        start = time.perf_counter()
        pipe.fit(train[features], train[target])
        fit_seconds = time.perf_counter() - start

        scores = pipe.predict_proba(val[features])[:, 1]
        threshold = cfg["threshold"]
        metrics = binary_metrics(val[target], predict_label(scores, threshold))

        mlflow.set_tags({
            "config_id": cfg["id"],
            "family": cfg["family"],
            "hypothesis": cfg["hypothesis"],
            "change": cfg["change"],
            "git_commit": commit,
            "dvc_prepared_md5": dvc_ref(Path("data/prepared.dvc")),
            "dvc_raw_md5": dvc_ref(Path("data/raw.dvc")),
            "val_row_id_sha256": file_sha256(prepared / "val.csv"),
            "positive_class": "1 = ataque",
            "zero_division": ",".join(metrics["zero_division"]) or "none",
        })
        mlflow.log_params({f"model.{k}": v for k, v in cfg["params"].items()})
        mlflow.log_params({f"pre.{k}": v for k, v in cfg["preprocessing"].items()})
        mlflow.log_params({"seed": seed, "threshold": threshold, "config_path": str(args.config),
                           "n_train": len(train), "n_val": len(val), "n_features": len(features)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"cm_{k}": v for k, v in metrics["confusion_matrix"].items()})
        mlflow.log_metric("fit_seconds", fit_seconds)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_path = tmp / "model.joblib"
            joblib.dump(pipe, model_path)
            model_sha = file_sha256(model_path)
            config = {"config_id": cfg["id"], "family": cfg["family"], "threshold": threshold,
                      "threshold_rule": "ataque si score_attack >= threshold", "features": features,
                      "categorical": categorical, "seed": seed, "model_sha256": model_sha,
                      "git_commit": commit, "mlflow_run_id": run.info.run_id}
            (tmp / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
            (tmp / "val_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            pd.DataFrame({"row_id": val.row_id, "y_true": val[target], "score_attack": scores,
                          "y_pred": predict_label(scores, threshold)}).to_csv(tmp / "val_scores.csv", index=False)
            mlflow.log_artifacts(str(tmp), artifact_path="bundle")
        mlflow.set_tag("model_sha256", model_sha)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

    m = metrics
    print(f"✓ {cfg['id']} run={run.info.run_id} recall={m['recall_attack']:.4f} "
          f"precision={m['precision_attack']:.4f} fpr={m['fpr']:.4f} macro_f1={m['macro_f1']:.4f} "
          f"({fit_seconds:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
