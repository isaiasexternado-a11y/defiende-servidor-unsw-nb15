"""Fija v1 y v2 (modelo + umbral) usando solo validación y escribe evidencias/seleccion.json.

    python -m src.select

El umbral de v2 sale de los scores de validación guardados en el run de MLflow
(evidencias/mlflow/<run_id>/val_scores.csv); no se reentrena ni se mira test.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.metrics import binary_metrics, predict_label


def sweep(scores: pd.DataFrame, step: float) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(step, 1.0, step), 6):
        m = binary_metrics(scores.y_true, predict_label(scores.score_attack, t))
        rows.append({"threshold": t, **{k: v for k, v in m.items() if isinstance(v, float)},
                     **m["confusion_matrix"]})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/selection.yaml"))
    parser.add_argument("--evidence", type=Path, default=Path("evidencias"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    runs = pd.read_csv(args.evidence / "experimentos.csv").set_index("config_id")
    versions = {}
    for version in ["v1", "v2"]:
        run = runs.loc[cfg[version]]
        run_dir = args.evidence / "mlflow" / run.mlflow_run_id
        scores = pd.read_csv(run_dir / "val_scores.csv")
        if version == "v1":
            threshold, reason = float(run.threshold), "Primera solución funcional: referencia lineal con umbral 0.5."
        else:
            table = sweep(scores, cfg["grid_step"])
            table.to_csv(args.evidence / "umbral_v2.csv", index=False)
            ok = table[table.fpr <= cfg["max_fpr"]]
            best = ok.sort_values(["recall_attack", "threshold"], ascending=[False, True]).iloc[0]
            threshold = float(best.threshold)
            reason = (f"Mejor macro-F1 de validación entre las 4 configuraciones a umbral 0.5 y modelo liviano. "
                      f"Umbral = mayor recall de ataque con FPR <= {cfg['max_fpr']} en validación "
                      f"(barrido en {args.evidence / 'umbral_v2.csv'}).")
        metrics = binary_metrics(scores.y_true, predict_label(scores.score_attack, threshold))
        versions[version] = {
            "config_id": cfg[version], "family": run.family, "mlflow_run_id": run.mlflow_run_id,
            "git_commit_trained": run.git_commit, "dvc_prepared_md5": run.dvc_prepared_md5,
            "model_sha256": run.model_sha256, "threshold": threshold,
            "threshold_rule": "ataque si score_attack >= threshold",
            "val_metrics": metrics, "reason": reason,
            "huggingface": {"repo": None, "revision_sha": None},  # se completa al publicar
        }

    selection = {
        "selected_for_test": "v2",
        "selection_data": "solo validación (data/prepared/val.csv); test no usado",
        "threshold_policy": {k: cfg[k] for k in ["threshold_policy", "max_fpr", "grid_step"]},
        "decided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": versions,
    }
    (args.evidence / "seleccion.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    for v, d in versions.items():
        m = d["val_metrics"]
        print(f"✓ {v} {d['config_id']} t={d['threshold']} recall={m['recall_attack']:.4f} "
              f"precision={m['precision_attack']:.4f} fpr={m['fpr']:.4f} macro_f1={m['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
