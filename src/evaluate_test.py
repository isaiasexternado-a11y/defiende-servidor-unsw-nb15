"""Evalúa UNA vez la versión congelada sobre el test completo.

    python -m src.evaluate_test

Requisitos: evidencias/seleccion.json commiteado sin cambios (congelación) y que
evidencias/metricas-test.json no exista. El modelo se descarga de Hugging Face por SHA,
así el artefacto evaluado es el mismo que se publica.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data.download import sha256
from src.metrics import binary_metrics, predict_label
from src.predict import load_version


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", type=Path, default=Path("evidencias"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    selection_path = args.evidence / "seleccion.json"
    metrics_path = args.evidence / "metricas-test.json"
    if metrics_path.exists():
        sys.exit("✗ el test ya se evaluó; no se repite")
    if git("status", "--porcelain", str(selection_path)):
        sys.exit("✗ seleccion.json tiene cambios sin commit: congela la selección antes de evaluar")
    freeze_commit = git("log", "-1", "--format=%H", "--", str(selection_path))
    freeze_date = git("log", "-1", "--format=%cI", "--", str(selection_path))

    selection = json.loads(selection_path.read_text())
    version = selection["selected_for_test"]
    v = selection["versions"][version]
    data_cfg = yaml.safe_load(args.data_config.read_text())

    model, cfg = load_version(v["huggingface"]["repo"], v["huggingface"]["revision_sha"])
    if cfg["model_sha256"] != v["model_sha256"] or cfg["threshold"] != v["threshold"]:
        sys.exit("✗ la revisión de Hugging Face no coincide con la selección congelada")

    test_path = Path(data_cfg["out_dir"]) / "test.csv"
    test = pd.read_csv(test_path)
    scores = model.predict_proba(test[cfg["features"]])[:, 1]
    y_pred = predict_label(scores, v["threshold"])
    y_true = test[data_cfg["target"]]

    preds = pd.DataFrame({"row_id": test.row_id, "y_true": y_true, "score_attack": scores, "y_pred": y_pred})
    preds.to_csv(args.evidence / "predicciones-test.csv", index=False)

    # Solo informativo: filas cuyas entradas ya aparecen en train/val (no se excluyen del total).
    splits = pd.read_csv(Path(data_cfg["out_dir"]) / "splits.csv")
    seen = set(splits.loc[splits.split.isin(["train", "val"]), "group"])
    test_groups = splits.loc[splits.split == "test"].set_index("row_id").group
    is_seen = test.row_id.map(test_groups).isin(seen).to_numpy()

    result = {
        "version": version,
        "config_id": v["config_id"],
        "model_sha256": v["model_sha256"],
        "huggingface": v["huggingface"],
        "mlflow_run_id": v["mlflow_run_id"],
        "frozen_selection": {"git_commit": freeze_commit, "committed_at": freeze_date},
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"file": str(test_path), "sha256": sha256(test_path),
                    "raw_file": data_cfg["test_file"],
                    "raw_sha256": sha256(Path(data_cfg["raw_dir"]) / data_cfg["test_file"]),
                    "rows_evaluated": len(test)},
        "threshold": v["threshold"],
        "threshold_rule": v["threshold_rule"],
        "positive_class": "1 = ataque",
        "metrics": binary_metrics(y_true, y_pred),
        "val_metrics_same_threshold": v["val_metrics"],
        "breakdown_informativo": {
            "inputs_seen_in_train_val": {"rows": int(is_seen.sum()),
                                         **binary_metrics(y_true[is_seen], y_pred[is_seen])},
            "inputs_unseen": {"rows": int((~is_seen).sum()),
                              **binary_metrics(y_true[~is_seen], y_pred[~is_seen])},
        },
        "recompute": "python -c \"import pandas as pd; from src.metrics import binary_metrics; "
                     "p=pd.read_csv('evidencias/predicciones-test.csv'); print(binary_metrics(p.y_true, p.y_pred))\"",
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    m = result["metrics"]
    print(f"✓ test {version}: recall={m['recall_attack']:.4f} precision={m['precision_attack']:.4f} "
          f"fpr={m['fpr']:.4f} macro_f1={m['macro_f1']:.4f} [{len(test)} filas]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
