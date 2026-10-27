"""Prueba de recuperación: entradas fijas, salidas de referencia y comparación.

    python -m src.recovery prepare     # entorno original: fija entradas y guarda salidas "antes"
    python -m src.recovery check DIR   # compara DIR/despues_v{1,2}.csv contra las de referencia

Las salidas "antes" salen del model.joblib guardado en el run de MLflow (artefacto original);
las "después" salen de descargar cada versión de Hugging Face por SHA en un entorno nuevo.
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import joblib
import numpy as np
import pandas as pd

from src.metrics import predict_label
from src.train import TRACKING_URI, file_sha256

OUT = Path("evidencias/recuperacion")
N_ROWS = 25
SEED = 42
TOLERANCE = 1e-9  # diferencia absoluta máxima aceptada en score_attack


def prepare() -> int:
    selection = json.loads(Path("evidencias/seleccion.json").read_text())
    val = pd.read_csv("data/prepared/val.csv")
    # Muestra fija con ambas clases: mitad normal, mitad ataque.
    sample = pd.concat([
        val[val.label == 0].sample(N_ROWS // 2, random_state=SEED),
        val[val.label == 1].sample(N_ROWS - N_ROWS // 2, random_state=SEED),
    ]).sort_values("row_id")
    OUT.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT / "entrada.csv", index=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    hashes = {"entrada.csv": file_sha256(OUT / "entrada.csv")}
    for version, v in selection["versions"].items():
        bundle = Path(client.download_artifacts(v["mlflow_run_id"], "bundle"))
        assert file_sha256(bundle / "model.joblib") == v["model_sha256"]
        model = joblib.load(bundle / "model.joblib")
        cfg = json.loads((bundle / "config.json").read_text())
        scores = model.predict_proba(sample[cfg["features"]])[:, 1]
        out = pd.DataFrame({"row_id": sample.row_id, "y_true": sample.label, "score_attack": scores,
                            "y_pred": predict_label(scores, v["threshold"])})
        out.to_csv(OUT / f"antes_{version}.csv", index=False)
        hashes[f"antes_{version}.csv"] = file_sha256(OUT / f"antes_{version}.csv")
        hashes[f"model_{version}.joblib"] = v["model_sha256"]
    (OUT / "referencia.json").write_text(json.dumps({"n_rows": len(sample), "seed": SEED, "hashes": hashes},
                                                    indent=2) + "\n")
    print(f"✓ {len(sample)} entradas fijas y salidas de referencia en {OUT}")
    return 0


def check(after_dir: Path) -> int:
    result, ok = {"tolerance_abs_score": TOLERANCE, "versions": {}}, True
    for version in ["v1", "v2"]:
        before = pd.read_csv(OUT / f"antes_{version}.csv")
        after = pd.read_csv(after_dir / f"despues_{version}.csv")
        same_rows = before.row_id.tolist() == after.row_id.tolist()
        max_diff = float(np.abs(before.score_attack - after.score_attack).max()) if same_rows else None
        same_labels = bool(same_rows and (before.y_pred == after.y_pred).all())
        passed = same_rows and same_labels and max_diff <= TOLERANCE
        ok &= passed
        result["versions"][version] = {"rows": len(after), "same_row_ids": same_rows,
                                       "max_abs_score_diff": max_diff, "same_labels": same_labels,
                                       "passed": passed}
        print(f"{'✓' if passed else '✗'} {version}: filas={len(after)} max|Δscore|={max_diff} etiquetas iguales={same_labels}")
    result["passed"] = ok
    (after_dir / "comparacion.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("check")
    c.add_argument("after_dir", type=Path)
    args = parser.parse_args()
    return prepare() if args.cmd == "prepare" else check(args.after_dir)


if __name__ == "__main__":
    sys.exit(main())
