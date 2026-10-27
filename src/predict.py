"""Descarga una versión del modelo desde Hugging Face por SHA y predice un CSV.

    python -m src.predict --repo USUARIO/defiende-servidor-unsw-nb15 --revision <SHA> \
        --input data/prepared/val.csv --out preds.csv

Verifica que el SHA-256 de model.joblib coincida con el de config.json antes de usarlo.
Salida: row_id, score_attack, y_pred (+ y_true si la entrada trae `label`).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

from src.metrics import predict_label
from src.train import file_sha256


def load_version(repo: str, revision: str, cache_dir: Path | None = None):
    path = lambda f: Path(hf_hub_download(repo, f, revision=revision, cache_dir=cache_dir))
    cfg = json.loads(path("config.json").read_text())
    model_path = path("model.joblib")
    if file_sha256(model_path) != cfg["model_sha256"]:
        sys.exit("✗ SHA-256 de model.joblib distinto al de config.json")
    return joblib.load(model_path), cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True, help="SHA completo del commit en Hugging Face")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    model, cfg = load_version(args.repo, args.revision, args.cache_dir)
    data = pd.read_csv(args.input)
    scores = model.predict_proba(data[cfg["features"]])[:, 1]
    out = pd.DataFrame({"row_id": data["row_id"]})
    if "label" in data:
        out["y_true"] = data["label"]
    out["score_attack"] = scores
    out["y_pred"] = predict_label(scores, cfg["threshold"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"✓ {cfg['version']} ({cfg['config_id']}, umbral {cfg['threshold']}) → {args.out} [{len(out)} filas]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
