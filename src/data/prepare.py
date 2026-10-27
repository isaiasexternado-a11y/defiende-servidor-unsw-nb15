"""Audita los CSV de UNSW-NB15 y crea la partición train/val fija.

    python -m src.data.prepare            # escribe data/prepared/ y evidencias/auditoria.json

Train y val salen solo del archivo de training. Las filas con entradas idénticas
(mismo grupo) quedan siempre en la misma partición. El archivo de test solo se
audita (esquema, duplicados, cruces); sus etiquetas no se usan para decidir nada.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from src.data.download import sha256


def load(path: Path, prefix: str) -> pd.DataFrame:
    # El CSV de training trae BOM UTF-8 en el encabezado.
    df = pd.read_csv(path, encoding="utf-8-sig")
    # row_id = archivo + número de fila (0-based, sin encabezado): identifica la fila sin depender de `id`.
    df.insert(0, "row_id", [f"{prefix}:{i}" for i in range(len(df))])
    return df


def group_keys(df: pd.DataFrame, features: list[str]) -> pd.Series:
    # Huella de las variables predictoras; igual huella = mismas entradas.
    h = pd.util.hash_pandas_object(df[features], index=False)
    return h.map(lambda v: f"{v:016x}")


def describe(df: pd.DataFrame, path: Path, cfg: dict) -> dict:
    return {
        "file": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(df),
        "columns": df.shape[1] - 1,  # sin row_id
        "classes": {str(k): int(v) for k, v in df[cfg["target"]].value_counts().sort_index().items()},
        "attack_cat": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
        "missing_total": int(df.drop(columns="row_id").isna().sum().sum()),
        "dtypes": {c: str(t) for c, t in df.drop(columns="row_id").dtypes.items()},
    }


def duplicates(df: pd.DataFrame, groups: pd.Series, target: str) -> dict:
    labels_per_group = df.groupby(groups)[target].nunique()
    contradictory = labels_per_group[labels_per_group > 1].index
    return {
        "rows": len(df),
        "unique_input_groups": int(groups.nunique()),
        "duplicate_rows_by_inputs": int(groups.duplicated().sum()),
        "contradictory_groups": int(len(contradictory)),
        "rows_in_contradictory_groups": int(groups.isin(contradictory).sum()),
    }


def list_hash(values) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--audit", type=Path, default=Path("evidencias/auditoria.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    raw, out = Path(cfg["raw_dir"]), Path(cfg["out_dir"])
    train_path, test_path = raw / cfg["train_file"], raw / cfg["test_file"]
    train, test = load(train_path, "train"), load(test_path, "test")

    if list(train.columns) != list(test.columns):
        print("✗ train y test tienen columnas distintas", file=sys.stderr)
        return 1
    features = [c for c in train.columns if c not in ["row_id", *cfg["excluded"]]]
    test = test.astype(train.dtypes.to_dict())  # mismos tipos → mismas huellas entre archivos
    target = cfg["target"]

    train["group"] = group_keys(train, features)
    test["group"] = group_keys(test, features)

    # Política: los grupos contradictorios no se pueden aprender; se excluyen de train/val.
    labels_per_group = train.groupby("group")[target].nunique()
    contradictory = set(labels_per_group[labels_per_group > 1].index)
    keep = train[~train["group"].isin(contradictory)].reset_index(drop=True)

    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    _, val_idx = next(sgkf.split(keep, keep[target], keep["group"]))
    keep["split"] = "train"
    keep.loc[val_idx, "split"] = "val"
    tr, va = keep[keep.split == "train"], keep[keep.split == "val"]
    assert not set(tr.group) & set(va.group), "un grupo quedó repartido entre train y val"

    out.mkdir(parents=True, exist_ok=True)
    cols = ["row_id", "id", *features, "attack_cat", target]
    files = {
        "train.csv": tr[cols],
        "val.csv": va[cols],
        "test.csv": test[cols],
        "splits.csv": pd.concat([
            keep[["row_id", "group", "split"]],
            train.loc[train.group.isin(contradictory), ["row_id", "group"]].assign(split="excluded_contradictory"),
            test[["row_id", "group"]].assign(split="test"),
        ]),
    }
    for name, df in files.items():
        df.to_csv(out / name, index=False, lineterminator="\n")

    tr_groups, va_groups, te_groups = set(tr.group), set(va.group), set(test.group)
    audit = {
        "source": cfg["source"],
        "download_date": cfg["download_date"],
        "files": {"train": describe(train.drop(columns="group"), train_path, cfg),
                  "test": describe(test.drop(columns="group"), test_path, cfg)},
        "duplicates": {"train_file": duplicates(train, train["group"], target),
                       "test_file": duplicates(test, test["group"], target)},
        "cleaning_policy": {
            "duplicates": "Se conservan (reflejan la frecuencia real de conexiones), pero cada grupo de entradas "
                          "idénticas queda completo en una sola partición.",
            "contradictions": f"{cfg['contradictions']}: las filas de grupos con etiquetas 0 y 1 se excluyen de "
                              "train/val (quedan marcadas en splits.csv). El test no se modifica.",
            "missing": "Sin faltantes en ningún archivo; el pipeline igual imputa por robustez.",
            "id_leak": "El archivo de training está ordenado por clase (id 1..56000 normal, luego ataques): "
                       "`id` predice la etiqueta y se excluye.",
        },
        "features": {"used": features, "categorical": cfg["categorical"], "excluded": cfg["excluded"]},
        "split": {
            "method": f"StratifiedGroupKFold(n_splits={cfg['n_folds']}, shuffle=True), fold 0 = val; grupo = huella de entradas",
            "seed": cfg["seed"],
            "sizes": {"train": len(tr), "val": len(va), "excluded_contradictory": len(train) - len(keep),
                      "test": len(test)},
            "classes": {"train": {str(k): int(v) for k, v in tr[target].value_counts().sort_index().items()},
                        "val": {str(k): int(v) for k, v in va[target].value_counts().sort_index().items()}},
            "row_id_sha256": {"train": list_hash(tr.row_id), "val": list_hash(va.row_id),
                              "test": list_hash(test.row_id)},
            "file_sha256": {name: sha256(out / name) for name in files},
        },
        "overlaps": {
            "train_val_groups": len(tr_groups & va_groups),
            "test_groups_in_train": len(te_groups & tr_groups),
            "test_groups_in_val": len(te_groups & va_groups),
            "test_rows_with_inputs_in_train_or_val": int(test.group.isin(tr_groups | va_groups).sum()),
            "test_rows_total": len(test),
        },
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    print(f"✓ train {len(tr)} · val {len(va)} · excluidas {len(train) - len(keep)} · test {len(test)}")
    print(f"✓ test con entradas vistas en train/val: {audit['overlaps']['test_rows_with_inputs_in_train_or_val']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
