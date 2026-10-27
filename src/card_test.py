"""Añade el resultado de test a la model card de la versión evaluada (solo README.md).

    python -m src.card_test

Sube una revisión nueva que cambia únicamente README.md y comprueba que model.joblib
y config.json siguen idénticos a los de la revisión evaluada. Registra ambas revisiones
en evidencias/model-card-test.json.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

FROZEN_FILES = ["model.joblib", "config.json", "requirements.txt"]


def file_ids(api: HfApi, repo: str, revision: str) -> dict:
    infos = api.get_paths_info(repo, FROZEN_FILES, revision=revision)
    return {i.path: (i.lfs.sha256 if i.lfs else i.blob_id) for i in infos}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", type=Path, default=Path("evidencias"))
    args = parser.parse_args()

    result = json.loads((args.evidence / "metricas-test.json").read_text())
    hf = result["huggingface"]
    repo, evaluated = hf["repo"], hf["revision_sha"]
    m, cm = result["metrics"], result["metrics"]["confusion_matrix"]
    val = result["val_metrics_same_threshold"]
    seen, unseen = result["breakdown_informativo"]["inputs_seen_in_train_val"], \
        result["breakdown_informativo"]["inputs_unseen"]

    card = Path(hf_hub_download(repo, "README.md", revision=evaluated)).read_text()
    card += f"""
## Evaluación final en test

Evaluación única sobre el test oficial completo ({result['dataset']['rows_evaluated']} filas,
SHA-256 `{result['dataset']['raw_sha256']}`), con la selección congelada en el commit Git
`{result['frozen_selection']['git_commit']}`.

- **Revisión evaluada:** `{evaluated}` (tag `{hf['tag']}`)
- **SHA-256 de `model.joblib` evaluado:** `{result['model_sha256']}` (sin cambios en esta revisión de documentación)
- **Umbral:** {result['threshold']} — `{result['threshold_rule']}`

| Conjunto | Recall ataque | Precision ataque | FPR | Macro-F1 |
|---|---|---|---|---|
| Validación | {val['recall_attack']:.4f} | {val['precision_attack']:.4f} | {val['fpr']:.4f} | {val['macro_f1']:.4f} |
| **Test** | **{m['recall_attack']:.4f}** | **{m['precision_attack']:.4f}** | **{m['fpr']:.4f}** | **{m['macro_f1']:.4f}** |
| Test, entradas vistas en train/val ({seen['rows']} filas) | {seen['recall_attack']:.4f} | {seen['precision_attack']:.4f} | {seen['fpr']:.4f} | {seen['macro_f1']:.4f} |
| Test, entradas nuevas ({unseen['rows']} filas) | {unseen['recall_attack']:.4f} | {unseen['precision_attack']:.4f} | {unseen['fpr']:.4f} | {unseen['macro_f1']:.4f} |

Matriz de confusión en test: TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']}.

El FPR sube de {val['fpr']:.3f} en validación a {m['fpr']:.3f} en test: el test oficial tiene otra
proporción de clases y otra distribución de tráfico normal, y el umbral elegido en validación
no se transfiere. El resultado se conserva sin reajustar el umbral.
"""
    before = file_ids(HfApi(), repo, evaluated)
    with_card = HfApi().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                                    commit_message=f"Model card: resultado de test de {hf['tag']} (modelo sin cambios)")
    after = file_ids(HfApi(), repo, with_card.oid)
    if before != after:
        sys.exit("✗ los archivos del modelo cambiaron entre revisiones")

    record = {"evaluated_revision": evaluated, "card_revision": with_card.oid, "repo": repo,
              "unchanged_files": after, "model_sha256": result["model_sha256"]}
    (args.evidence / "model-card-test.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"✓ model card actualizada: {repo}@{with_card.oid} (model.joblib idéntico a {evaluated})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
