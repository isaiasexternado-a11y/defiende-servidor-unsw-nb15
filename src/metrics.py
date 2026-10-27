"""Métricas binarias con ataque (1) como clase positiva.

Denominadores cero: la métrica vale 0.0 y se marca en `zero_division` (p. ej. precision
cuando el modelo no predice ningún ataque). Así el valor nunca es NaN y queda trazable.
"""

import numpy as np


def _ratio(num: int, den: int, name: str, flags: list[str]) -> float:
    if den == 0:
        flags.append(name)
        return 0.0
    return num / den


def binary_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    flags: list[str] = []
    precision = _ratio(tp, tp + fp, "precision_attack", flags)
    recall = _ratio(tp, tp + fn, "recall_attack", flags)
    fpr = _ratio(fp, fp + tn, "fpr", flags)
    # F1 de cada clase; macro-F1 es su promedio simple.
    f1_attack = _ratio(2 * tp, 2 * tp + fp + fn, "f1_attack", flags)
    f1_normal = _ratio(2 * tn, 2 * tn + fn + fp, "f1_normal", flags)
    return {
        "recall_attack": recall,
        "precision_attack": precision,
        "fpr": fpr,
        "f1_attack": f1_attack,
        "f1_normal": f1_normal,
        "macro_f1": (f1_attack + f1_normal) / 2,
        "accuracy": (tp + tn) / len(y_true),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "zero_division": flags,
    }


def predict_label(scores, threshold: float) -> np.ndarray:
    # Regla del umbral: ataque si score_attack >= umbral.
    return (np.asarray(scores) >= threshold).astype(int)
