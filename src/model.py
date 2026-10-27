"""Pipeline completo (preprocesamiento + estimador) a partir de una configuración."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def signed_log1p(x):
    # Comprime colas pesadas (bytes, load, jitter) conservando el signo.
    return np.sign(x) * np.log1p(np.abs(x))


def build_pipeline(cfg: dict, features: list[str], categorical: list[str], seed: int) -> Pipeline:
    numeric = [c for c in features if c not in categorical]
    num_steps = [SimpleImputer(strategy="median")]
    if cfg["preprocessing"].get("log1p"):
        num_steps.append(FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))
    num_steps.append(StandardScaler())

    pre = ColumnTransformer([
        ("num", make_pipeline(*num_steps), numeric),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            # Categorías raras o nuevas (p. ej. protocolos no vistos) caen en "infrequent".
            OneHotEncoder(handle_unknown="infrequent_if_exist",
                          min_frequency=cfg["preprocessing"].get("min_frequency", 20),
                          sparse_output=False),
        ), categorical),
    ])
    params = dict(cfg["params"])
    params["random_state"] = seed
    estimator = ESTIMATORS[cfg["family"]](**params)
    return Pipeline([("preprocess", pre), ("model", estimator)])
