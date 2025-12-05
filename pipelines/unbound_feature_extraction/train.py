"""Train classifiers on structural TCR features.

Supports:
- Single-dataset training (backwards compatible with previous config)
- Multiple feature sets (merged on tcr_id) defined in config["feature_sets"]
- Multiple models defined in config["models"]

Optionally computes CV metrics and saves model, metrics, and test predictions.
"""

from __future__ import annotations
import argparse
import json
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Optional heavier models
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

LABEL_SUFFIX = "_binary"


# --------------------------------------------------------------------------- #
# Config / data loading
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _merge_on_id(cfg, dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge multiple feature tables on 'tcr_id'."""
    id_col = cfg.get("id_column", "tcr_id")
    if len(dfs) == 1:
        return dfs[0]
    def _merge(acc, nxt):
        if id_col not in nxt.columns:
            raise ValueError(f"All feature files must contain '{id_col}'.")
        return acc.merge(nxt, how="outer", on=id_col)
    return reduce(_merge, dfs)


def load_dataset(cfg: Dict) -> pd.DataFrame:
    """
    Backwards-compatible single-dataset loader.

    If cfg["feature_files"] is set, merge those CSVs on 'tcr_id'.
    Else fall back to parquet/csv specified by output_feature_table/output_dataset_csv.
    """
    files = cfg.get("feature_files")
    if files:
        dfs = []
        for f in files:
            path = Path(f)
            if not path.exists():
                raise FileNotFoundError(f"Feature file not found: {path}")
            df_i = pd.read_csv(path)
            if "tcr_id" not in df_i.columns:
                raise ValueError(f"Feature file {path} missing 'tcr_id' column.")
            dfs.append(df_i)
        df = _merge_on_id(cfg, dfs)
    else:
        feature_path = Path(cfg["output_feature_table"])
        if feature_path.exists():
            df = pd.read_parquet(feature_path)
        else:
            csv_path = Path(cfg["output_dataset_csv"])
            df = pd.read_csv(csv_path)

    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def add_label_from_metadata(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    If the label column is absent from the feature table, pull it from metadata_csv
    using id_column.
    """
    label_col = cfg.get("label_column")
    if not label_col:
        return df
    if label_col in df.columns:
        return df

    meta_path = Path(cfg["metadata_csv"])
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

    id_col = cfg["id_column"]
    meta = pd.read_csv(meta_path, encoding=cfg.get("metadata_encoding", "utf-8"))
    meta[id_col] = meta[id_col].astype(str).str.replace(".pdb", "", regex=False)

    if label_col not in meta.columns:
        raise ValueError(f"label_column '{label_col}' not found in metadata.")

    merged = df.merge(meta[[id_col, label_col]], how="left", left_on=id_col, right_on=id_col)
    merged = merged.drop(columns=[id_col])
    return merged


# --------------------------------------------------------------------------- #
# Feature utilities
# --------------------------------------------------------------------------- #

def apply_feature_modifications(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Optional, lightweight feature tweaks driven by config."""
    mods = cfg.get("feature_modifications", {})
    df_mod = df.copy()

    # log1p transform for skewed positives
    for col in mods.get("log1p", []):
        if col in df_mod.columns:
            series = df_mod[col]
            df_mod[col] = np.log1p(series.clip(lower=0))

    # winsorize numeric extremes via percentile clipping
    clip_pct = mods.get("clip_percentile")
    if clip_pct and isinstance(clip_pct, (list, tuple)) and len(clip_pct) == 2:
        lo, hi = clip_pct
        num_cols = df_mod.select_dtypes(include=["number"]).columns
        for col in num_cols:
            qlo, qhi = df_mod[col].quantile([lo, hi])
            df_mod[col] = df_mod[col].clip(lower=qlo, upper=qhi)

    return df_mod


def select_feature_columns(df: pd.DataFrame, label_col: str) -> Dict[str, List[str]]:
    skip_cols = {"tcr_id", "pdb_path"}
    if label_col in df.columns:
        skip_cols.add(label_col)
    skip_cols.update({c for c in df.columns if c.endswith("error")})
    skip_cols.update({c for c in df.columns if c.endswith(LABEL_SUFFIX)})

    feature_cols = [c for c in df.columns if c not in skip_cols]
    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == object or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return {"all": feature_cols, "categorical": cat_cols, "numerical": num_cols}


def get_feature_names(preproc: ColumnTransformer, cat_cols: List[str], num_cols: List[str]) -> List[str]:
    """Return final feature names after preprocessing."""
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        # Fallback: manual names
        num_names = num_cols
        cat_names = [f"cat__{c}" for c in cat_cols]
        return list(num_names) + cat_names


# --------------------------------------------------------------------------- #
# Model builder and training
# --------------------------------------------------------------------------- #

def build_model(model_name: str, cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    """Factory for different model architectures with preprocessing."""
    model_name = model_name.lower()
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=13,
        )
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=13)
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        clf = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=13,
            n_jobs=-1,
        )
    # elif model_name == "svm":
    #     clf = svm.SVC()
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    return Pipeline([("preprocess", preproc), ("clf", clf)])


def train_and_evaluate(
    df: pd.DataFrame,
    label_col: str,
    cfg: Dict,
    model_name: str,
) -> Dict:
    """Train model with CV on train, evaluate on held-out test."""

    cols = select_feature_columns(df, label_col)
    feature_cols = cols["all"]

    df_proc = apply_feature_modifications(df, cfg)
    X = df_proc[feature_cols]

    le = LabelEncoder()
    y = le.fit_transform(df_proc[label_col])
    # y = np.vstack([y, 1 - y]).T

    random_seed = cfg.get("random_seed", 13)
    test_size = cfg.get("test_size", 0.2)
    cv_folds = cfg.get("cv_folds", 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    model = build_model(model_name, cols["categorical"], cols["numerical"])

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_seed,
    )
    cv_pred = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    cv_auc = roc_auc_score(y_train, cv_pred)
    cv_ap = average_precision_score(y_train, cv_pred)

    model.fit(X_train, y_train)
    test_prob = model.predict_proba(X_test)
    test_pred = (test_prob[:, 1] >= 0.5).astype(int)

    metrics = {
        "train_cv_auc": cv_auc,
        "train_cv_ap": cv_ap,
        "test_auc": roc_auc_score(y_test, test_prob[:, 1]),
        "test_ap": average_precision_score(y_test, test_prob[:, 1]),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(
            y_test, test_pred, output_dict=True
        ),
        "feature_columns": feature_cols,
        "model_name": model_name,
    }

    return {
        "model": model,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test.tolist(),
        "test_prob_0": test_prob[:, 0].tolist(),
        "test_prob_1": test_prob[:, 1].tolist(),
    }


# --------------------------------------------------------------------------- #
# Saving artifacts
# --------------------------------------------------------------------------- #

def save_artifacts(
    result: Dict,
    cfg: Dict,
    model_suffix: str,
) -> Tuple[Path, Path, Path]:
    """Save model, metrics, and test predictions. Return paths."""
    model_dir = Path(cfg["output_model_dir"])
    report_dir = Path(cfg["output_report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = model_dir / f"maura_hnncc_{model_suffix}.joblib"
    joblib.dump(result["model"], model_path)

    # Metrics
    metrics_path = report_dir / f"maura_hnncc_{model_suffix}_metric.json"
    with metrics_path.open("w") as f:
        json.dump(result["metrics"], f, indent=2)

    # Predictions
    predictions_path = report_dir / f"maura_hnncc_{model_suffix}_metric_pred.csv"
    pd.DataFrame(
        {
            "row_idx": result["X_test"].index.tolist(),
            "y_true": result["y_test"],
            "y_prob": result["test_prob"],
        }
    ).to_csv(predictions_path, index=False)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")

    return model_path, metrics_path, predictions_path


# --------------------------------------------------------------------------- #
# High-level experiment runner
# --------------------------------------------------------------------------- #

def run_single_dataset(cfg: Dict, model_name: str, run_suffix: Optional[str] = None) -> None:
    """Backwards-compatible: one dataset, one model."""
    df = load_dataset(cfg)
    df = add_label_from_metadata(df, cfg)

    label_col = cfg.get("label_column")
    if not label_col or label_col not in df.columns:
        raise ValueError("label_column must be set in config and exist in the feature table.")

    if cfg.get("filter_na_labels", True):
        df = df.loc[~df[label_col].isna()]

    print(f"[single] Loaded dataset with {df.shape[0]} samples and {df.shape[1]} columns.")
    result = train_and_evaluate(df, label_col, cfg, model_name=model_name)

    suffix_core = run_suffix or model_name
    save_artifacts(result, cfg, suffix_core)


def run_multi_experiments(cfg: Dict, run_suffix: Optional[str] = None) -> None:
    """
    If cfg["feature_sets"] and/or cfg["models"] are defined, loop over all combos.

    Example YAML:
      feature_sets:
        struct:
          - /path/to/struct_features.csv
        inter:
          - /path/to/interface_features.csv
        struct_inter:
          - /path/to/struct_features.csv
          - /path/to/interface_features.csv

      models: ["xgboost", "logreg", "random_forest"]
    """
    feature_sets: Dict[str, List[str]] = cfg.get("feature_sets", {})
    models: List[str] = cfg.get("models", [])

    if not feature_sets or not models:
        raise ValueError("feature_sets and models must be provided in config for multi-experiments.")

    # Pre-load each unique feature file only once
    unique_paths = sorted({p for paths in feature_sets.values() for p in paths})
    feature_cache: Dict[str, pd.DataFrame] = {}
    for path_str in unique_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        df = pd.read_csv(path)
        if "tcr_id" not in df.columns:
            raise ValueError(f"Feature file {path} missing 'tcr_id' column.")
        feature_cache[path_str] = df

    label_col = cfg.get("label_column")
    if not label_col:
        raise ValueError("label_column must be set in config.")

    for model_name in models:
        print(f"\n=== MODEL: {model_name} ===")
        for fs_name, paths in feature_sets.items():
            print(f"--- Feature set: {fs_name} ---")
            dfs = [feature_cache[p] for p in paths]
            df_features = _merge_on_id(cfg, dfs)

            # Attach label from metadata
            df = add_label_from_metadata(df_features, cfg)

            # Optionally drop auxiliary columns defined in cfg["feature_columns"]
            if "feature_columns" in cfg:
                df = df.loc[:, ~df.columns.isin(cfg["feature_columns"])]

            if label_col not in df.columns:
                raise ValueError(f"label_column '{label_col}' missing after merge for feature set '{fs_name}'.")

            if cfg.get("filter_na_labels", True):
                df = df.loc[~df[label_col].isna()]

            print(f"Dataset [{fs_name}] has {df.shape[0]} samples, {df.shape[1]} columns.")

            result = train_and_evaluate(df, label_col, cfg, model_name=model_name)

            suffix_parts = [model_name, fs_name]
            if run_suffix:
                suffix_parts.insert(0, run_suffix)
            suffix = "_".join(suffix_parts)

            save_artifacts(result, cfg, suffix)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main(args: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train classifier on structural features")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        help="Model to train if not using config['models']: "
             "logreg, random_forest, gradient_boosting, xgboost",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional prefix for saved artifacts (e.g. 'specific')",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="If set, use config['feature_sets'] and config['models'] to run multiple experiments.",
    )
    parsed = parser.parse_args(args)

    cfg = load_config(parsed.config)
    if parsed.run_suffix:
        cfg["run_suffix_override"] = parsed.run_suffix

    # Prefer random_seed key
    if "seed" in cfg and "random_seed" not in cfg:
        cfg["random_seed"] = cfg["seed"]

    if parsed.multi:
        run_multi_experiments(cfg, run_suffix=parsed.run_suffix)
    else:
        # Single dataset + single model (backwards compatible)
        run_single_dataset(cfg, model_name=parsed.model, run_suffix=parsed.run_suffix)


if __name__ == "__main__":
    main()
