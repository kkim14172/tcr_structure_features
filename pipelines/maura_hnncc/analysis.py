"""Lightweight analysis helpers for the maura_hnncc pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

LABEL_SUFFIX = "_binary"


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_metrics(cfg: Dict, model_name: str) -> Dict:
    metrics_path = Path(cfg["output_report_dir"]) / f"metrics_{model_name}.json"
    with metrics_path.open("r") as f:
        return json.load(f)


def load_model(cfg: Dict, model_name: str):
    model_path = Path(cfg["output_model_dir"]) / f"maura_hnncc_{model_name}.joblib"
    return joblib.load(model_path)


def load_dataset(cfg: Dict) -> pd.DataFrame:
    feature_path = Path(cfg["output_feature_table"])
    if feature_path.exists():
        return pd.read_parquet(feature_path)
    return pd.read_csv(Path(cfg["output_dataset_csv"]))


def describe_dataset(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    label_col = cfg["label_column"] + LABEL_SUFFIX
    summary = {
        "n_total": len(df),
        "n_labeled": int(df[label_col].notna().sum()),
        "positives": int((df[label_col] == 1).sum()),
        "negatives": int((df[label_col] == 0).sum()),
    }
    return pd.DataFrame([summary])


def top_coefficients(model, feature_names, k: int = 10) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    coef = clf.coef_.ravel()
    order = np.argsort(coef)
    top_neg = [(feature_names[i], coef[i]) for i in order[:k]]
    top_pos = [(feature_names[i], coef[i]) for i in order[-k:][::-1]]
    return pd.DataFrame(top_pos + top_neg, columns=["feature", "coefficient"])


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect metrics and model coefficients")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"), help="YAML config path")
    parser.add_argument("--topk", type=int, default=10, help="Number of top coefficients to display from each tail")
    parser.add_argument("--model", type=str, default="logreg", help="Model name used in training")
    parser.add_argument("--show-shap", action="store_true", help="Display top SHAP mean |values| if available")
    parsed = parser.parse_args(args)

    cfg = load_config(parsed.config)
    df = load_dataset(cfg)
    metrics = load_metrics(cfg, parsed.model)
    model = load_model(cfg, parsed.model)

    print("Dataset summary:")
    print(describe_dataset(df, cfg).to_string(index=False))
    print("\nSaved metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
    # feature names preserved from metrics
    feature_names = metrics.get("feature_columns") or []
    if parsed.model == "logreg":
        print("\nTop coefficients (positive then negative):")
        coeff_df = top_coefficients(model, feature_names, k=parsed.topk)
        print(coeff_df.to_string(index=False))
    else:
        print("\nCoefficient view only available for logreg.")

    if parsed.show_shap and metrics.get("shap_mean_abs"):
        print("\nTop SHAP mean |value| features:")
        shap_vals = np.array(metrics["shap_mean_abs"])
        order = np.argsort(shap_vals)[::-1][:parsed.topk]
        for idx in order:
            fname = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
            print(f"  {fname}: {shap_vals[idx]:.4f}")


if __name__ == "__main__":
    main()
