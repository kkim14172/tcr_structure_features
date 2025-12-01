# save as e.g. inspect_feature_importance.py and run with `python inspect_feature_importance.py`
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# Ensure repo root is on sys.path when running from pipelines/maura_hnncc
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pipelines.maura_hnncc.train import (
    load_dataset, add_label_from_metadata, select_feature_columns, apply_feature_modifications
)

cfg = yaml.safe_load(Path("pipelines/maura_hnncc/config.yaml").read_text())
model_path = Path("outputs/maura_hnncc/models/maura_hnncc_specific_xgb_struct_resenergy_interenergy.joblib")

# load data exactly as in training
df = load_dataset(cfg)
df = add_label_from_metadata(df, cfg)
df = apply_feature_modifications(df, cfg)
cols = select_feature_columns(df, cfg["label_column"])
X = df[cols["all"]]

# load model and pull pieces
pipeline = joblib.load(model_path)
preproc = pipeline.named_steps["preprocess"]
clf = pipeline.named_steps["clf"]

# feature names after preprocessing (includes one-hot names)
feature_names = list(preproc.get_feature_names_out())
print(f"Number of features after preprocessing: {len(feature_names)}")
# XGBoost gain-based importance mapped to names
booster = clf.get_booster()
gain = booster.get_score(importance_type="gain")
mapped = []
for fid, val in gain.items():  # fid like "f123"
    idx = int(fid[1:])
    name = feature_names[idx] if idx < len(feature_names) else fid
    mapped.append((name, val))
imp_df = pd.DataFrame(mapped, columns=["feature", "gain"]).sort_values("gain", ascending=False)

# plot top 30
plt.figure(figsize=(8, 10))
imp_df.head(30).plot(kind="barh", x="feature", y="gain", legend=False)
plt.gca().invert_yaxis()
plt.tight_layout()
out_path = Path("outputs/maura_hnncc/reports/xgb_gain_importance.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=200)
print(f"Saved gain plot to {out_path}")
