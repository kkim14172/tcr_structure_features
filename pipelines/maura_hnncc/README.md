# Maura HNNCC structural pipeline

This folder contains a minimal, reproducible pipeline to turn tcrbuilder2 structures into a feature table and train a binary FEST classifier.

## Layout
- `config.yaml` — paths for raw PDBs, metadata, outputs, and split settings.
- `feature_extraction.py` — parses each PDB, aligns frameworks to a reference (least-squares on non-CDR residues), derives structural summary features (V-domain geometry, CDR loop geometry/composition, SASA, shape descriptors), and joins labels.
- `train.py` — trains a balanced logistic regression with cross-validation and saves metrics/model.
- `analysis.py` — inspects saved metrics and model coefficients.

Outputs are written under `outputs/maura_hnncc/` (features, models, reports).

## Usage
Assumes dependencies from `requirements.txt` plus `pandas`, `numpy`, `scikit-learn`, `pyyaml`, and `joblib` are installed.

1. Extract features (writes parquet+csv):

```bash
python pipelines/maura_hnncc/feature_extraction.py --config pipelines/maura_hnncc/config.yaml
```

2. Train baseline classifier and save metrics/model:

```bash
python pipelines/maura_hnncc/train.py --config pipelines/maura_hnncc/config.yaml
```

3. Inspect results and top coefficients:

```bash
python pipelines/maura_hnncc/analysis.py --config pipelines/maura_hnncc/config.yaml
```

## Notes
- Labels default to `HPV_specific`; any value in `positive_labels` is treated as positive and everything else as negative. Adjust in `config.yaml` if you want to target `fest_label` instead.
- Feature extraction tolerates missing chains but records flags (`has_chainA`, `has_chainB`) and skips failed structures instead of aborting the run.
- The training script stratifies the train/test split and uses `class_weight="balanced"` to account for label imbalance.

## Feature set (per PDB)
- **Framework alignment**: `framework_alignment_rmsd` after least-squares fit of non-CDR residues to a reference (optional `reference_pdb`, otherwise first PDB).
- **V-domain geometry**: chain lengths, radius of gyration (`chainA_rg`, `chainB_rg`), span (max CA-CA distance), mean CA B-factor, COM distance, interdomain axis/twist angles, inter-chain contact fractions (`lt8A`, `lt5A`), min inter-chain CA distance, presence flags.
- **SASA/contact**: per-chain SASA (`chainA_sasa`, `chainB_sasa`), complex SASA, inter-chain contact area.
- **Global shape**: overall Rg (`rg_all`), asphericity, acylindricity, eccentricity, bounding-box volume.
- **CDR loop geometry** (for each CDR1/2/3 of α and β): length, Rg, span, centroid (x,y,z), height (mean/max) from V-domain plane, orientation vector, mean SASA.
- **CDR side-chain reach**: mid-loop side-chain stretch (max/mean/p95 CA-to-sidechain-tip distance) for each CDR.
- **CDR composition**: hydrophobic/positive/negative/aromatic/polar counts plus 20-aa frequency vectors per loop.
