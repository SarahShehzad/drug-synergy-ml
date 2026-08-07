# ML-Powered Drug Synergy Predictor

Extends the Composite Matrix Reduction Score (CMRS) work from
["Prioritizing Combinational Drug Screening" (Zhou, Shehzad, Ahmed, et al., 2025)](https://www.biorxiv.org/content/10.1101/2025.08.02.667047v1)
by training a machine learning model to **predict** synergy scores for
drug pairs, rather than only ranking pairs that were already screened.

CMRS is a rule-based scoring system applied *after* a drug combination has
been tested in vitro. It cannot say anything about a pair that hasn't been
run through the assay. This project trains a model on the same underlying
screening data (Ferrer et al. 2018, Synapse `syn5611796`) so that, given two
drug structures and a cell line, it can output a predicted synergy score for
combinations that were never directly tested.

## Project status: v1 skeleton

This repo is scaffolded but not yet populated with real data — you need a
free Synapse account and access approval for `syn5611796` before
`data_loader.py` will pull anything real. Everything else (DB schema,
feature engineering, training script, demo app) is ready to run once data
is loaded.

## Architecture

```
Synapse (Ferrer et al. 2018 screening data)
        |
        v
data_loader.py  --------->  synergy.db (SQLite)
        |                        ^
        v                        |
features.py (RDKit fingerprints) |
        |                        |
        v                        |
train_baseline.py  -------------->  models/baseline_model.pkl
        |
        v
app/streamlit_app.py  (pick 2 drugs -> predicted synergy score)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You'll need a Synapse account (free) and to request access to the dataset:
https://www.synapse.org/Synapse:syn5611796

Then configure credentials (do NOT commit these):
```bash
export SYNAPSE_USERNAME=your_username
export SYNAPSE_AUTH_TOKEN=your_personal_access_token
```

## Roadmap

- [x] Repo scaffold, SQLite schema, feature/training skeletons
- [ ] Pull real Ferrer et al. data via Synapse, load into `synergy.db`
- [ ] Reproduce CMRS normalization as ground-truth baseline (see paper Eq. 1)
- [ ] Train baseline XGBoost model on RDKit fingerprints + cell line
- [ ] Evaluate: recover paper's top combos (Alvespimycin+Ganetespib,
      carfilzomib+selumetinib) as a sanity check
- [ ] Streamlit demo: pick two drugs -> predicted synergy score
- [ ] Stretch: pretrain on DrugComb/O'Neil public synergy datasets,
      fine-tune on NF1 data
- [ ] Stretch: graph neural network instead of fingerprints
- [ ] Stretch: SHAP interpretability on feature importance
- [ ] Stretch: compute Bliss on reference line too, use tumor−reference
      Bliss delta as a selectivity-adjusted synergy score; likely
      correlates more strongly with CMRS than raw tumor-only Bliss does

## Write up
- Model shows real, moderate predictive signal on genuinely held-out drug pairs (Spearman p = 0.57), correctly recovering moderately ranked known combinations (Carfilzomib+Selumetinib: true rank 38, predicted rank 103) even without training on that specific pair. It struggles specifically with extreme statistical outliers. It would be the single highest-scoring pair in the entire dataset (18 points beyond the next-best) which was consistently mispredicted across every feature configuration tried, consistent with a known limitation of tree-based models: they can't extrapolate meaningfully beyond the range of values seen in similar training examples, especially with a dataset this size (39 usable drugs).

- **Exploratory predictions on genuinely novel compounds** (never screened
in this dataset, paired against proven top performers by shared
mechanism) ranked proteasome inhibitors (Carfilzomib + Marizomib/
Ixazomib/Bortezomib) highest, entirely on structural/mechanism features, independently converging on the same drug class the paper's authors
flagged as a notable recurring performer. Treated as lower-confidence
extrapolation, not validated results.

## Limitations
- Predictions are only meaningful for drugs structurally/mechanistically
  similar to the training panel. This is not a general-purpose synergy
  predictor for arbitrary drug pairs.
- Small dataset (39 usable drugs, ~740 usable pairs after excluding
  missing-structure entries) limits how well any model can extrapolate.
- The model reliably underperforms on extreme outliers (see Key Findings).
- One drug (`NCGC00183656-04`) has no resolvable name or SMILES in the
  source data and is excluded from feature-based analysis.

## Data sources

- Primary: Ferrer, M. et al. (2018), Synapse `syn5611796`
- Optional pretraining: [DrugComb](https://drugcomb.org),
  NCI-ALMANAC, O'Neil et al. dataset

## Reference

Zhou, K., Shehzad, S., Ahmed, Z., Zhao, O., Sapriza, C., Zamora, M.,
Santamaria, U., & Zamora, P. (2025). Prioritizing Combinational Drug
Screening: A Ranking System for In Vitro Drug Combinations in
Neurofibromatosis Type 1. *bioRxiv*. https://doi.org/10.1101/2025.08.02.667047
