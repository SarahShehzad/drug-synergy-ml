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

## Data sources

- Primary: Ferrer, M. et al. (2018), Synapse `syn5611796`
- Optional pretraining: [DrugComb](https://drugcomb.org),
  NCI-ALMANAC, O'Neil et al. dataset

## Reference

Zhou, K., Shehzad, S., Ahmed, Z., Zhao, O., Sapriza, C., Zamora, M.,
Santamaria, U., & Zamora, P. (2025). Prioritizing Combinational Drug
Screening: A Ranking System for In Vitro Drug Combinations in
Neurofibromatosis Type 1. *bioRxiv*. https://doi.org/10.1101/2025.08.02.667047
