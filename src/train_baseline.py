"""
Trains a baseline XGBoost model to predict synergy score from drug-pair
fingerprints + cell line type.

Ground truth target: reproduced CMRS score (see paper Section 2.2) or
Bliss synergy score, whichever you've computed into synergy_scores table.

Usage:
    python -m src.train_baseline
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

from src.db import get_connection
from src.features import pair_features

MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "baseline_model.pkl"


def load_training_data() -> pd.DataFrame:
    conn = get_connection()
    query = """
    SELECT
        sc.block_id,
        da.name AS drug_a_name, da.smiles AS smiles_a,
        db_.name AS drug_b_name, db_.smiles AS smiles_b,
        cl.is_tumor,
        sc.cmrs_score AS target
    FROM synergy_scores sc
    JOIN drugs da ON sc.drug_a_id = da.drug_id
    JOIN drugs db_ ON sc.drug_b_id = db_.drug_id
    JOIN cell_lines cl ON sc.cell_line_id = cl.cell_line_id
    WHERE sc.cmrs_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def train():
    df = load_training_data()
    if df.empty:
        print("No training data found — run data_loader.py first and populate synergy_scores.")
        return

    X = np.vstack([
        pair_features(row.smiles_a, row.smiles_b, bool(row.is_tumor))
        for row in df.itertuples()
    ])
    y = df["target"].values

    # Group by block_id so the same drug pair never appears in both
    # train and test splits — critical to avoid leaking pair identity,
    # a common mistake in synergy-prediction literature.
    groups = df["block_id"].values

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(
        model, X, y, cv=gkf, groups=groups, scoring="neg_mean_absolute_error"
    )
    print(f"Cross-val MAE: {-scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    train()
