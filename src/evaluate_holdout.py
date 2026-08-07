import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
import xgboost as xgb

from src.train_baseline import load_training_data
from src.features import pair_features, get_moa_vocabulary

KNOWN_TOP_COMBOS = [
    ("Alvespimycin", "Topotecan"),
    ("Panobinostat", "Ganetespib"),
    ("Carfilzomib", "Selumetinib"),
]


def find_pair_row(df, name_a, name_b):
    mask = (
        (df["drug_a_name"].str.contains(name_a, case=False, na=False) &
         df["drug_b_name"].str.contains(name_b, case=False, na=False))
        |
        (df["drug_a_name"].str.contains(name_b, case=False, na=False) &
         df["drug_b_name"].str.contains(name_a, case=False, na=False))
    )
    match = df[mask]
    return match.iloc[0] if not match.empty else None


def main():
    df = load_training_data()
    if df.empty:
        print("No training data found.")
        return

    moa_vocab = get_moa_vocabulary(list(df["moa_a"]) + list(df["moa_b"]))

    X = np.vstack([
        pair_features(
            row.smiles_a, row.smiles_b, bool(row.is_tumor),
            moa_a=row.moa_a, moa_b=row.moa_b, moa_vocab=moa_vocab,
        )
        for row in df.itertuples()
    ])
    y = df["target"].values

    # Group by drug-PAIR identity (not block_id) -- the same pair never
    # appears in both train and test, even if tested in two cell lines.
    pair_keys = [tuple(sorted((a, b))) for a, b in zip(df["drug_a_name"], df["drug_b_name"])]
    group_ids = pd.factorize(pd.Series(pair_keys))[0]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )

    gkf = GroupKFold(n_splits=5)
    held_out_preds = cross_val_predict(model, X, y, cv=gkf, groups=group_ids)

    results = df.copy()
    results["predicted_score"] = held_out_preds
    results["actual_score"] = y
    results["predicted_rank"] = results["predicted_score"].rank(ascending=False)
    results["actual_rank"] = results["actual_score"].rank(ascending=False)

    spearman = results["predicted_rank"].corr(results["actual_rank"], method="spearman")
    print(f"Spearman correlation (held-out predicted vs actual rank): {spearman:.3f}\n")

    true_top10 = set(results.nsmallest(10, "actual_rank").index)
    pred_top10 = set(results.nsmallest(10, "predicted_rank").index)
    overlap = len(true_top10 & pred_top10)
    print(f"Top-10 precision: {overlap}/10 of the true top 10 also appear in the "
          f"model's held-out predicted top 10\n")

    print("Known top combos -- held-out prediction check:")
    for name_a, name_b in KNOWN_TOP_COMBOS:
        row = find_pair_row(results, name_a, name_b)
        if row is None:
            print(f"  {name_a} + {name_b}: not found")
            continue
        print(
            f"  {name_a} + {name_b}: actual rank {int(row['actual_rank'])}, "
            f"held-out PREDICTED rank {int(row['predicted_rank'])} "
            f"(model never trained on this specific pair)"
        )


if __name__ == "__main__":
    main()