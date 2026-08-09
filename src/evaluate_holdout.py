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


def run_holdout_evaluation() -> dict:
    """
    Runs leave-pair-out cross-validation and returns the results as a
    dict, so both the CLI script and the Streamlit dashboard can use the
    same computation without duplicating logic.
    """
    df = load_training_data()
    if df.empty:
        return {"error": "No training data found."}

    moa_vocab = get_moa_vocabulary(list(df["moa_a"]) + list(df["moa_b"]))

    X = np.vstack([
        pair_features(
            row.smiles_a, row.smiles_b, bool(row.is_tumor),
            moa_a=row.moa_a, moa_b=row.moa_b, moa_vocab=moa_vocab,
        )
        for row in df.itertuples()
    ])
    y = df["target"].values

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

    true_top10_idx = set(results.nsmallest(10, "actual_rank").index)
    pred_top10_idx = set(results.nsmallest(10, "predicted_rank").index)
    top10_precision = len(true_top10_idx & pred_top10_idx)

    def top10_table(idx_set, sort_col):
        subset = results.loc[list(idx_set)].sort_values(sort_col)
        records = []
        for _, row in subset.iterrows():
            records.append({
                "pair": f"{row['drug_a_name']} + {row['drug_b_name']}",
                "cell_line": row["cell_line_name"],
                "actual_rank": int(row["actual_rank"]),
                "predicted_rank": int(row["predicted_rank"]),
                "in_both": row.name in (true_top10_idx & pred_top10_idx),
            })
        return records

    true_top10_list = top10_table(true_top10_idx, "actual_rank")
    pred_top10_list = top10_table(pred_top10_idx, "predicted_rank")

    known_combo_results = []
    for name_a, name_b in KNOWN_TOP_COMBOS:
        row = find_pair_row(results, name_a, name_b)
        if row is not None:
            known_combo_results.append({
                "pair": f"{name_a} + {name_b}",
                "actual_rank": int(row["actual_rank"]),
                "predicted_rank": int(row["predicted_rank"]),
            })

    return {
        "spearman": spearman,
        "top10_precision": top10_precision,
        "known_combos": known_combo_results,
        "n_pairs": len(results),
        "true_top10": true_top10_list,
        "pred_top10": pred_top10_list,
    }


def main():
    result = run_holdout_evaluation()
    if "error" in result:
        print(result["error"])
        return

    print(f"Spearman correlation (held-out predicted vs actual rank): {result['spearman']:.3f}\n")
    print(f"Top-10 precision: {result['top10_precision']}/10 of the true top 10 also appear in the "
          f"model's held-out predicted top 10\n")

    print("Known top combos -- held-out prediction check:")
    for combo in result["known_combos"]:
        print(
            f"  {combo['pair']}: actual rank {combo['actual_rank']}, "
            f"held-out PREDICTED rank {combo['predicted_rank']} "
            f"(model never trained on this specific pair)"
        )