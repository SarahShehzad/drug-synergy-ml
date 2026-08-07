import pickle
from pathlib import Path

import pandas as pd

from src.db import get_connection
from src.features import pair_features, get_moa_vocabulary
from src.retry_smiles import lookup_smiles, clean_name

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_model.pkl"

# Anchor drugs: proven top performers already in your panel.
# Candidates: real compounds sharing that anchor's mechanism, never
# screened in this dataset.
EXPLORATION_SETS = [
    {
        "anchor": "Ganetespib",
        "moa": "HSP90 inhibitor",
        "candidates": ["Onalespib", "Luminespib", "Tanespimycin", "Retaspimycin"],
    },
    {
        "anchor": "Carfilzomib",
        "moa": "Proteasome inhibitor",
        "candidates": ["Bortezomib", "Ixazomib", "Marizomib"],
    },
    {
        "anchor": "Selumetinib",
        "moa": "Mek1/2 inhibitor",
        "candidates": ["Trametinib", "Binimetinib", "Cobimetinib"],
    },
    {
        "anchor": "Panobinostat",
        "moa": "HDAC inhibitor",
        "candidates": ["Vorinostat", "Belinostat", "Romidepsin"],
    },
]


def get_anchor_info(conn, name: str):
    row = conn.execute(
        "SELECT smiles, moa FROM drugs WHERE name LIKE ?", (f"%{name}%",)
    ).fetchone()
    return row if row else (None, None)


def main():
    if not MODEL_PATH.exists():
        print("No trained model found -- run src.train_baseline first.")
        return

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    conn = get_connection()

    # Must match the vocabulary the model was actually trained on.
    all_moas = [r[0] for r in conn.execute("SELECT moa FROM drugs WHERE moa IS NOT NULL").fetchall()]
    moa_vocab = get_moa_vocabulary(all_moas)

    results = []
    for exploration in EXPLORATION_SETS:
        anchor_smiles, anchor_moa = get_anchor_info(conn, exploration["anchor"])
        if anchor_smiles is None:
            print(f"Skipping {exploration['anchor']} -- not found or missing SMILES in database.")
            continue

        for candidate_name in exploration["candidates"]:
            smiles = lookup_smiles(candidate_name)
            if smiles is None:
                smiles = lookup_smiles(clean_name(candidate_name))
            if smiles is None:
                print(f"  Could not find SMILES for {candidate_name}, skipping.")
                continue

            features = pair_features(
                anchor_smiles, smiles, cell_line_is_tumor=True,
                moa_a=anchor_moa, moa_b=exploration["moa"], moa_vocab=moa_vocab,
            ).reshape(1, -1)

            predicted_score = model.predict(features)[0]
            results.append({
                "anchor": exploration["anchor"],
                "candidate": candidate_name,
                "shared_moa": exploration["moa"],
                "predicted_cmrs": round(float(predicted_score), 1),
            })

    conn.close()

    if not results:
        print("No predictions generated.")
        return

    df = pd.DataFrame(results).sort_values("predicted_cmrs", ascending=False)
    print("\nEXPLORATORY predictions -- novel compounds never screened in this dataset.")
    print("These extrapolate beyond validated training data; treat as lower-confidence.\n")
    print(df.to_string(index=False))

    out_path = Path(__file__).resolve().parent.parent / "data" / "candidate_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()