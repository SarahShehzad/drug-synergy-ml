"""
Minimal demo: pick two drugs + a cell line type, get a predicted synergy
score from the trained baseline model.

Usage:
    streamlit run app/streamlit_app.py
"""

import pickle
from pathlib import Path

import streamlit as st

from src.db import get_connection
from src.features import pair_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_model.pkl"

st.set_page_config(page_title="Drug Synergy Predictor", page_icon="🧬")
st.title("ML-Powered Drug Synergy Predictor")
st.caption(
    "Predicts a CMRS-style synergy score for drug pairs, extending "
    "Zhou, Shehzad, Ahmed et al. (2025), bioRxiv 2025.08.02.667047"
)

if not MODEL_PATH.exists():
    st.warning("No trained model found yet. Run `python -m src.train_baseline` first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

conn = get_connection()
drug_names = [r[0] for r in conn.execute("SELECT name FROM drugs ORDER BY name").fetchall()]
conn.close()

if not drug_names:
    st.warning("No drugs in the database yet. Run `python -m src.data_loader` first.")
    st.stop()

col1, col2 = st.columns(2)
drug_a = col1.selectbox("Drug A", drug_names)
drug_b = col2.selectbox("Drug B", drug_names, index=min(1, len(drug_names) - 1))
is_tumor = st.checkbox("Tumor cell line (vs. non-tumor reference)", value=True)

if st.button("Predict synergy"):
    conn = get_connection()
    smiles_a = conn.execute("SELECT smiles FROM drugs WHERE name = ?", (drug_a,)).fetchone()[0]
    smiles_b = conn.execute("SELECT smiles FROM drugs WHERE name = ?", (drug_b,)).fetchone()[0]
    conn.close()

    features = pair_features(smiles_a, smiles_b, is_tumor).reshape(1, -1)
    prediction = model.predict(features)[0]

    st.metric("Predicted synergy score", f"{prediction:.2f}")
    st.caption("Higher scores indicate greater predicted tumor-selective effect, matching the CMRS 0-6 scale.")
