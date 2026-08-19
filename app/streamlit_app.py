"""
Analyst dashboard for browsing drug synergy predictions.

Requires the API server running separately in another terminal:
    uvicorn src.api:app --reload

Run this dashboard with:
    streamlit run app/streamlit_app.py
"""

#Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
import streamlit as st

from src.evaluate_holdout import run_holdout_evaluation
from src.db import get_connection

import requests
import pandas as pd
import streamlit as st

from src.db import get_connection


API_URL = "http://127.0.0.1:8000"
SORT_COLUMNS = {"CMRS score": "cmrs_score", "Bliss score": "bliss_score"}

st.set_page_config(page_title="NF1 Drug Synergy Dashboard", page_icon="assets/NF1dashboardIcon.png", layout="wide")
st.title("Neurofibromatosis Type 1 Drug Synergy Analyst Dashboard")
st.caption(
    "Own ETL | CMRS & Bliss reproduction | Trained model, extending "
    "Zhou, Shehzad, Ahmed et al. (2025), bioRxiv 2025.08.02.667047"
)

conn = get_connection()

# Basic info 
total_pairs = conn.execute(
    "SELECT COUNT(*) FROM synergy_scores WHERE cmrs_score IS NOT NULL"
).fetchone()[0]
avg_cmrs = conn.execute(
    "SELECT AVG(cmrs_score) FROM synergy_scores WHERE cmrs_score IS NOT NULL"
).fetchone()[0]
n_cell_lines = conn.execute("SELECT COUNT(*) FROM cell_lines").fetchone()[0]
n_with_smiles = conn.execute("SELECT COUNT(*) FROM drugs WHERE smiles IS NOT NULL").fetchone()[0]
n_drugs_total = conn.execute("SELECT COUNT(*) FROM drugs").fetchone()[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Drug pairs scored", total_pairs)
col2.metric("Avg CMRS score", f"{avg_cmrs:.1f}" if avg_cmrs else "N/A")
col3.metric("Cell lines", n_cell_lines)
col4.metric("Drugs w/ structures", f"{n_with_smiles}/{n_drugs_total}")

st.divider()

# Browse & filter
st.subheader("Browse ranked drug pairs")

cell_lines = [r[0] for r in conn.execute("SELECT name FROM cell_lines WHERE is_tumor = 1").fetchall()]
fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
selected_line = fcol1.selectbox("Cell line", cell_lines)
search_term = fcol2.text_input("Search drug name", "")
sort_label = fcol3.selectbox("Sort by", list(SORT_COLUMNS.keys()))
sort_col = SORT_COLUMNS[sort_label]

query = f"""
SELECT sc.cmrs_score, sc.bliss_score, da.name AS drug_a, da.moa AS moa_a,
       db_.name AS drug_b, db_.moa AS moa_b
FROM synergy_scores sc
JOIN cell_lines cl ON sc.cell_line_id = cl.cell_line_id
JOIN drugs da ON sc.drug_a_id = da.drug_id
JOIN drugs db_ ON sc.drug_b_id = db_.drug_id
WHERE cl.name = ?
ORDER BY sc.{sort_col} DESC
"""
df = pd.read_sql_query(query, conn, params=(selected_line,))

if search_term:
    mask = (
        df["drug_a"].str.contains(search_term, case=False, na=False)
        | df["drug_b"].str.contains(search_term, case=False, na=False)
    )
    df = df[mask]

df.insert(0, "rank", range(1, len(df) + 1))

display_df = df.rename(columns={
    "rank": "Rank",
    "cmrs_score": "CMRS Score",
    "bliss_score": "Bliss Score",
    "drug_a": "Drug A",
    "moa_a": "Mechanism (A)",
    "drug_b": "Drug B",
    "moa_b": "Mechanism (B)",
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# CMRS vs Bliss comparison (scatter plot)
st.subheader("CMRS vs Bliss: do they agree?")
st.caption(
    "They anti-correlate. CMRS measures tumor-selectivity, Bliss measures "
    "raw interaction magnitude. See README for the full finding."
)
compare_df = df[["cmrs_score", "bliss_score"]].dropna()
if not compare_df.empty:
    st.scatter_chart(compare_df, x="bliss_score", y="cmrs_score")
    corr = compare_df["cmrs_score"].corr(compare_df["bliss_score"], method="spearman")
    st.write(f"Spearman correlation (this cell line, current filter): **{corr:.3f}**")

st.divider()

# Models performance (held-out validation)
st.subheader("Model performance: genuinely held-out pairs")
st.caption(
    "Leave-pair-out cross-validation. The model never trained on the "
    "specific pair it's predicting here. This approach "
    "generalizes. The numbers here pool both"
    "tumor cell lines together (1560 pairs)."
)

@st.cache_resource
def get_holdout_results():
    return run_holdout_evaluation()

with st.spinner("Running held-out evaluation (cached after first load)..."):
    holdout = get_holdout_results()

if "error" in holdout:
    st.warning(holdout["error"])
else:
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Spearman correlation", f"{holdout['spearman']:.3f}")
    mcol2.metric("Top-10 precision", f"{holdout['top10_precision']}/10")
    mcol3.metric("Pairs evaluated", holdout["n_pairs"])

    st.write("**True top 10 (green = also in the model's held-out predicted top 10):**")

    true_df_full = pd.DataFrame(holdout["true_top10"])
    overlap_flags = true_df_full["in_both"].tolist()

    true_df = true_df_full.drop(columns=["in_both"]).rename(columns={
        "pair": "Pair", "cell_line": "Cell Line",
        "actual_rank": "Actual Rank", "predicted_rank": "Predicted Rank (Held-Out)",
    })

    def style_overlap(row):
        color = "background-color: rgba(46, 160, 67, 0.25)" if overlap_flags[row.name] else ""
        return [color] * len(row)

    st.dataframe(true_df.style.apply(style_overlap, axis=1), use_container_width=True, hide_index=True)
    st.write("**Known top combinations. Predicted without training on them:**")
    combo_df = pd.DataFrame(holdout["known_combos"])
    combo_display = combo_df.rename(columns={
    "pair": "Drug Pair",
    "actual_rank": "Actual Rank",
    "predicted_rank": "Predicted Rank (Held-Out)",
    })
    st.dataframe(combo_display, use_container_width=True, hide_index=True)

st.divider()

# Exploratory: predictions on entirely novel compounds 
st.subheader("Exploratory: predictions on novel compounds")
st.caption(
    "These drugs were never screened in this dataset at all. They were paired "
    "against proven top performers by shared mechanism of action, using "
    "structures pulled live from PubChem. "
    "**Treat these "
    "as lower-confidence, exploratory signal, not validated results.**"
)

candidates_path = Path(__file__).resolve().parent.parent / "data" / "candidate_predictions.csv"
if candidates_path.exists():
    candidates_df = pd.read_csv(candidates_path)
    candidates_display = candidates_df.rename(columns={
        "anchor": "Anchor Drug (known)",
        "candidate": "Novel Compound",
        "shared_moa": "Shared Mechanism",
        "predicted_cmrs": "Predicted CMRS Score",
    })
    st.dataframe(candidates_display, use_container_width=True, hide_index=True)
    st.write(
        "Notably, the top three predictions were all proteasome inhibitors "
        "paired with Carfilzomib, a mechanism the original paper's authors "
        "independently flagged as a recurring strong performer, arrived at "
        "here from structural/mechanism features."
    )
else:
    st.info("Run `python -m src.explore_new_compounds` to generate this data.")

# Predicts a new pair via the live API 
st.subheader("Predict a new pair")
st.caption("Calls the FastAPI /predict endpoint running separately.")

all_drug_names = [r[0] for r in conn.execute("SELECT name FROM drugs ORDER BY name").fetchall()]
conn.close()

pcol1, pcol2 = st.columns(2)
drug_a_input = pcol1.selectbox("Drug A", all_drug_names, key="drug_a_select")
drug_b_input = pcol2.selectbox(
    "Drug B", all_drug_names, index=min(1, len(all_drug_names) - 1), key="drug_b_select"
)

if st.button("Predict"):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"drug_a": drug_a_input, "drug_b": drug_b_input, "is_tumor": True},
            timeout=5,
        )
        if resp.status_code == 200:
            result = resp.json()
            st.metric("Predicted CMRS score", result["predicted_cmrs"])
            if not result["both_drugs_have_structures"]:
                st.warning(
                    "One or both drugs are missing a real molecular structure. "
                    "Prediction may be unreliable."
                )
            st.write("**Analysis:**")
            analysis_conn = get_connection()

            actual_rows = analysis_conn.execute(
                """
                SELECT sc.cmrs_score, sc.bliss_score, cl.name
                FROM synergy_scores sc
                JOIN drugs da ON sc.drug_a_id = da.drug_id
                JOIN drugs db_ ON sc.drug_b_id = db_.drug_id
                JOIN cell_lines cl ON sc.cell_line_id = cl.cell_line_id
                WHERE cl.is_tumor = 1
                  AND ((da.name = ? AND db_.name = ?) OR (da.name = ? AND db_.name = ?))
                """,
                (drug_a_input, drug_b_input, drug_b_input, drug_a_input),
            ).fetchall()

            if actual_rows:
                for actual_cmrs, actual_bliss, cell_line_name in actual_rows:
                    diff = result["predicted_cmrs"] - actual_cmrs
                    bliss_note = f" Real Bliss score: {actual_bliss:.2f}." if actual_bliss is not None else ""
                    st.write(
                        f"- In **{cell_line_name}**, this pair was already screened: "
                        f"real CMRS score **{actual_cmrs:.0f}**, model predicted "
                        f"**{result['predicted_cmrs']:.1f}** "
                        f"({'+' if diff >= 0 else ''}{diff:.1f} off).{bliss_note}"
                    )
                st.caption(
                    "This panel was tested exhaustively, so most pairs picked here were "
                    "already screened. This is the model recalling a known result, not "
                    "a genuine new prediction. See the exploratory section below for "
                    "predictions on compounds never screened at all."
                )
            else:
                st.info("This exact pair wasn't found in the screened dataset for either tumor cell line.")

            all_scores = pd.read_sql_query(
                "SELECT cmrs_score FROM synergy_scores WHERE cmrs_score IS NOT NULL", analysis_conn
            )["cmrs_score"]
            percentile = (all_scores < result["predicted_cmrs"]).mean() * 100
            st.write(
                f"- A score of **{result['predicted_cmrs']:.1f}** falls in the "
                f"**{percentile:.0f}th percentile** of all {len(all_scores)} historically scored pairs."
            )
            analysis_conn.close()
        else:
            st.error(f"API error: {resp.json().get('detail', resp.text)}")
    except requests.exceptions.ConnectionError:
        st.error("Couldn't reach the API. Is `uvicorn src.api:app --reload` running in another terminal?")