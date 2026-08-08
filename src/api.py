import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.db import get_connection
from src.features import pair_features, get_moa_vocabulary

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline_model.pkl"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts -- load the model and build the
    # mechanism-of-action vocabulary a single time, not per-request.
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            state["model"] = pickle.load(f)
    else:
        state["model"] = None

    conn = get_connection()
    all_moas = [r[0] for r in conn.execute("SELECT moa FROM drugs WHERE moa IS NOT NULL").fetchall()]
    state["moa_vocab"] = get_moa_vocabulary(all_moas)
    conn.close()

    yield
    state.clear()


app = FastAPI(title="Drug Synergy Predictor API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    drug_a: str
    drug_b: str
    is_tumor: bool = True


class PredictionResponse(BaseModel):
    drug_a: str
    drug_b: str
    predicted_cmrs: float
    both_drugs_have_structures: bool


def find_drug(conn, name: str):
    row = conn.execute(
        "SELECT name, smiles, moa FROM drugs WHERE name LIKE ?", (f"%{name}%",)
    ).fetchone()
    return row


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": state.get("model") is not None}


@app.get("/drugs")
def list_drugs():
    conn = get_connection()
    rows = conn.execute("SELECT name FROM drugs ORDER BY name").fetchall()
    conn.close()
    return {"drugs": [r[0] for r in rows]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded -- run src.train_baseline first.")

    conn = get_connection()
    drug_a_row = find_drug(conn, request.drug_a)
    drug_b_row = find_drug(conn, request.drug_b)
    conn.close()

    if drug_a_row is None:
        raise HTTPException(status_code=404, detail=f"Drug not found: {request.drug_a}")
    if drug_b_row is None:
        raise HTTPException(status_code=404, detail=f"Drug not found: {request.drug_b}")

    name_a, smiles_a, moa_a = drug_a_row
    name_b, smiles_b, moa_b = drug_b_row
    both_have_structures = smiles_a is not None and smiles_b is not None

    features = pair_features(
        smiles_a, smiles_b, request.is_tumor,
        moa_a=moa_a, moa_b=moa_b, moa_vocab=state["moa_vocab"],
    ).reshape(1, -1)

    prediction = state["model"].predict(features)[0]

    return PredictionResponse(
        drug_a=name_a,
        drug_b=name_b,
        predicted_cmrs=round(float(prediction), 1),
        both_drugs_have_structures=both_have_structures,
    )