"""
Turns a drug name/SMILES into a numeric feature vector (Morgan fingerprint)
that a model can consume, and combines two drugs' fingerprints into a
single pair-level feature vector.
"""

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_moa_vocabulary(moa_values) -> list[str]:
    """Builds a fixed list of known mechanism-of-action strings from the
    dataset, so each one can be one-hot encoded consistently."""
    return sorted(set(m for m in moa_values if isinstance(m, str) and m.strip()))


def moa_to_onehot(moa, vocabulary: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocabulary), dtype=int)
    if moa in vocabulary:
        vec[vocabulary.index(moa)] = 1
    return vec

def smiles_to_fingerprint(smiles, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return np.zeros(n_bits, dtype=int)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def pair_features(
    smiles_a: str, smiles_b: str, cell_line_is_tumor: bool,
    moa_a: str = None, moa_b: str = None, moa_vocab: list[str] = None,
    n_bits: int = 1024,
) -> np.ndarray:
    fp_a = smiles_to_fingerprint(smiles_a, n_bits)
    fp_b = smiles_to_fingerprint(smiles_b, n_bits)
    combined = np.maximum(fp_a, fp_b)
    cell_line_flag = np.array([int(cell_line_is_tumor)])

    parts = [combined, cell_line_flag]
    if moa_vocab is not None:
        moa_a_vec = moa_to_onehot(moa_a, moa_vocab)
        moa_b_vec = moa_to_onehot(moa_b, moa_vocab)
        parts.append(np.maximum(moa_a_vec, moa_b_vec))

    return np.concatenate(parts)


def build_feature_matrix(pairs: list[dict]) -> np.ndarray:
    """
    pairs: list of dicts, each with keys 'smiles_a', 'smiles_b', 'is_tumor'
    Returns a 2D numpy array ready for sklearn/xgboost.
    """
    rows = [
        pair_features(p["smiles_a"], p["smiles_b"], p["is_tumor"])
        for p in pairs
    ]
    return np.vstack(rows)
