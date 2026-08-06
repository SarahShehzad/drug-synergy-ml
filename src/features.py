"""
Turns a drug name/SMILES into a numeric feature vector (Morgan fingerprint)
that a model can consume, and combines two drugs' fingerprints into a
single pair-level feature vector.
"""

import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_fingerprint(smiles, n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return np.zeros(n_bits, dtype=int)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def pair_features(smiles_a: str, smiles_b: str, cell_line_is_tumor: bool, n_bits: int = 1024) -> np.ndarray:
    """
    Combines two drug fingerprints (order-invariant, via elementwise max
    so Drug A + Drug B == Drug B + Drug A) plus a cell-line-type flag.
    """
    fp_a = smiles_to_fingerprint(smiles_a, n_bits)
    fp_b = smiles_to_fingerprint(smiles_b, n_bits)
    combined = np.maximum(fp_a, fp_b)  # order-invariant combination
    cell_line_flag = np.array([int(cell_line_is_tumor)])
    return np.concatenate([combined, cell_line_flag])


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
