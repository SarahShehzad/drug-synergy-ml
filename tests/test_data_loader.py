import pandas as pd
import pytest
from src.data_loader import (
    parse_conc_list,
    resolve_drug_name,
    merge_metadata_and_responses,
)

# test 1
class TestParseConcList:
    def test_parses_normal_dilution_series(self):
        assert parse_conc_list("10.0,2.0,0.4,0.08,0.016,0.0032") == [
            10.0, 2.0, 0.4, 0.08, 0.016, 0.0032
        ]

    def test_handles_trailing_comma(self):
        assert parse_conc_list("10.0,2.0,") == [10.0, 2.0]

    def test_empty_string_returns_empty_list(self):
        assert parse_conc_list("") == []

# test 2
class TestResolveDrugName:
    def test_returns_name_when_present(self):
        assert resolve_drug_name("Selumetinib", "NCGC00189073-02") == "Selumetinib"

    def test_falls_back_to_sid_when_name_missing(self):
        """This is the exact bug you hit with BlockId 9 -- RowName was
        NaN, RowSid was a real compound ID."""
        assert resolve_drug_name(float("nan"), "NCGC00021305-06") == "NCGC00021305-06"

    def test_returns_none_when_both_missing(self):
        assert resolve_drug_name(float("nan"), float("nan")) is None

    def test_treats_blank_string_as_missing(self):
        assert resolve_drug_name("   ", "NCGC00021305-06") == "NCGC00021305-06"

# test 3
class TestMergeMetadataAndResponses:
    def test_resolves_concentration_by_row_col_index(self):
        metadata = pd.DataFrame({
            "BlockId": [1],
            "RowName": ["DrugA"], "ColName": ["DrugB"],
            "RowTarget": ["Target A"], "ColTarget": ["Target B"],
            "RowConcs": ["10.0,2.0,0.4"],
            "ColConcs": ["5.0,1.0,0.2"],
        })
        responses = pd.DataFrame({
            "BlockId": [1, 1],
            "Row": [1, 2],
            "Col": [1, 3],
            "Value": [50.0, 60.0],
            "Replicate": [1, 1],
        })

        merged = merge_metadata_and_responses(metadata, responses)

        # Row=1 -> first value in RowConcs (10.0); Col=1 -> first in ColConcs (5.0)
        assert merged.iloc[0]["conc_a"] == 10.0
        assert merged.iloc[0]["conc_b"] == 5.0
        # Row=2 -> second value (2.0); Col=3 -> third value (0.2)
        assert merged.iloc[1]["conc_a"] == 2.0
        assert merged.iloc[1]["conc_b"] == 0.2

    def test_merge_survives_blockid_dtype_mismatch(self):
        """metadata reads BlockId as int, responses as string (or vice
        versa) -- a naive merge would silently produce all-null rows."""
        metadata = pd.DataFrame({
            "BlockId": [1],
            "RowName": ["DrugA"], "ColName": ["DrugB"],
            "RowTarget": ["T1"], "ColTarget": ["T2"],
            "RowConcs": ["10.0"], "ColConcs": ["5.0"],
        })
        responses = pd.DataFrame({
            "BlockId": ["1"],
            "Row": [1], "Col": [1], "Value": [42.0], "Replicate": [1],
        })

        merged = merge_metadata_and_responses(metadata, responses)

        assert merged.iloc[0]["RowName"] == "DrugA"
        assert not pd.isna(merged.iloc[0]["conc_a"])