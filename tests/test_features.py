"""Unit tests for src/features.py"""

import pandas as pd
import pytest

from src.features import (
    CAPACITY_DEGRADED_SOH,
    OVERCHARGE_VOLTAGE,
    OVERHEAT_TEMP,
    OVER_DISCHARGE_VOLTAGE,
    build_feature_matrix,
    engineer_features,
    label_faults,
)


@pytest.fixture
def sample_df():
    """Simple ungrouped data (no battery_id/cycle_number) -- exercises
    the fallback single-group behavior used by e.g. the live API."""
    return pd.DataFrame({
        "voltage": [3.7, 3.8, 3.9, 3.85, 3.8],
        "current": [-2.0, -2.1, -2.0, -1.9, -2.0],
        "temperature": [25.0, 26.0, 27.0, 26.5, 26.0],
    })


@pytest.fixture
def grouped_df():
    """Data shaped like the real dataset: two cycles for one battery."""
    return pd.DataFrame({
        "battery_id": ["B0001"] * 3 + ["B0001"] * 3,
        "cycle_number": [1, 1, 1, 2, 2, 2],
        "voltage": [4.0, 3.9, 3.8, 4.1, 4.0, 3.9],
        "current": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        "temperature": [20.0, 21.0, 22.0, 20.0, 21.0, 22.0],
    })


class TestEngineerFeatures:
    def test_adds_expected_columns(self, sample_df):
        out = engineer_features(sample_df)
        for col in ["voltage_delta", "current_delta", "temp_delta",
                    "voltage_roll_mean", "temp_roll_mean", "temp_roll_std"]:
            assert col in out.columns

    def test_first_row_delta_is_zero(self, sample_df):
        out = engineer_features(sample_df)
        assert out["voltage_delta"].iloc[0] == 0.0

    def test_delta_computed_correctly(self, sample_df):
        out = engineer_features(sample_df)
        # second row: 3.8 - 3.7 = 0.1
        assert out["voltage_delta"].iloc[1] == pytest.approx(0.1)

    def test_does_not_mutate_input(self, sample_df):
        original = sample_df.copy()
        engineer_features(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_grouped_data_does_not_leak_across_cycles(self, grouped_df):
        out = engineer_features(grouped_df)
        # Row 3 is the first row of cycle 2 -- its delta must be 0
        # (fresh cycle), NOT (4.1 - 3.8) which would be cross-cycle leakage.
        first_row_of_cycle_2 = out.iloc[3]
        assert first_row_of_cycle_2["voltage_delta"] == 0.0

    def test_grouped_data_deltas_correct_within_cycle(self, grouped_df):
        out = engineer_features(grouped_df)
        # second row of cycle 1: 3.9 - 4.0 = -0.1
        assert out["voltage_delta"].iloc[1] == pytest.approx(-0.1)


class TestLabelFaults:
    def test_normal_reading_labeled_normal(self):
        df = pd.DataFrame({"voltage": [3.7], "current": [-2.0], "temperature": [25.0]})
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "normal"

    def test_overcharge_detected(self):
        df = pd.DataFrame({"voltage": [OVERCHARGE_VOLTAGE + 0.1], "current": [-2.0], "temperature": [25.0]})
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "overcharge"

    def test_over_discharge_detected(self):
        df = pd.DataFrame({"voltage": [OVER_DISCHARGE_VOLTAGE - 0.1], "current": [-2.0], "temperature": [25.0]})
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "over_discharge"

    def test_overheat_detected(self):
        df = pd.DataFrame({"voltage": [3.7], "current": [-2.0], "temperature": [OVERHEAT_TEMP + 5]})
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "overheat"

    def test_overheat_takes_priority_over_overcharge(self):
        # Reading that satisfies both overheat and overcharge thresholds
        df = pd.DataFrame({
            "voltage": [OVERCHARGE_VOLTAGE + 0.1],
            "current": [-2.0],
            "temperature": [OVERHEAT_TEMP + 5],
        })
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "overheat"

    def test_capacity_degraded_detected_when_soh_present(self):
        df = pd.DataFrame({
            "voltage": [3.7], "current": [-2.0], "temperature": [25.0],
            "SoH": [CAPACITY_DEGRADED_SOH - 5],
        })
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "capacity_degraded"

    def test_capacity_degraded_skipped_when_soh_column_absent(self):
        # Should not raise, and should not label anything as capacity_degraded
        df = pd.DataFrame({"voltage": [3.7], "current": [-2.0], "temperature": [25.0]})
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "normal"

    def test_overheat_takes_priority_over_capacity_degraded(self):
        df = pd.DataFrame({
            "voltage": [3.7], "current": [-2.0], "temperature": [OVERHEAT_TEMP + 5],
            "SoH": [CAPACITY_DEGRADED_SOH - 5],
        })
        out = label_faults(df)
        assert out["fault_type"].iloc[0] == "overheat"


class TestBuildFeatureMatrix:
    def test_pipeline_produces_features_and_labels(self, sample_df):
        out = build_feature_matrix(sample_df)
        assert "fault_type" in out.columns
        assert "voltage_delta" in out.columns
        assert len(out) == len(sample_df)

    def test_pipeline_works_on_grouped_data(self, grouped_df):
        out = build_feature_matrix(grouped_df)
        assert "fault_type" in out.columns
        assert len(out) == len(grouped_df)
