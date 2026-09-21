"""Unit tests for src/model.py

These focus on contract/behavior (does train->predict->evaluate work,
does it fail loudly when misused) rather than asserting specific
accuracy numbers, since accuracy on synthetic data isn't meaningful
to hard-code as a pass/fail threshold.
"""

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.features import build_feature_matrix
from src.model import (
    KAGGLE_COLUMN_MAP,
    LABEL_COLUMN,
    BatteryFaultClassifier,
    generate_synthetic_dataset,
    load_kaggle_dataset,
    train_test_split_by_battery,
)


@pytest.fixture(scope="module")
def labeled_data():
    raw = generate_synthetic_dataset(n_samples=500, seed=1)
    return build_feature_matrix(raw)


@pytest.fixture(scope="module")
def train_test_split_data(labeled_data):
    return train_test_split(
        labeled_data, test_size=0.25, random_state=1, stratify=labeled_data[LABEL_COLUMN]
    )


class TestBatteryFaultClassifier:
    def test_predict_before_train_raises(self):
        clf = BatteryFaultClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(generate_synthetic_dataset(n_samples=10))

    def test_save_before_train_raises(self, tmp_path):
        clf = BatteryFaultClassifier()
        with pytest.raises(RuntimeError):
            clf.save(str(tmp_path / "model.joblib"))

    def test_train_and_predict(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        clf = BatteryFaultClassifier()
        clf.train(train_df)

        predictions = clf.predict(test_df)
        assert len(predictions) == len(test_df)

    def test_evaluate_returns_expected_keys(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        clf = BatteryFaultClassifier()
        clf.train(train_df)

        results = clf.evaluate(test_df)
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                    "confusion_matrix", "critical_fault_false_negative_rate"]:
            assert key in results

    def test_evaluate_accuracy_in_valid_range(self, train_test_split_data):
        train_df, test_df = train_test_split_data
        clf = BatteryFaultClassifier()
        clf.train(train_df)

        results = clf.evaluate(test_df)
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["critical_fault_false_negative_rate"] <= 1.0

    def test_save_and_load_roundtrip(self, train_test_split_data, tmp_path):
        train_df, test_df = train_test_split_data
        clf = BatteryFaultClassifier()
        clf.train(train_df)

        model_path = tmp_path / "model.joblib"
        clf.save(str(model_path))

        reloaded = BatteryFaultClassifier()
        reloaded.load(str(model_path))

        original_preds = clf.predict(test_df)
        reloaded_preds = reloaded.predict(test_df)
        assert list(original_preds) == list(reloaded_preds)

    def test_load_missing_file_raises(self, tmp_path):
        clf = BatteryFaultClassifier()
        with pytest.raises(FileNotFoundError):
            clf.load(str(tmp_path / "does_not_exist.joblib"))


class TestLoadKaggleDataset:
    def test_renames_columns_correctly(self, tmp_path):
        csv_path = tmp_path / "sample.csv"
        raw = pd.DataFrame({
            "Voltage_measured": [3.7, 3.8],
            "Current_measured": [-2.0, -2.1],
            "Temperature_measured": [25.0, 26.0],
            "SoC": [90.0, 89.0],
            "cycle_number": [1, 1],
            "battery_id": ["B0001", "B0001"],
            "SoH": [85.0, 85.0],
        })
        raw.to_csv(csv_path, index=False)

        loaded = load_kaggle_dataset(str(csv_path))
        for renamed_col in KAGGLE_COLUMN_MAP.values():
            assert renamed_col in loaded.columns
        # Original raw column names should no longer be present
        assert "Voltage_measured" not in loaded.columns

    def test_missing_column_raises(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"Voltage_measured": [3.7]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            load_kaggle_dataset(str(csv_path))


class TestTrainTestSplitByBattery:
    def test_no_battery_overlap_between_train_and_test(self):
        df = pd.DataFrame({
            "battery_id": ["B1"] * 10 + ["B2"] * 10 + ["B3"] * 10 + ["B4"] * 10,
            "voltage": range(40),
            "current": [-2.0] * 40,
            "temperature": [25.0] * 40,
            "fault_type": ["normal"] * 40,
        })
        train_df, test_df = train_test_split_by_battery(df, test_size=0.25, random_state=1)

        train_batteries = set(train_df["battery_id"].unique())
        test_batteries = set(test_df["battery_id"].unique())
        assert train_batteries.isdisjoint(test_batteries)

    def test_falls_back_to_row_split_without_battery_id(self):
        df = pd.DataFrame({
            "voltage": range(20),
            "current": [-2.0] * 20,
            "temperature": [25.0] * 20,
            "fault_type": ["normal"] * 20,
        })
        train_df, test_df = train_test_split_by_battery(df, test_size=0.25, random_state=1)
        assert len(train_df) + len(test_df) == len(df)


class TestSyntheticDataset:
    def test_generates_requested_row_count(self):
        df = generate_synthetic_dataset(n_samples=100)
        assert len(df) == 100

    def test_has_expected_columns(self):
        df = generate_synthetic_dataset(n_samples=50)
        for col in ["voltage", "current", "temperature", "timestamp"]:
            assert col in df.columns

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_dataset(n_samples=50, seed=7)
        df2 = generate_synthetic_dataset(n_samples=50, seed=7)
        pd_testing_equal = df1.equals(df2)
        assert pd_testing_equal
