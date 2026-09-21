"""
model.py
--------
Fault classification model: training, evaluation, and persistence.

Uses a Random Forest classifier — chosen over deep learning approaches
because it (a) trains fast on tabular sensor data without needing a GPU,
(b) gives interpretable feature importances, useful for explaining
predictions in the project report, and (c) is easy to validate with
standard sklearn metrics, which keeps the "quality measurement" story
simple and defensible.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Column names as they appear in the raw Kaggle/NASA CSV -> our internal names.
KAGGLE_COLUMN_MAP = {
    "Voltage_measured": "voltage",
    "Current_measured": "current",
    "Temperature_measured": "temperature",
    # SoC, SoH, battery_id, cycle_number keep their original names --
    # SoH is used for fault labeling (see features.py), SoC/battery_id/
    # cycle_number are kept for grouping and analysis but are NOT model
    # input features (see note on FEATURE_COLUMNS below).

}

FEATURE_COLUMNS = [
    "voltage",
    "current",
    "temperature",
    "voltage_delta",
    "current_delta",
    "temp_delta",
    "voltage_roll_mean",
    "temp_roll_mean",
    "temp_roll_std",
]

LABEL_COLUMN = "fault_type"

# Faults where a missed detection (false negative) is safety-critical.
# Used to report a separate, stricter metric beyond overall accuracy.
# 'capacity_degraded' is a long-term health signal, not an acute safety
# risk, so it is deliberately excluded from this set.
CRITICAL_FAULTS = {"overheat", "short_circuit_indication"}

# Why SoC/SoH are NOT in FEATURE_COLUMNS even though they're in the raw
# dataset: SoH is used to *derive* the 'capacity_degraded' label in
# features.py::label_faults(), so using it as a model input as well
# would leak the answer directly into the input (the model would just
# learn "predict capacity_degraded when SoH < 80", which is trivial and
# not a real prediction). SoC is left out for a different, practical
# reason: it wouldn't always be available in a live deployment (the API
# in api.py only accepts voltage/current/temperature), and training a
# model on a feature the serving path can't supply would silently break
# at inference time. Keeping FEATURE_COLUMNS identical between offline
# training and the live API is a deliberate consistency choice worth
# mentioning in your report as a real MLOps quality consideration.


class BatteryFaultClassifier:
    """Wraps a RandomForestClassifier with domain-specific evaluation."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",  # fault classes are rare relative to 'normal'
        )
        self._is_trained = False

    def train(self, df: pd.DataFrame) -> None:
        """Trains the model on a labeled, feature-engineered DataFrame."""
        x = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN]
        self.model.fit(x, y)
        self._is_trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts fault_type for each row. df must contain FEATURE_COLUMNS."""
        if not self._is_trained:
            raise RuntimeError("Model has not been trained or loaded yet.")
        x = df[FEATURE_COLUMNS]
        return self.model.predict(x)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluates the model on a labeled test set and returns a metrics dict
        suitable for dropping directly into the project report.
        """
        x = df[FEATURE_COLUMNS]
        y_true = df[LABEL_COLUMN]
        y_pred = self.model.predict(x)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "confusion_matrix_labels": sorted(y_true.unique().tolist()),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }

        metrics["critical_fault_false_negative_rate"] = self._critical_fnr(y_true, y_pred)
        return metrics

    @staticmethod
    def _critical_fnr(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Of all samples that were TRULY a critical fault, what fraction did
        we predict as 'normal' or a non-critical class? This is the number
        that matters most for a safety-relevant system — report it
        separately from overall accuracy in your write-up.
        """
        y_pred_series = pd.Series(y_pred, index=y_true.index)
        critical_mask = y_true.isin(CRITICAL_FAULTS)
        n_critical = critical_mask.sum()
        if n_critical == 0:
            return 0.0
        missed = ((y_pred_series != y_true) & critical_mask).sum()
        return float(missed / n_critical)

    def save(self, path: str = "model.joblib") -> None:
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self.model, path)

    def load(self, path: str = "model.joblib") -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"No model file found at {path}")
        self.model = joblib.load(path)
        self._is_trained = True


def generate_synthetic_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic telemetry dataset for development and testing
    before you plug in a real dataset (e.g. NASA Li-ion Battery Dataset).

    This is NOT a substitute for real data in your final report — use it
    only to get the pipeline running end-to-end early, then swap in real
    CSV data via pandas.read_csv() once you've sourced it.
    """
    rng = np.random.default_rng(seed)

    voltage = rng.normal(3.7, 0.3, n_samples).clip(2.0, 4.5)
    current = rng.normal(2.0, 5.0, n_samples).clip(-40, 40)
    temperature = rng.normal(30.0, 8.0, n_samples).clip(-10, 90)

    # Inject a few artificial fault-like spikes so the dataset isn't trivially "all normal"
    n_fault_samples = max(3, int(n_samples * 0.08))  # ensure at least a few of each fault type
    fault_idx = rng.choice(n_samples, size=n_fault_samples, replace=False)
    third = n_fault_samples // 3

    overcharge_idx = fault_idx[:third]
    overheat_idx = fault_idx[third: 2 * third]
    discharge_idx = fault_idx[2 * third:]

    voltage[overcharge_idx] = rng.uniform(4.3, 4.6, len(overcharge_idx))
    temperature[overheat_idx] = rng.uniform(65, 95, len(overheat_idx))
    voltage[discharge_idx] = rng.uniform(1.8, 2.4, len(discharge_idx))

    df = pd.DataFrame({
        "timestamp": np.arange(n_samples),
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
    })
    return df


def load_kaggle_dataset(path: str) -> pd.DataFrame:
    """
    Loads the real battery_health_dataset.csv (NASA PCoE discharge data,
    as distributed on Kaggle) and renames columns to our internal schema.

    Expects raw columns: Voltage_measured, Current_measured,
    Temperature_measured, SoC, cycle_number, battery_id, SoH.
    """
    df = pd.read_csv(path)
    missing = set(KAGGLE_COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected column(s): {missing}. "
            f"Check the CSV matches the raw Kaggle/NASA schema."
        )
    return df.rename(columns=KAGGLE_COLUMN_MAP)


def train_test_split_by_battery(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data by battery_id (not randomly by row) so that all cycles
    from a given battery end up entirely in train OR entirely in test,
    never both.

    Why this matters: rows from the same battery/cycle are highly
    correlated (they're consecutive readings of the same physical decay
    process). A plain random row-level split would let the model
    effectively "see" near-identical neighboring readings from the same
    cycle during training and evaluate on their neighbors, which
    inflates accuracy and doesn't reflect how the model would perform on
    a genuinely unseen battery. Splitting by battery_id is the
    methodologically honest choice here and is worth explicitly calling
    out in your report as an ML-quality (not just software-quality)
    decision.
    """
    if "battery_id" not in df.columns:
        # Fallback for data without battery grouping (e.g. synthetic data)
        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[LABEL_COLUMN])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["battery_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


if __name__ == "__main__":
    # Run `python -m src.model` from the project root.
    #
    # By default this now trains on the REAL dataset if present at
    # data/battery_health_dataset.csv, falling back to synthetic data
    # only if the real file isn't found -- so this script keeps working
    # even before you've placed the dataset, but prefers real data.
    from pathlib import Path as _Path

    from src.features import build_feature_matrix

    real_data_path = _Path("data/battery_health_dataset.csv")

    if real_data_path.exists():
        print(f"Loading real dataset from {real_data_path} ...")
        raw = load_kaggle_dataset(str(real_data_path))
        labeled = build_feature_matrix(raw)
        train_df, test_df = train_test_split_by_battery(labeled, test_size=0.25, random_state=42)
        print(f"Train batteries: {train_df['battery_id'].nunique()}, "
              f"Test batteries: {test_df['battery_id'].nunique()}")
    else:
        print(f"No real dataset found at {real_data_path}, using synthetic data instead.")
        raw = generate_synthetic_dataset()
        labeled = build_feature_matrix(raw)
        train_df, test_df = train_test_split(
            labeled, test_size=0.2, random_state=42, stratify=labeled[LABEL_COLUMN]
        )

    print("\nLabel distribution (full dataset):")
    print(labeled[LABEL_COLUMN].value_counts())

    clf = BatteryFaultClassifier()
    clf.train(train_df)
    results = clf.evaluate(test_df)

    print("\nAccuracy:", results["accuracy"])
    print("Critical fault false negative rate:", results["critical_fault_false_negative_rate"])
    print(results["classification_report"])

    clf.save("model.joblib")
    print("\nModel saved to model.joblib")
