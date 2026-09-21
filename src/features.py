"""
features.py
-----------
Feature engineering and rule-based fault labeling for battery telemetry.

Two data shapes are supported:
1. Real dataset (e.g. the Kaggle/NASA-derived battery_health_dataset.csv):
   rows are grouped by ('battery_id', 'cycle_number'), 20 readings per
   discharge cycle. Rolling/delta features MUST be computed within each
   group only -- otherwise a feature would incorrectly compute a "delta"
   between the last reading of one cycle and the first reading of the
   next, which are unrelated points in time.
2. Simple/live data (e.g. a short window from the API, or the synthetic
   generator): no battery_id/cycle_number columns exist, so the whole
   DataFrame is treated as a single group -- this is the same behavior
   the original version of this module had.

engineer_features() auto-detects which case applies.

Two responsibilities live here:
1. engineer_features(): derive rolling/rate-of-change features the
   classifier will use.
2. label_faults(): apply documented threshold rules to generate a
   'fault_type' ground-truth column. Thresholds are calibrated against
   the real dataset's observed distribution (see docs/SRS.md, section 5)
   plus one industry-standard rule (80% SoH = end-of-life), not just
   made-up numbers.
"""

import pandas as pd

GROUP_COLS = ["battery_id", "cycle_number"]

# --- Fault thresholds (documented in SRS.md section 5; keep in sync) ---
#
# These were recalibrated against the real battery_health_dataset.csv
# distribution (NASA PCoE Li-ion discharge data, 24 batteries, ~29K rows):
#   Voltage range observed:      2.73V - 4.23V
#   Temperature range observed:  4.7C  - 66.7C
#   Current range observed:      -4.03A - -0.89A  (discharge-only, always negative)
#
# IMPORTANT DATASET LIMITATION (documented honestly, not hidden):
# This dataset contains ONLY discharge cycles, never charging. That means:
#   - `overcharge` (a charging-phase fault) will essentially never fire
#     on this dataset, because voltage only ever decreases from a
#     starting value during discharge -- it's kept in the code for
#     generality (e.g. if you later add a charging dataset) but you
#     should expect ~0 occurrences here and say so in your report.
#   - `short_circuit_indication` relies on an abrupt current jump; the
#     max observed within-cycle current delta in this dataset is only
#     1.61A (discharge current is close to constant), so this rule is
#     also expected to rarely/never fire here. This is a property of
#     the dataset, not a bug in the rule.
OVERCHARGE_VOLTAGE = 4.25           # volts (see limitation note above)
OVER_DISCHARGE_VOLTAGE = 3.0        # volts -- common conservative Li-ion low-voltage
                                     # cutoff used by real BMS designs (dataset's own
                                     # 6th percentile is ~2.96V, consistent with this)
OVERHEAT_TEMP = 55.0                # degrees C -- dataset's 95th percentile is ~54.9C;
                                     # chosen as a conservative early-warning threshold
                                     # below the point where thermal runaway risk rises
SHORT_CIRCUIT_CURRENT = 3.0         # amps jump between consecutive readings (see limitation note)
CAPACITY_DEGRADED_SOH = 80.0        # percent -- industry-standard Li-ion end-of-life
                                     # definition (cell considered degraded once capacity
                                     # fades to 80% of rated capacity)


def _has_group_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in GROUP_COLS)


def engineer_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Adds derived features to a telemetry DataFrame.

    If the DataFrame has 'battery_id' and 'cycle_number' columns, rolling
    and delta features are computed independently within each
    (battery_id, cycle_number) group so nothing leaks across cycle
    boundaries. Otherwise, the whole DataFrame is treated as one
    continuous sequence (matches simple/live API usage).

    New columns:
        voltage_delta       - change in voltage from previous row (within group)
        current_delta       - change in current from previous row (within group)
        temp_delta           - change in temperature from previous row (within group, gradient)
        voltage_roll_mean   - rolling mean of voltage over `window` samples (within group)
        temp_roll_mean       - rolling mean of temperature over `window` samples (within group)
        temp_roll_std         - rolling std of temperature (spikiness indicator, within group)
    """
    out = df.copy()

    if _has_group_columns(out):
        grouped = out.groupby(GROUP_COLS, sort=False)
        out["voltage_delta"] = grouped["voltage"].diff().fillna(0.0)
        out["current_delta"] = grouped["current"].diff().fillna(0.0)
        out["temp_delta"] = grouped["temperature"].diff().fillna(0.0)

        out["voltage_roll_mean"] = grouped["voltage"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out["temp_roll_mean"] = grouped["temperature"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        out["temp_roll_std"] = grouped["temperature"].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0.0)
        )
    else:
        out["voltage_delta"] = out["voltage"].diff().fillna(0.0)
        out["current_delta"] = out["current"].diff().fillna(0.0)
        out["temp_delta"] = out["temperature"].diff().fillna(0.0)

        out["voltage_roll_mean"] = out["voltage"].rolling(window, min_periods=1).mean()
        out["temp_roll_mean"] = out["temperature"].rolling(window, min_periods=1).mean()
        out["temp_roll_std"] = out["temperature"].rolling(window, min_periods=1).std().fillna(0.0)

    return out


def label_faults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rule-based labeling to produce a ground-truth 'fault_type' column.

    Priority order (checked first-to-last; a row keeps the first match):
        1. overheat                  - immediate thermal safety risk
        2. short_circuit_indication  - immediate electrical safety risk
        3. over_discharge            - electrical stress / cell damage risk
        4. capacity_degraded         - long-term health signal (requires 'SoH' column;
                                        skipped gracefully if not present)
        5. overcharge                 - charging-phase fault (rarely applicable to
                                        discharge-only datasets, see module docstring)

    Fault types: 'normal', 'overcharge', 'over_discharge', 'overheat',
    'short_circuit_indication', 'capacity_degraded'
    """
    out = df.copy()
    out["fault_type"] = "normal"

    out.loc[out["temperature"] >= OVERHEAT_TEMP, "fault_type"] = "overheat"

    if "current_delta" in out.columns:
        spike_mask = out["current_delta"].abs() >= SHORT_CIRCUIT_CURRENT
        out.loc[spike_mask & (out["fault_type"] == "normal"), "fault_type"] = "short_circuit_indication"

    discharge_mask = out["voltage"] <= OVER_DISCHARGE_VOLTAGE
    out.loc[discharge_mask & (out["fault_type"] == "normal"), "fault_type"] = "over_discharge"

    if "SoH" in out.columns:
        degraded_mask = out["SoH"] < CAPACITY_DEGRADED_SOH
        out.loc[degraded_mask & (out["fault_type"] == "normal"), "fault_type"] = "capacity_degraded"

    overcharge_mask = out["voltage"] >= OVERCHARGE_VOLTAGE
    out.loc[overcharge_mask & (out["fault_type"] == "normal"), "fault_type"] = "overcharge"

    return out


def build_feature_matrix(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Convenience wrapper: engineer features and labels in one call."""
    featured = engineer_features(df, window=window)
    labeled = label_faults(featured)
    return labeled
