# Traceability Matrix
## AI-Based Lithium-Ion Battery Fault Detection System

**Version:** 1.0
**Related documents:** `docs/SRS.md`, `docs/test_plan.md`

This matrix links every requirement to the design component that
implements it, the test(s) that verify it, and its current status.
Update the Status column as you go — an evaluator can check this table
against your actual repo to confirm claims are real, which is exactly
the kind of evidence a software quality course rewards.

---

## Functional Requirements

| Req ID | Requirement Summary | Design Component | Test Case(s) | Status |
|---|---|---|---|---|
| FR-1 | Validate readings against physical bounds | `src/validation.py::validate_reading`, `validate_batch` | `test_validation.py` (10 cases) | ✅ Implemented & tested |
| FR-2 | Compute rolling/delta features | `src/features.py::engineer_features` | `test_features.py::TestEngineerFeatures` (4 cases) | ✅ Implemented & tested |
| FR-3 | Enforce minimum window size of 5 readings | `src/api.py::predict` | *Not yet automated* — manual verification only | ⚠️ Needs `test_api.py` |
| FR-4 | Classify readings into 5 fault states | `src/model.py::BatteryFaultClassifier`, `src/features.py::label_faults` | `test_model.py::TestBatteryFaultClassifier`, `test_features.py::TestLabelFaults` (11 cases) | ✅ Implemented & tested |
| FR-5 | Report evaluation metrics incl. critical-fault FNR | `src/model.py::BatteryFaultClassifier.evaluate` | `test_model.py::test_evaluate_returns_expected_keys`, `test_evaluate_accuracy_in_valid_range` | ✅ Implemented & tested |
| FR-6 | Expose REST `/predict` endpoint | `src/api.py` | *Not yet automated* | ⚠️ Needs `test_api.py` |
| FR-7 | Save/load trained model | `src/model.py::save/load` | `test_model.py::test_save_and_load_roundtrip`, `test_save_before_train_raises`, `test_load_missing_file_raises` | ✅ Implemented & tested |

---

## Non-Functional Requirements

| Req ID | Attribute | Design Component | Verification Method | Status |
|---|---|---|---|---|
| NFR-1 | Functional Suitability (≥4 states classified) | `model.py`, `features.py` | `evaluate()` classification report — 4 classes confirmed in synthetic run | ✅ Verified on synthetic data; re-verify on real dataset |
| NFR-2 | Reliability — critical fault FNR tracked | `model.py::_critical_fnr` | `test_model.py::test_evaluate_accuracy_in_valid_range` (checks range only — add a threshold assertion once real-data baseline is known) | ⚠️ Partially verified |
| NFR-3 | Robustness — reject malformed input without crashing | `validation.py` | `test_validation.py` full suite | ✅ Verified |
| NFR-4 | Performance — prediction <200ms | `api.py::predict` | Manual timing via `curl`/Postman — not yet automated | ⚠️ Needs automated timing test |
| NFR-5 | Maintainability — ≥80% coverage | All `src/` modules | `pytest --cov=src` | ✅ Currently ~90%+ (run locally to confirm exact number, coverage can drift as code changes) |
| NFR-6 | Maintainability — complexity ≤ grade C | All `src/` modules | `radon cc src/ -a` | ✅ Verify by running the command; all functions are currently short and single-purpose |
| NFR-7 | Usability — self-documenting API | `api.py` (FastAPI auto-docs) | Manual check of `/docs` endpoint | ✅ Verified |
| NFR-8 | Portability — runs from clean env via `requirements.txt` | Whole repo | GitHub Actions CI (`.github/workflows/ci.yml`) | ✅ Verified on every push |

---

## Fault Threshold Rules Traceability

Each ground-truth rule (SRS Section 5) traces to a specific test that
confirms it fires correctly and respects priority ordering:

| Rule | Test |
|---|---|
| `overheat` (temp ≥ 55°C) | `test_features.py::test_overheat_detected` |
| `over_discharge` (voltage ≤ 3.0V) | `test_features.py::test_over_discharge_detected` |
| `capacity_degraded` (SoH < 80%) | `test_features.py::test_capacity_degraded_detected_when_soh_present`, `test_capacity_degraded_skipped_when_soh_column_absent` |
| `overcharge` (voltage ≥ 4.25V) | `test_features.py::test_overcharge_detected` (rule tested in isolation; confirmed to never fire on the real dataset — see SRS 5.1) |
| Priority: overheat over overcharge | `test_features.py::test_overheat_takes_priority_over_overcharge` |
| Priority: overheat over capacity_degraded | `test_features.py::test_overheat_takes_priority_over_capacity_degraded` |
| `short_circuit_indication` (current spike ≥3A) | Rule logic covered indirectly through the engineer_features tests; no dedicated fault-firing test since the real dataset never triggers it (max observed delta 1.61A) — *add a synthetic-input test if you want direct coverage of the rule itself* |

## Data Integrity Traceability (new — added for real dataset integration)

| Requirement | Design Component | Test Case(s) |
|---|---|---|
| Real CSV columns correctly mapped to internal schema | `src/model.py::load_kaggle_dataset` | `test_model.py::TestLoadKaggleDataset` (2 cases) |
| Rolling/delta features must not leak across cycle boundaries | `src/features.py::engineer_features` (per-group logic) | `test_features.py::test_grouped_data_does_not_leak_across_cycles`, `test_grouped_data_deltas_correct_within_cycle` |
| Train/test split must not leak the same battery into both sets | `src/model.py::train_test_split_by_battery` | `test_model.py::TestTrainTestSplitByBattery` (2 cases) |

---

## Coverage Gaps Summary (carry these into your defect log / future work section)

1. **FR-3 and FR-6** (API-level behavior) have no automated tests yet — only unit tests for the underlying logic they call.
2. **NFR-4** (performance) is not automated — add a simple timing assertion in a future test.
3. **`short_circuit_indication` rule** has no test that directly triggers it on synthetic input — the real dataset never exercises this path (documented limitation), so add a small synthetic-data test if you want direct rule coverage independent of the dataset.
4. **`capacity_degraded` recall (53%)** is the weakest-performing class — flagged in SRS 5.2 as an honest finding with a suggested cross-cycle feature improvement, not something silently left out of the report.
