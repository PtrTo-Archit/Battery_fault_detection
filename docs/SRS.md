# Software Requirements Specification (SRS)
## AI-Based Lithium-Ion Battery Fault Detection System

**Version:** 1.0
**Course:** Software Quality (Elective)
**Document status:** Draft — update version/date as the project evolves

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non-functional requirements
for a software system that detects fault conditions in lithium-ion
batteries using telemetry data (voltage, current, temperature). It
serves as the baseline against which the system's test plan,
traceability matrix, and final evaluation are measured.

### 1.2 Scope
The system accepts battery telemetry readings (either as a batch CSV
file or a live stream via API), applies input validation, engineers
time-series-derived features, and classifies each reading window into
one of several operational states using a trained machine learning
model. The system does **not** control the battery or BMS hardware
directly — it is a decision-support / monitoring layer only.

### 1.3 Definitions

| Term | Definition |
|---|---|
| Telemetry reading | A single sample of voltage, current, and temperature at a point in time |
| Fault | A classified abnormal state: overcharge, over-discharge, overheat, or short-circuit indication |
| Critical fault | A fault type where a missed detection (false negative) poses a safety risk — here, `overheat` and `short_circuit_indication` |
| Window | A short chronological sequence of readings used to compute rolling/delta features |
| SQA | Software Quality Assurance |

### 1.4 Intended Audience
Course evaluator, project team, and (as a design fiction) a hypothetical
battery monitoring engineer who would consume this system's output.

---

## 2. Overall Description

### 2.1 Product Perspective
Standalone academic project. Not integrated with real BMS hardware.
Trained and evaluated on `data/battery_health_dataset.csv`, a NASA
PCoE Li-ion battery discharge dataset (distributed via Kaggle) covering
24 batteries, ~29,180 readings, 20 readings per discharge cycle, with
columns `Voltage_measured`, `Current_measured`, `Temperature_measured`,
`SoC`, `SoH`, `cycle_number`, `battery_id`. A synthetic data generator
(`src/model.py::generate_synthetic_dataset`) remains available as a
fallback for pipeline smoke-testing when the real file isn't present.

**Known dataset limitation (documented deliberately, not glossed
over):** this dataset contains discharge cycles only — no charging
data. As a direct consequence, the `overcharge` fault type and the
current-spike-based `short_circuit_indication` fault type essentially
never occur in this dataset (current stays within a narrow, near-constant
negative range throughout discharge). This is stated explicitly in the
project report rather than presented as if the system detects faults
it was never actually tested against.

### 2.2 Product Functions (Summary)
1. Validate incoming telemetry against physical sensor bounds.
2. Engineer rolling statistical and rate-of-change features from raw readings.
3. Classify the current battery state into `normal` or a specific fault type.
4. Expose classification via a REST API.
5. Report model quality metrics (accuracy, precision, recall, F1, and critical-fault false negative rate).

### 2.3 Assumptions and Constraints
- Assumes single-cell telemetry, bounds calibrated to the real dataset's observed range (voltage 2.0–4.3V, current -6.0–1.0A, temperature -10–80°C). Pack-level systems or different chemistries would require rescaling these bounds.
- Assumes readings arrive in chronological order within a window, and — for the real dataset — are grouped by `battery_id` and `cycle_number` so rolling/delta features don't compute across cycle boundaries.
- Ground-truth fault labels are derived via documented threshold rules (Section 5), calibrated against this dataset's real observed distribution plus one industry-standard rule (80% SoH = end-of-life) rather than arbitrary numbers — this is still a simplification appropriate for a course project (no manufacturer fault annotations exist), not a claim of clinical/industrial validation.
- No GPU or specialized hardware required; the Random Forest model is intentionally chosen to be trainable on a laptop.
- Train/test split is performed **by battery_id**, not by row, so that all cycles from a given battery fall entirely into train or entirely into test — this avoids data leakage from highly-correlated consecutive readings within the same cycle inflating apparent accuracy.

---

## 3. Non-Functional Requirements (Quality Attributes)

Mapped to **ISO/IEC 25010** product quality characteristics.

| ID | Attribute | Requirement | How it's measured |
|---|---|---|---|
| NFR-1 | Functional Suitability | System must correctly classify at least 4 distinct states (normal + 3+ fault types) | Classification report (per-class precision/recall) |
| NFR-2 | Reliability | Critical fault false negative rate must be minimized and explicitly reported, not hidden inside an aggregate accuracy figure | `critical_fault_false_negative_rate` metric in `model.py` |
| NFR-3 | Reliability (robustness) | System must reject malformed/out-of-range input without crashing | Validation unit tests (`test_validation.py`) |
| NFR-4 | Performance Efficiency | A single prediction must return in under 200ms on typical laptop hardware, given a valid window | Manual timing / API response time |
| NFR-5 | Maintainability | Each pipeline stage (validation, features, model, API) must be independently unit testable, target ≥80% line coverage | `pytest --cov` report |
| NFR-6 | Maintainability (complexity) | No function should exceed a cyclomatic complexity grade of "C" (radon) without justification | `radon cc` report |
| NFR-7 | Usability | API must expose self-documenting interactive docs | FastAPI `/docs` endpoint (automatic) |
| NFR-8 | Portability | System must run from a clean environment using only `requirements.txt` | Verified via CI pipeline on GitHub Actions |

---

## 4. Functional Requirements

### FR-1: Input Validation
The system shall validate every incoming reading against defined
physical bounds (voltage 0–5V, current -50–50A, temperature -20–150°C)
and reject readings outside these bounds with a descriptive reason.

*Traces to:* `src/validation.py`, `tests/test_validation.py`

### FR-2: Feature Engineering
The system shall compute, for each reading in a window: voltage delta,
current delta, temperature delta, and rolling mean/std over a
configurable window size (default 5).

*Traces to:* `src/features.py::engineer_features`, `tests/test_features.py`

### FR-3: Minimum Window Enforcement
The system shall require a minimum of 5 valid readings in any
prediction request, since rolling/delta features cannot be computed
from a single point. Requests with fewer valid readings shall return
an HTTP 400/422 error with a clear explanation.

*Traces to:* `src/api.py::predict`

### FR-4: Fault Classification
The system shall classify each processed window into one of: `normal`,
`overcharge`, `over_discharge`, `overheat`, `short_circuit_indication`.

*Traces to:* `src/model.py::BatteryFaultClassifier`, `src/features.py::label_faults`

### FR-5: Model Evaluation Reporting
The system shall report accuracy, macro-precision, macro-recall,
macro-F1, a confusion matrix, and critical-fault false negative rate
for any trained model against a held-out test set.

*Traces to:* `src/model.py::BatteryFaultClassifier.evaluate`

### FR-6: API Access
The system shall expose a REST endpoint (`POST /predict`) accepting a
JSON window of readings and returning the classified fault type for
the most recent reading in that window.

*Traces to:* `src/api.py`

### FR-7: Model Persistence
The system shall support saving a trained model to disk and reloading
it without retraining.

*Traces to:* `src/model.py::BatteryFaultClassifier.save/.load`

---

## 5. Fault Threshold Definitions (ground truth rules)

These thresholds define what counts as a "correct" label in the
absence of manufacturer-provided fault annotations. Calibrated against
`data/battery_health_dataset.csv`'s real observed distribution (24
batteries, ~29,180 readings: voltage 2.73–4.23V, temperature 4.7–66.7°C,
current -4.03–-0.89A) plus one industry-standard rule, rather than
arbitrary numbers. Documented here so grading/traceability can
reference a single source of truth (kept in sync with `src/features.py`).

| Fault type | Rule | Justification |
|---|---|---|
| `overheat` | temperature ≥ 55°C | Dataset's 95th percentile is ~54.9°C; chosen as a conservative early-warning threshold below the point where thermal runaway risk meaningfully rises |
| `short_circuit_indication` | abrupt current change ≥ 3A between consecutive readings | Dataset's max observed within-cycle current delta is 1.61A, so this threshold sits above normal discharge variation. **Essentially never fires on this dataset** (see Section 2.1 limitation) since it's discharge-only with near-constant current |
| `over_discharge` | voltage ≤ 3.0V | Common conservative Li-ion low-voltage cutoff used by real BMS designs; dataset's own 6th percentile (~2.96V) is consistent with this |
| `capacity_degraded` | SoH < 80% | Industry-standard Li-ion end-of-life definition: a cell is considered degraded once capacity fades to 80% of rated capacity |
| `overcharge` | voltage ≥ 4.25V | Kept for generality (e.g. a future charging dataset). **Essentially never fires on this dataset** (see Section 2.1 limitation) since it contains discharge cycles only |
| `normal` | none of the above | |

Priority when multiple conditions are met simultaneously: `overheat` >
`short_circuit_indication` > `over_discharge` > `capacity_degraded` >
`overcharge`, since thermal and short-circuit conditions are the more
immediately safety-critical failure modes, and long-term degradation
is the least urgent signal.

### 5.1 Observed Label Distribution (actual run on real dataset)

| Fault type | Count | % of dataset |
|---|---|---|
| `normal` | 16,731 | 57.3% |
| `capacity_degraded` | 9,281 | 31.8% |
| `over_discharge` | 1,751 | 6.0% |
| `overheat` | 1,417 | 4.9% |
| `overcharge` | 0 | 0.0% (expected — see limitation) |
| `short_circuit_indication` | 0 | 0.0% (expected — see limitation) |

### 5.2 Real Model Evaluation Results

Trained via `python -m src.model`, split by `battery_id` (18 train
batteries / 6 test batteries, `random_state=42`):

| Metric | Value |
|---|---|
| Overall accuracy | 86.7% |
| Critical-fault false negative rate | 0.10% |
| `overheat` F1-score | 1.00 |
| `over_discharge` F1-score | 1.00 |
| `normal` F1-score | 0.90 |
| `capacity_degraded` F1-score | 0.64 |

**Interpretation worth including in your report:** the model catches
overheat and over-discharge almost perfectly since they're sharp
threshold-based signals directly visible in single readings.
`capacity_degraded` is meaningfully harder (64% F1, 53% recall) because
gradual capacity fade is a subtler, cross-cycle pattern rather than a
simple instantaneous threshold — this is an honest and interesting
finding to discuss, not a flaw to hide. A natural "future work" item is
engineering features that look across cycles (e.g. trend of discharge
duration or voltage-drop rate over recent cycles) rather than
within-cycle features alone, which would likely help this class
specifically.

---

## 6. Out of Scope
- Real-time hardware/BMS integration
- Multi-cell pack balancing logic
- Battery State-of-Health (SoH) / State-of-Charge (SoC) estimation (a related but distinct problem)
- Mobile app / embedded deployment

---

## 7. Traceability Summary

| Requirement | Design Component | Test File |
|---|---|---|
| FR-1 | `validation.py` | `test_validation.py` |
| FR-2 | `features.py::engineer_features` | `test_features.py` |
| FR-3 | `api.py::predict` | (add API-level test — see note below) |
| FR-4 | `model.py`, `features.py::label_faults` | `test_model.py`, `test_features.py` |
| FR-5 | `model.py::evaluate` | `test_model.py` |
| FR-6 | `api.py` | (add API-level test — see note below) |
| FR-7 | `model.py::save/load` | `test_model.py` |

**Note:** API-level tests (using FastAPI's `TestClient`) aren't included
in the current test suite yet — worth adding a `test_api.py` if your
rubric expects end-to-end coverage, not just unit-level tests.

---

## 8. Software Quality Metrics Used

This project measures quality at two levels: conventional **code/process
metrics** (does the software itself meet engineering standards) and
**domain-specific model metrics** (does the fault classifier make
correct, safety-appropriate decisions). Reporting both — rather than
only ML accuracy, or only code coverage — is the core "software quality
+ domain" argument of this project.

### 8.1 Code & Process Quality Metrics

| Metric | Tool | Target / Result |
|---|---|---|
| Test coverage (% lines covered) | `pytest-cov` | ≥80% target |
| Cyclomatic complexity | `radon cc` | Grade C or better per function |
| Static analysis score | `pylint` | No unaddressed errors |
| Number of automated test cases | `pytest` | 41 test cases across validation, features, and model modules |
| CI pass/fail history | GitHub Actions | Runs tests + lint + complexity check on every push (`.github/workflows/ci.yml`) |
| Requirements traceability | `docs/traceability_matrix.md` | Every functional/non-functional requirement mapped to a design component and test case |
| Defect count & resolution | `docs/qa_process_and_defect_log.md` | 3 logged defects, each with root cause, fix, and verification method |

### 8.2 Model / Domain-Specific Quality Metrics

| Metric | Purpose | Result (real dataset) |
|---|---|---|
| Accuracy | Overall correctness of fault classification | 86.7% |
| Precision (macro-avg) | How often predicted faults are real faults | 0.91 |
| Recall (macro-avg) | How often real faults are actually caught | 0.87 |
| F1-score (macro-avg) | Balance of precision/recall across classes | 0.88 |
| Confusion matrix | Per-class error breakdown | See Section 5.2 |
| **Critical-fault false negative rate** | Domain-specific safety metric — how often a truly dangerous fault (overheat) is missed | 0.10% |

The critical-fault false negative rate is the metric worth emphasizing
most in evaluation: it's deliberately reported separately from overall
accuracy because, in this domain, missing one real overheat event
matters far more than a few extra false alarms — a generic accuracy
number alone would hide that distinction.
