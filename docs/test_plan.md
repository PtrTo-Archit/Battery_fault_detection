# Test Plan
## AI-Based Lithium-Ion Battery Fault Detection System

**Version:** 1.0
**Related documents:** `docs/SRS.md`, `docs/traceability_matrix.md`

---

## 1. Purpose

This document defines what will be tested, how, and what counts as a
pass, for the battery fault detection system. It covers both
**software-level testing** (does the code behave correctly) and
**model-level testing** (does the classifier make correct decisions),
since this project has both a conventional software component and an
ML component — they need different pass/fail criteria.

---

## 2. Scope

In scope:
- Unit testing of validation, feature engineering, and model logic
- Model evaluation against held-out test data (precision/recall/F1/confusion matrix)
- Boundary and negative testing of input validation
- API contract testing (request/response shape, error handling)
- Static analysis (pylint) and complexity analysis (radon)
- Code coverage measurement

Out of scope:
- Load/stress testing at production scale (not meaningful for an academic prototype)
- Security penetration testing (no auth layer exists in this version)
- Hardware-in-the-loop testing (no physical battery/BMS integration)

---

## 3. Test Levels

### 3.1 Unit Testing
**Tooling:** `pytest`, run via `pytest tests/`
**Target coverage:** ≥ 80% line coverage on `src/`, measured with `pytest-cov`

| Module | What's tested |
|---|---|
| `validation.py` | Valid readings pass; each out-of-range field (voltage/current/temp) is independently rejected with a correct reason string; boundary values (min/max inclusive) are accepted; missing fields rejected; batch validation correctly splits valid/rejected |
| `features.py` | Rolling/delta features computed correctly on known input; first-row edge case (no previous value) handled; original DataFrame not mutated; each fault rule (overcharge/over-discharge/overheat) fires on its threshold; priority ordering when multiple thresholds are met simultaneously |
| `model.py` | Predict/save fail loudly before training; train→predict produces correct output shape; evaluate() returns all required metric keys; accuracy and critical-fault FNR fall within valid [0,1] range; save/load round-trip produces identical predictions; loading a nonexistent file raises a clear error |

### 3.2 Integration / API Testing
**Tooling:** FastAPI `TestClient` (recommended addition — see Section 7)

| Scenario | Expected result |
|---|---|
| POST `/predict` with ≥5 valid readings | 200 OK, returns a `fault_type` string |
| POST `/predict` with <5 readings | 400 Bad Request, explains minimum window size |
| POST `/predict` with some invalid readings but ≥5 valid remain | 200 OK, `rejected_readings` count > 0 |
| POST `/predict` with too many invalid readings (fewer than 5 valid remain) | 422 Unprocessable Entity |
| GET `/health` before model is loaded | `model_loaded: false` |
| GET `/health` after model is loaded | `model_loaded: true` |

### 3.3 Model Quality Evaluation
This is *not* the same as unit testing — it validates the model's
decisions are correct, not that the code runs without crashing.

**Method:** train/test split (75/25, stratified by fault_type), report:
- Overall accuracy
- Per-class precision, recall, F1 (macro-averaged, since fault classes are imbalanced)
- Confusion matrix
- **Critical-fault false negative rate** — the single most important number for this domain, since missing a real overheat/short-circuit event is a safety issue, not just an accuracy blip

**Pass criteria (suggested — adjust to your rubric):**
- Overall accuracy ≥ 85% on held-out data
- Critical-fault false negative rate ≤ 5%
- No class with 0% recall on classes that actually occur in the dataset

**Actual results on the real dataset** (`data/battery_health_dataset.csv`,
split by `battery_id`, 18 train / 6 test batteries): 86.7% accuracy,
0.10% critical-fault false negative rate. Full breakdown and honest
discussion of the weakest class (`capacity_degraded`, 64% F1) is in
`docs/SRS.md` Section 5.2 — report these real numbers, not placeholder
targets.

**Caveat about the bundled synthetic dataset:** `generate_synthetic_dataset()`
remains in the codebase purely as a pipeline smoke-test fallback for
when the real CSV isn't present. On synthetic data, accuracy is close
to 100% because the labels are generated from the same simple
thresholds the model easily learns — this is expected and not
meaningful as a reported result. Only the real-dataset numbers above
belong in your final report.

### 3.4 Static Analysis / Code Quality
- `pylint src/` — track score out of 10, document any disabled/ignored warnings and why
- `radon cc src/ -a` — flag any function with complexity grade D or worse; refactor or justify in report
- `pytest --cov=src --cov-report=term-missing` — report final coverage %

---

## 4. Test Environment

- Python 3.11+ (CI uses 3.11; local dev tested on 3.12)
- Dependencies pinned via `requirements.txt`
- CI runs on GitHub Actions (`ubuntu-latest`) on every push/PR — see `.github/workflows/ci.yml`

---

## 5. Entry / Exit Criteria

**Entry criteria** (before testing a module is considered "ready"):
- Module implements the functional requirement it's tied to (see traceability matrix)
- Module has type hints and docstrings

**Exit criteria** (before a module is considered "done"):
- All associated unit tests pass
- Module-level coverage ≥ 80%
- No pylint errors (warnings acceptable if documented)
- Any known limitation is logged in the defect log or noted in SRS assumptions

---

## 6. Test Case Summary Table

Full test cases live directly in the test files (`tests/test_validation.py`,
`tests/test_features.py`, `tests/test_model.py`) — 30 cases as of this
version. Summary:

| Test file | # cases | Focus |
|---|---|---|
| `test_validation.py` | 10 | Boundary values, rejection reasons, batch splitting |
| `test_features.py` | 10 | Feature correctness, fault rule thresholds, priority ordering |
| `test_model.py` | 10 | Train/predict/evaluate contract, persistence, synthetic data generator |

Run `pytest -v` for the live pass/fail status of each individual case.

---

## 7. Known Gaps (be upfront about these in your report — it shows maturity, not weakness)

1. No API-level tests exist yet (`test_api.py` not implemented). Add using `fastapi.testclient.TestClient` if your rubric expects end-to-end coverage.
2. No test currently exercises the model against real (non-synthetic) data — do this once you've sourced a dataset, and record the real metrics separately from the synthetic-data sanity check.
3. No performance/load testing — acceptable for course scope, but worth a sentence in your report explaining why it's out of scope rather than silently omitting it.
