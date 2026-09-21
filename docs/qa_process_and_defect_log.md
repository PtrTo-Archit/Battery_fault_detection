# Quality Assurance Process & Defect Log
## AI-Based Lithium-Ion Battery Fault Detection System

**Version:** 1.0

---

## Part 1: Software Quality Assurance Process

This section describes the actual workflow used during development —
not an idealized process, but what was really followed, since a
credible SQA writeup describes real practice rather than theory.

### 1.1 Version Control Workflow
- Single `main` branch for a project of this size is acceptable, but
  prefer feature branches (`feature/validation-module`,
  `feature/api-endpoint`) merged via pull request, even solo — it
  creates a reviewable diff and a commit history that documents
  *why* changes were made, not just *what* changed.
- Commit message convention: short imperative summary line, e.g.
  `Add rolling feature computation for voltage/temperature`,
  `Fix synthetic data generator index shape mismatch`.
- Avoid single giant "final project" commits — incremental commits
  are themselves quality evidence an evaluator can see directly in
  the repo's commit graph.

### 1.2 Code Review Self-Checklist
Used before merging any module, even when self-reviewing solo:
- [ ] Does every public function have a docstring explaining *why*, not just *what*?
- [ ] Are magic numbers (thresholds, bounds) named constants, not inline literals?
- [ ] Does the function fail loudly (raise) rather than silently on invalid state?
- [ ] Is there at least one test for the happy path and one for a failure/edge case?
- [ ] Would someone unfamiliar with the domain understand *why* a threshold exists (comment/docstring), not just what it is?

### 1.3 Continuous Integration
Every push and pull request triggers (`.github/workflows/ci.yml`):
1. Dependency install from `requirements.txt` (portability check)
2. `pytest --cov=src` (correctness + coverage)
3. `pylint src/` (static analysis)
4. `radon cc src/ -a` (complexity check)

This turns "we tested it" from a claim into something verifiable —
anyone can open the Actions tab and see pass/fail history over time.

### 1.4 Definition of Done (per module)
A module is considered complete when:
1. It implements a specific, traceable requirement (see `traceability_matrix.md`)
2. Unit tests exist for both success and failure paths
3. It passes `pylint` without unaddressed errors
4. It's documented in the README if it changes how the system is used

### 1.5 Quality Metrics Tracked Throughout Development

| Metric | Tool | Target |
|---|---|---|
| Line coverage | `pytest-cov` | ≥ 80% |
| Cyclomatic complexity | `radon` | Grade C or better per function |
| Static analysis score | `pylint` | Document score; no unexplained errors |
| Critical-fault false negative rate | Custom (`model.py`) | ≤ 5% (domain-specific safety metric, not a generic code metric) |

---

## Part 2: Defect Log

A real defect found and fixed during development, logged here as
process evidence. Add new rows as you find and fix issues during your
own work — a log with only one entry looks thin; a log with several
real entries (even small ones) is much more convincing evidence of an
actual QA process to an evaluator.

| ID | Date Found | Description | Severity | Root Cause | Fix | Verified By |
|---|---|---|---|---|---|---|
| DEF-001 | Development | `generate_synthetic_dataset()` raised `ValueError: shape mismatch` when injecting fault samples into small datasets (e.g. `n_samples=50`) — integer division across three fault-type slices didn't always sum back to the total selected fault count, causing an array-length mismatch on assignment. | Medium — blocked all model tests, since every test using the synthetic dataset failed at collection/setup | Fault-index slicing used `len(fault_idx) // 3` independently for each slice boundary instead of deriving all three slice lengths from a single consistent split point, so rounding could leave a mismatched final slice. | Rewrote the slicing to compute `third = n_fault_samples // 3` once and derive all three index ranges from that single value, with the final slice taking the remainder explicitly. Also added `max(3, ...)` to guarantee at least a few fault samples even at very small `n_samples`. | Re-ran full test suite (`pytest tests/ -v`) — 30/30 passed after fix. Also manually ran `python -m src.model` to confirm the end-to-end pipeline executes cleanly. |
| DEF-002 | Real dataset integration | Original `engineer_features()` computed rolling means and row-to-row deltas across the *entire* DataFrame with no awareness of cycle boundaries. Once the real dataset (grouped by `battery_id` + `cycle_number`, 20 rows/cycle) was loaded, this meant the first reading of every new discharge cycle had its "delta" computed against the *last* reading of the *previous, unrelated* cycle — a subtle correctness bug that wouldn't crash anything, just silently produce wrong feature values. | High — would have silently degraded model quality/validity without raising any error, the most dangerous kind of bug since nothing would look obviously broken | Function was written and originally tested only against small, single-sequence toy DataFrames that had no grouping structure, so the cross-boundary case was never exercised until real multi-cycle data was loaded. | Added group-aware logic: `engineer_features()` now detects `battery_id`/`cycle_number` columns and, when present, computes rolling/delta features independently per group via `groupby(...).transform(...)`; falls back to the original whole-DataFrame behavior when those columns are absent (keeps the live-API/simple-window use case unchanged). | Added dedicated regression tests (`test_grouped_data_does_not_leak_across_cycles`, `test_grouped_data_deltas_correct_within_cycle`) that explicitly assert the first row of a new cycle has delta 0.0, not a cross-boundary value. Also re-ran the full real-dataset training run and manually spot-checked a handful of cycle-boundary rows in the output DataFrame. |
| DEF-003 | Real dataset integration | Initial validation bounds (current -50 to 50A) and fault thresholds (overcharge 4.25V, overheat 60°C) were placeholder values chosen before any real data was available. Against the real dataset these were either too loose to be meaningful (current bounds) or slightly miscalibrated relative to the actual observed distribution (overheat threshold sat right above where real high-temperature readings cluster). | Low-Medium — not a crash, but would have produced a validation layer that accepted physically-impossible-for-this-dataset values, and a fault labeling scheme not grounded in the data it was applied to | Thresholds were written before the real dataset was available and never revisited once it was. | Recalibrated all bounds/thresholds against the real dataset's actual percentiles (documented in `docs/SRS.md` Section 5) instead of leaving placeholder values in place after real data arrived. | Re-ran `python -m src.model` on the real dataset and confirmed the resulting label distribution (Section 5.1 of SRS) is sensible — no fault class is empty or absurdly dominant, and `overcharge`/`short_circuit_indication` are correctly near-zero for the documented dataset-limitation reason rather than a threshold bug. |

### How to keep adding to this log
As you build out the project further (real dataset integration, API
tests, dashboard if you build one), log anything non-trivial you fix:
- What broke and how you noticed it (test failure, manual testing, review)
- Why it happened (root cause, not just symptom)
- What you changed
- How you confirmed the fix actually worked

Even 3–5 honest entries by submission time is far more convincing to
an evaluator than a suspiciously clean "we had zero bugs" narrative —
it shows the SQA process was actually catching things, which is the
entire point of the course.

---

## Part 3: Suggested Next QA Actions (carry into your remaining project weeks)

1. Add `test_api.py` using `fastapi.testclient.TestClient` to close the FR-3/FR-6 coverage gap noted in the traceability matrix.
2. Add a dedicated test for the `short_circuit_indication` fault rule (currently untested at the unit level).
3. Once real dataset is sourced, re-run `evaluate()` and record real metrics separately from the synthetic-data sanity check — don't let the synthetic 100% accuracy number make it into your final report as if it were a real result.
4. Turn on `pylint` as a blocking CI step (remove `--exit-zero` from `ci.yml`) once the codebase is clean, so future regressions actually fail the build instead of just being logged.
