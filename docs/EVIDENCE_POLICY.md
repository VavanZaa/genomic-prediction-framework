# Evidence policy

Adopted 11 September 2026, after the audit found serious defects behind a passing test suite (57 tests passed while C1–C10 were present).

**No result survives merely because the test suite passes.** Tests show that the code does what the tests expect; they do not show that the expectations are scientifically right. A number enters a report or walkthrough only when it passes all three gates below.

## The three gates

1. **Software correctness.** The code path that produced the number has a regression test that reproduces the failure it guards against, not only a success check, and the producing script ran end to end on the current code.
2. **Statistical validity.** Preprocessing, environmental adjustment, tuning and thresholds are fitted on training records only. Predictive uncertainty comes from held-out or out-of-fold residuals. Comparisons use paired intervals with the method named (paired t, paired bootstrap). No truth model is built from the quantity being evaluated.
3. **Evidence provenance.** Every dataset directory carries a `PROVENANCE` file (`empirical` or `synthetic`), and the loaders expose it as `.provenance`. Every result is labelled with one evidence source:

| Label | Meaning | Examples |
|---|---|---|
| EMPIRICAL | Measured animals | Florida multibreed herd (Layers 3–6), Spanish trios (Layer 5B) |
| SYNTHETIC | Generated data stored in the shape of real data | `data/synthetic_cattle/` (the Mini Cattle Genomic Benchmark) |
| SIMULATION | The engine's own simulation under assumed parameters | Stages A–F, Case 1 and Case 2 |

`output/audit_post_fix/EVIDENCE.csv` lists the label for every result file.

## Standing rules for code that produces results

- **Result generation is independent of the IDE or tool that launched it.** Scripts write only under the project (`output/`); copying figures elsewhere is a manual step (`scripts/export_figures_for_review.py`). `tests/test_pipeline_hygiene.py` enforces this.
- **No fixed result sentences.** Every number or verdict a script prints is computed from that run. Four scripts printed fixed numbers or claims (4B, 6A, the simulated-vs-real summary, and the Case 2 narrative); each contradicted its own output.
- **Support is empirical.** “Inside the training domain” means inside the leave-one-out k-NN support (`src/evaluation/support.py`), never between the minimum and maximum.
- **Simulated genomes are compared after array ascertainment** (`src/simulation/ascertainment.py`), so measurement differences are not mistaken for biology.
- **Frozen baselines are never edited.** `scripts/freeze_baseline.py` refuses to freeze on a failing test suite or into an existing folder.

## When a result fails a gate

Withdraw it; do not edit its numbers. Regenerate it from scratch on corrected code, and record the change as *old result → defect → new result → conclusion changed?* in `reports/2026-09-11_before_after.md`. Keep the old files (`output/audit_pre_fix/`) for comparison only.
