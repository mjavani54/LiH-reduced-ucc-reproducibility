# Post-review benchmark extension

This directory contains the calculations added after peer review of the LiH reduced-UCC manuscript. It supplements, but does not overwrite, the archived Phase-7 calculation record in the repository root.

## What is included

- `lih_benchmark_extension.py`: four held-out LiH geometries, a same-pool fermionic ADAPT comparator, all 53 Boundary-46-plus-one controls, 100 size/composition/closure-matched random supports, spin-partner-block selection, ranking sensitivity, and matched circuit compilation.
- `h4_transfer_validation.py`: an independent four-electron linear-H4 boundary test at one selection spacing and three held-out spacings.
- `make_revision_figures.py`: regenerates the five revised manuscript figures from the deposited result tables.
- `make_graphical_abstract.py`: regenerates the revised graphical abstract with exact text, including “Strict numerical energy and S² agreement.”
- `results/`: CSV, JSON, and NPZ calculation records used by the revised manuscript and Supporting Information.
- `figures/`: publication PNG and vector PDF outputs.
- `verify_post_review_extension.py`: validates checksums and the principal numerical invariants without rerunning the electronic-structure calculations.
- `MANIFEST.sha256`: SHA-256 checksums for every deposited extension file other than the manifest itself.

## Environment

The post-review calculations were run with Python 3, NumPy 2.5.4, SciPy 1.18.1, pandas 2.3.3, Matplotlib 3.11.2, PySCF 2.14.0, Qiskit 2.5.2, Qiskit Nature 0.8.0, and Qiskit Algorithms 0.4.0. Install the exact extension dependencies with:

```bash
python -m pip install -r post_review_extension/requirements.txt
```

The historical files in the repository root retain their own provenance caveat and environment record.

## Fast verification

From the repository root:

```bash
python post_review_extension/verify_post_review_extension.py
```

The verifier checks all file hashes and confirms the geometry count, support counts, held-out labels, numerical-precision result for Partner-47, exhaustive one-operator control, 100 matched random controls, ADAPT support identity, ranking-sensitivity result, circuit-resource ordering, and the stretched-H4 negative boundary.

## Regenerate the revised figures

```bash
python post_review_extension/make_revision_figures.py
```

The script reads `post_review_extension/results/` and writes PNG and PDF files to `post_review_extension/figures/`.

## Rerun the LiH extension

The LiH calculation imports the original repository modules so that the excitation indexing, orbital construction, ordered factor convention, and analytic gradients remain consistent with the archived workflow:

```bash
python post_review_extension/lih_benchmark_extension.py \
  --repository . \
  --output-dir post_review_extension/results \
  --group all \
  --closed-control-count 100
```

This is substantially more expensive than checksum verification. Results are deterministic under the recorded seed, subject to normal platform-level floating-point variation.

## Rerun the H4 boundary test

```bash
python post_review_extension/h4_transfer_validation.py \
  --output-dir post_review_extension/results
```

## Claim boundary

The four added LiH geometries are out of sample with respect to support selection, but they were designed after peer review and were not prospectively preregistered. Linear H4 is a second, four-electron model and supplies a negative boundary: both the frozen 10-operator support and the full one-factor-per-excitation UCCSD ansatz exceed chemical accuracy at stretched spacings. These results do not establish molecular universality, formal spin adaptation, or near-term quantum advantage.

