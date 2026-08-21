# Reproducibility guide

The repository supports three different tasks. They should not be described as if they were equivalent.

## Level 1: verify the deposited processed results

Run from the repository root:

```bash
python scripts/verify_package.py
```

This level checks file integrity and the central numerical assertions directly from the deposited CSV and JSON files. It requires only the Python standard library and should finish quickly.

The verifier checks:

- every file listed in `MANIFEST.sha256`;
- the Phase-7B, Phase-7C, and Phase-7D completion states;
- the Phase-7C and Phase-7D support/protocol fingerprints;
- the four expected Phase-7D variants over three geometries;
- strict energy and spin thresholds for `spin_complete_candidate47`;
- chemical accuracy but spin incompleteness of `chemical_boundary46`;
- the full-statevector spot-check and its memory-safe implementation record.

## Level 2: regenerate the figures from deposited results

```bash
conda env create -f environment.yml
conda activate lih-reduced-ucc
bash scripts/reproduce_figures.sh
```

The figure script reads only these deposited data products:

- `results/phase6/phase6_final_conclusions.json`;
- `results/phase7a1/phase7a1_embedding_conclusions.json`;
- `results/phase7b/phase7b_transfer_conclusions.json`;
- `results/phase7c/phase7c_deterministic_path_summary.csv`;
- `results/phase7d/phase7d_variant_geometry_summary.csv`.

It overwrites the figure files in `figures/`. PNG output is 450 dpi; PDF output remains vector-based.

## Level 3: rerun the Phase-7 numerical pipeline

```bash
conda env create -f environment.yml
conda activate lih-reduced-ucc
bash scripts/run_phase7_pipeline.sh
```

The script uses the archived Phase-6 conclusion record as the starting scientific input and writes new outputs under `work/`. The stages are:

1. Phase 7A canonical cross-basis audit;
2. Phase 7A.1 rotated-orbital embedding diagnostic;
3. Phase 7A.2 independent rotated-Hamiltonian audit;
4. Phase 7B transferred-support VQE and controls;
5. Phase 7C expanded-space discovery and frozen candidate;
6. Phase 7D independent frozen confirmation.

This is a long calculation. Resume behavior is enabled by the phase scripts. Keep the output directory unchanged when restarting.

## Phase 2B through Phase 6

The corresponding scripts are included in `src/` because they document the discovery, geometry-transfer, diagnosis, compact-support, and mechanism stages. Phase 5 can generate the Phase-5 inputs required by Phase 6. However, do not claim a forensically exact full-pipeline rerun until the source-recovery blocker in `SOURCE_PROVENANCE.md` is resolved and the exact original environment is exported.

## Numerical comparison policy

Do not require byte-identical CSV or PDF files across platforms. BLAS/LAPACK implementations, optimizer termination, PySCF integral ordering, Qiskit transpiler changes, and PDF metadata can alter low-order digits or hashes. Compare declared physical quantities and frozen fingerprints at their stated tolerances.

The scientific thresholds used by the confirmation are:

- chemical accuracy: `1.6e-3 Ha`;
- guard band: `8.0e-4 Ha`;
- strict energy equivalence: `1.0e-6 Ha`;
- spin-square tolerance: `1.0e-7`.

