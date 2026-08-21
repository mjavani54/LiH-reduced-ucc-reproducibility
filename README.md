# LiH reduced-UCC support: code, processed data, and reproducibility materials

This repository accompanies the manuscript:

> **Discovery, Cross-Basis Transfer, and Spin-Complete Active-Space Augmentation of Reduced UCC Operator Supports for LiH**  
> Mohammad H. Javani

It contains the staged calculation scripts, frozen protocol/support records, processed numerical results, figure-generation code, and manuscript figures for the LiH study.

## Main archived result

For frozen-core LiH/6-31G in the 2e,10o active space, the frozen 47-operator spin-complete support reproduces the exact active-space energies at all three confirmation geometries with:

- worst absolute energy error: `2.8108715355301683e-10 Ha`;
- maximum `<S^2>`: `2.587583823799815e-09`;
- 47 operators instead of the 99-operator UCCSD control;
- Phase-7D support fingerprint: `0049b07d140790c3d87336bccf293a344fa527001526b014fd074e6c8a2ac082`;
- Phase-7D protocol fingerprint: `454022626abb6f3096e3fda104e2d34c05050eee9a63176c50a62423e8b3c6aa`.

The claim is deliberately bounded to noiseless calculations for the three frozen LiH/6-31G Hamiltonians in the deposited workflow. It does not establish transfer to other molecules, basis sets, active spaces, geometries, or hardware noise models.

## Release-candidate warning

This is **version 0.9.0, not the final public release**. One shared source file available during repository assembly, `lih_reference_failure_diagnostics.py`, was truncated mid-statement at exactly 16 KiB. Its surviving code was preserved and the API required by the later scripts was reconstructed. The reconstructed file compiles, but it is not a byte-identical copy of the helper used for the original calculations.

Before assigning version 1.0.0 or depositing on Zenodo, replace that helper with the exact file from the original calculation machine and execute the release checklist in [`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md). Full details and checksums are in [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md).

## Repository contents

```text
analysis/       Figure-generation code
docs/           Reproduction guide, data dictionary, provenance, release checklist
figures/        Final PNG and vector PDF figures
results/        Archived processed outputs and frozen JSON records
scripts/        Verification, environment capture, and reproduction entry points
src/            Phase 2B through Phase 7D calculation scripts
```

The Wiley manuscript template, publisher artwork, local checkpoints, `__pycache__` files, operating-system metadata, and incomplete Phase-7B execution stages are intentionally excluded.

## Quick verification

The processed-data verification uses only the Python standard library:

```bash
python scripts/verify_package.py
```

It verifies the SHA-256 manifest, frozen fingerprints, completion flags, expected rows/geometries/variants, energy and spin thresholds, and the memory-safe full-statevector spot-check record.

After an intentional file change, regenerate the checksum manifest with:

```bash
python scripts/generate_manifest.py
python scripts/verify_package.py
```

## Recreate the manuscript figures

Create the environment first, then run:

```bash
conda env create -f environment.yml
conda activate lih-reduced-ucc
bash scripts/reproduce_figures.sh
```

The script regenerates Figures 1–4 from `results/`. Both dual-panel figures are vertically stacked, and all typography is set for single-column readability.

## Full numerical rerun

The Phase-7A through Phase-7D pipeline can be launched with:

```bash
bash scripts/run_phase7_pipeline.sh
```

This is a long, CPU- and memory-intensive calculation. It is not run by the quick verifier. The Phase-7D statevector check uses explicit Pauli-evolution circuit synthesis and avoids constructing a dense `2^20 x 2^20` operator, which would require 16 TiB in complex128 format.

For the distinction between archived-result verification, figure reproduction, and a numerical rerun, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Environment status

The original calculation environment was not exported at run time. `environment.yml` and `requirements.txt` therefore describe a compatible reconstruction target centered on Python 3.14, Qiskit Nature 0.8, and Qiskit Algorithms 0.4; they are not an exact historical lock file. Capture the environment that passes the final rerun with:

```bash
bash scripts/capture_environment.sh
```

## Citation and archival release

Citation metadata are provided in `CITATION.cff`. After the GitHub repository is public, create a Zenodo release, obtain the DOI, then add that DOI and repository URL to `CITATION.cff`, this README, and the manuscript Data Availability Statement.

## License

The repository is released under the MIT License. See `LICENSE`.
