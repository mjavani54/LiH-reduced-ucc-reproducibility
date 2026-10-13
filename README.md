# Symmetry-aware factorized-UCC selection and transfer: reproducibility materials

This repository accompanies the manuscript:

> **Symmetry-aware selection and transfer of factorized UCC operators in LiH and linear H4**  
> Mohammad H. Javani

It contains the frozen historical calculation record plus the separately dated post-review extension used in the revised manuscript. The extension adds held-out LiH geometries, same-pool fermionic ADAPT comparisons, exhaustive and matched controls, ranking-sensitivity tests, matched circuit resources, and a four-electron linear-H4 boundary test.

## Revised result and claim boundary

For the frozen-core LiH/6-31G 2e,10o model, Partner-47 gives a maximum absolute energy error of `1.68754e-14 Ha`, maximum `<S^2>` of `1.11571e-13`, and maximum final gradient infinity norm of `4.40778e-08` over three selection geometries and four held-out geometries. The selected support is identical to the converged 47-member support from ideal same-pool fermionic ADAPT at all three selection geometries.

The 46-to-47 result is controlled directly: only E17 among all 53 one-operator additions to Boundary-46 reaches strict numerical agreement, while none of 100 size-, composition-, Reduced-10-, and partner-closure-matched random supports reaches chemical accuracy. Under matched abstract compilation, Partner-47 uses 6006 CNOTs and depth 6670, approximately 53% below Full-99.

The conclusion is deliberately bounded. The four additional LiH geometries were designed after peer review rather than prospectively preregistered. In linear H4/STO-3G, both the frozen 10-operator support and full 26-operator factorized UCCSD exceed chemical accuracy at stretched spacings. The repository therefore supports a validated selection-and-testing workflow, not molecular universality, formal spin adaptation, or near-term quantum advantage.

## Frozen historical result

For frozen-core LiH/6-31G in the 2e,10o active space, the originally archived 47-operator support reproduced the exact active-space energies at all three confirmation geometries with:

- worst absolute energy error: `2.8108715355301683e-10 Ha`;
- maximum `<S^2>`: `2.587583823799815e-09`;
- 47 operators instead of the 99-operator UCCSD control;
- Phase-7D support fingerprint: `0049b07d140790c3d87336bccf293a344fa527001526b014fd074e6c8a2ac082`;
- Phase-7D protocol fingerprint: `454022626abb6f3096e3fda104e2d34c05050eee9a63176c50a62423e8b3c6aa`.

This historical record is retained unchanged so the revised claims can be audited against the originally deposited calculations.

## Release-candidate warning

This is **version 0.10.0, not the final archival release**. One shared source file available during repository assembly, `lih_reference_failure_diagnostics.py`, was truncated mid-statement at exactly 16 KiB. Its surviving code was preserved and the API required by the later scripts was reconstructed. The reconstructed file compiles, but it is not a byte-identical copy of the helper used for the original calculations.

Before assigning version 1.0.0 or depositing on Zenodo, replace that helper with the exact file from the original calculation machine and execute the release checklist in [`docs/PUBLIC_RELEASE_CHECKLIST.md`](docs/PUBLIC_RELEASE_CHECKLIST.md). Full details and checksums are in [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md).

## Repository contents

```text
analysis/                Historical figure-generation code
docs/                    Reproduction guide, data dictionary, provenance, release checklist
figures/                 Manuscript graphics, including the revised graphical abstract
post_review_extension/   New scripts, outputs, figures, checksums, and verifier
results/                 Archived processed outputs and frozen JSON records
scripts/                 Historical verification and reproduction entry points
src/                     Phase 2B through Phase 7D calculation scripts
```

The Wiley manuscript template, publisher artwork, local checkpoints, `__pycache__` files, operating-system metadata, and incomplete Phase-7B execution stages are intentionally excluded.

## Quick verification

Both processed-data verifiers use only the Python standard library:

```bash
python scripts/verify_package.py
python post_review_extension/verify_post_review_extension.py
```

The first command verifies the frozen historical record, fingerprints, completion flags, expected geometries and variants, numerical thresholds, and the memory-safe full-statevector spot check. The second verifies every post-review file hash plus the held-out LiH result, exhaustive and matched controls, ADAPT support identity, ranking stability, circuit-resource ordering, and stretched-H4 boundary.

After an intentional repository change, regenerate both manifests and verify again:

```bash
python post_review_extension/verify_post_review_extension.py --write-manifest
python scripts/generate_manifest.py
python scripts/verify_package.py
python post_review_extension/verify_post_review_extension.py
```

## Recreate the figures

Create the historical environment first, then run:

```bash
conda env create -f environment.yml
conda activate lih-reduced-ucc
bash scripts/reproduce_figures.sh
```

Regenerate the five revised figures and graphical abstract with:

```bash
python post_review_extension/make_revision_figures.py
python post_review_extension/make_graphical_abstract.py
```

## Full numerical rerun

The Phase-7A through Phase-7D historical pipeline can be launched with:

```bash
bash scripts/run_phase7_pipeline.sh
```

This is a long, CPU- and memory-intensive calculation. It is not run by the quick verifier. The Phase-7D statevector check uses explicit Pauli-evolution circuit synthesis and avoids constructing a dense `2^20 x 2^20` operator, which would require 16 TiB in complex128 format.

The new calculations can be rerun from the commands in [`post_review_extension/README.md`](post_review_extension/README.md). For the distinction between archived-result verification, figure reproduction, and a numerical rerun, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Environment status

The original calculation environment was not exported at run time. Root `environment.yml` and `requirements.txt` therefore describe a compatible reconstruction target; they are not an exact historical lock file. The post-review environment is pinned separately in `post_review_extension/requirements.txt` and records the versions used for the new calculations. Capture the environment that passes a final historical rerun with:

```bash
bash scripts/capture_environment.sh
```

## Citation and archival release

Citation metadata are provided in `CITATION.cff`. After the final repository audit, create an archival release and Zenodo DOI, then add that DOI to `CITATION.cff`, this README, and the manuscript Data and Code Availability statement.

## License

The repository is released under the MIT License. See `LICENSE`.

