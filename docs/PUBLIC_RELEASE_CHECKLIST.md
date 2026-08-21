# Public release checklist

Do not tag `v1.0.0` or create a Zenodo DOI until every blocking item is complete.

## Blocking items

- [ ] Replace the reconstructed `lih_reference_failure_diagnostics.py` with the exact original full source.
- [ ] Run the Phase-7A through Phase-7D pipeline in a clean environment.
- [ ] Confirm both frozen Phase-7D fingerprints and all reported scientific thresholds.
- [ ] Export `python --version`, `pip freeze`, and `conda list --explicit` from the successful environment.
- [ ] Review every author name, affiliation, email, license, and funding statement.

## Repository hygiene

- [ ] Run `python scripts/verify_package.py` with no failures.
- [ ] Regenerate Figures 1–4 and visually inspect every PNG and PDF.
- [ ] Confirm there are no absolute local paths, credentials, private notes, publisher-owned templates, or operating-system metadata.
- [ ] Confirm `MANIFEST.sha256` is current.
- [ ] Confirm the ZIP extracts into one top-level directory and passes `unzip -t`.

## GitHub and Zenodo

- [ ] Create the public GitHub repository from the clean package contents.
- [ ] Add the final GitHub URL to `CITATION.cff` and `README.md`.
- [ ] Create a GitHub release tagged `v1.0.0`.
- [ ] Archive that release with Zenodo and reserve or mint the DOI.
- [ ] Add the DOI to `CITATION.cff`, `README.md`, and the manuscript.
- [ ] Use the version-specific DOI for the released package and the concept DOI where an all-versions citation is intended.

## Suggested manuscript Data Availability Statement

> The calculation scripts, frozen support and protocol records, processed numerical results, and figure-generation code supporting this study are available in the public repository at [GitHub URL] and are archived on Zenodo at https://doi.org/[DOI]. The archived release includes a SHA-256 manifest and automated verification of the principal numerical claims.

