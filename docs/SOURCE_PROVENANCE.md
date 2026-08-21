# Source provenance and known blocker

## Verbatim archived scripts

The Phase 2B–6 and Phase 7A–7D scripts in `src/` were copied from the files supplied for repository assembly. The duplicate file `lih_phase7d_memory_safe_confirmation.py` was omitted because it was byte-for-byte identical to the canonical `lih_phase7d_frozen_qiskit_confirmation.py` at assembly time. Their common SHA-256 value was:

```text
85a8ec1a2244ecf9d7917a2687951a690f437b20b3691265b567892502450318
```

The final Phase-7D script differs from the earlier failing copy whose user-reported SHA-256 was `467a5413d38ad6487474321abcd7f96d6b3417489018ac8e1b5dbecac6e0c57e`. The deposited canonical script contains the memory-safe statevector synthesis used for the completed confirmation.

## Reconstructed helper

The supplied `lih_reference_failure_diagnostics.py` ended at byte 16,384 in the middle of this statement:

```text
if isinstanc
```

Its archived truncated SHA-256 was:

```text
66c4a21fee479badd838bab40ffc8c862a2813d2b6ab13d168b2c9b5d447ffd5
```

The surviving portion defined the molecular configuration, `SystemData`, Qiskit/PySCF Hamiltonian construction, reference validation, excitation normalization, and the beginning of selection-file parsing. For this release candidate, only the missing API used by the surviving phase scripts was reconstructed:

- selection-record normalization;
- the frozen UCC excitation callback;
- selected-UCC ansatz construction;
- deterministic restart initialization;
- circuit resource reporting;
- direct statevector/SciPy optimization.

This reconstruction makes the source tree syntactically complete and documents the intended workflow. It does **not** prove byte identity or numerical identity with the unavailable original tail.

## Required action before public version 1.0.0

Retrieve the exact full helper from the machine or backup used for the calculations. Replace `src/lih_reference_failure_diagnostics.py`, rerun the Phase-7 pipeline, compare scientific quantities and fingerprints, capture the exact environment, update this document, regenerate `MANIFEST.sha256`, and only then tag the public release.

The path normalization in `analysis/make_figures.py` is intentional and non-scientific: local `upload/`, `recovered_phase6/`, and `paper_data/` references were replaced by the public `results/` layout.

Five absolute Windows/WSL provenance strings in `results/phase6/phase6_final_conclusions.json` were replaced by stable `source-id:` labels. No numerical value, scientific classification, source SHA-256 value, or frozen fingerprint was changed. This prevents disclosure of workstation-specific paths while retaining the identity of the base and two rescue sources.
