# Data dictionary

## Phase 6: compact-support mechanism and ablation

- `phase6_final_conclusions.json`: consolidated scientific classifications, thresholds, support rule, fingerprints, and leave-one-out penalties.
- `phase6_consolidated_leave_one_out_summary.csv`: reoptimized leave-one-out results used to assess whether each compact10 operator is necessary.

## Phase 7A.1: cross-basis embedding diagnostic

- `phase7a1_embedding_conclusions.json`: decision record for rejection of a canonical 6-31G five-orbital subset and acceptance of the rotated embedding.
- `phase7a1_geometry_summary.csv`: geometry-resolved mapping and reliability summary.
- `phase7a1_canonical_subspace_scan.csv`: exhaustive canonical-subspace diagnostic.
- `phase7a1_reproduction_audit.csv`: reproduction of the earlier Phase-7A failure.
- `phase7a1_rotated_orbital_continuity.csv`: continuity of the rotated orbitals across geometry.

## Phase 7B: transferred support in rotated 2e,5o

- `phase7b_transfer_conclusions.json`: completed Phase-7B result. The archived file was the final execution stage named `phase7b_transfer_conclusions(2).json`; it was renamed to remove the local duplicate suffix.
- `phase7b_random_support_summary.csv`: descriptive results for the 20 frozen random ten-operator supports.

Earlier incomplete Phase-7B conclusion files were excluded because they were checkpoint-stage summaries, not separate scientific results.

## Phase 7C: expanded-space augmentation discovery

- `phase7c_frozen_discovery_support.json`: signed support pool, ranking, nested path, random-control definition, and support fingerprint frozen before evaluation.
- `phase7c_preregistered_protocol.json`: signed execution and decision protocol.
- `phase7c_augmentation_conclusions.json`: final discovery-stage decision and Phase-7D candidate.
- `phase7c_deterministic_path_summary.csv`: errors along the 10-through-47-operator deterministic augmentation path.
- `phase7c_operator_scores.csv`: frozen perturbative ranking information.
- `phase7c_random_support_summary.csv` and `phase7c_random_geometry_summary.csv`: random-control summaries.
- `phase7c_full99_control_summary.csv`: 99-operator control results.
- `phase7c_exact_hamiltonian_audit.csv`: post-freeze exact-Hamiltonian checks.
- `PHASE7C_REPORT.md`: generated human-readable stage report.

## Phase 7D: independent Qiskit confirmation

- `phase7d_frozen_confirmation_support.json`: four frozen confirmation variants and support fingerprint.
- `phase7d_preregistered_protocol.json`: confirmation protocol and protocol fingerprint.
- `phase7d_confirmation_conclusions.json`: final classification, candidate statistics, boundary-control interpretation, and statevector spot-check summary.
- `phase7d_variant_geometry_summary.csv`: manuscript-level table of energy error and `<S^2>` for four variants at three geometries.
- `phase7d_restart_runs.csv`: restart-level optimization records.
- `phase7d_exact_hamiltonian_audit.csv`: exact reference checks.
- `phase7d_hamiltonian_construction_audit.csv`: independent construction audit.
- `phase7d_generator_gradient_audit.csv`: projected-generator gradient checks.
- `phase7d_rotation_reconstruction.csv`: orbital-rotation reconstruction residuals.
- `phase7d_full_statevector_spotcheck.json`: full 20-qubit Qiskit statevector comparison with the 100-dimensional sector simulation.
- `PHASE7D_REPORT.md`: generated human-readable confirmation report.

All energies are in Hartree unless a filename or column explicitly states otherwise. Bond lengths are in angstrom. Operator labels and global indices are frozen identifiers defined by the deposited support records and source code.

