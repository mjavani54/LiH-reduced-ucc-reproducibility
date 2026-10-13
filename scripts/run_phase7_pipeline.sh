#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:-$REPO_ROOT/work}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_ROOT"

python "$REPO_ROOT/src/lih_phase7a_cross_basis_transfer.py" \
  --phase6-conclusions "$REPO_ROOT/results/phase6/phase6_final_conclusions.json" \
  --output-dir "$RUN_ROOT/phase7a"

python "$REPO_ROOT/src/lih_phase7a1_orbital_embedding_diagnostic.py" \
  --phase7a-dir "$RUN_ROOT/phase7a" \
  --output-dir "$RUN_ROOT/phase7a1"

python "$REPO_ROOT/src/lih_phase7a2_rotated_hamiltonian_audit.py" \
  --phase7a-dir "$RUN_ROOT/phase7a" \
  --phase7a1-dir "$RUN_ROOT/phase7a1" \
  --output-dir "$RUN_ROOT/phase7a2"

python "$REPO_ROOT/src/lih_phase7b_cross_basis_support_transfer.py" \
  --phase7a-dir "$RUN_ROOT/phase7a" \
  --phase7a1-dir "$RUN_ROOT/phase7a1" \
  --phase7a2-dir "$RUN_ROOT/phase7a2" \
  --output-dir "$RUN_ROOT/phase7b" \
  --run-group all

python "$REPO_ROOT/src/lih_phase7c_expanded_space_augmentation.py" \
  --phase7a-dir "$RUN_ROOT/phase7a" \
  --phase7a1-dir "$RUN_ROOT/phase7a1" \
  --phase7a2-dir "$RUN_ROOT/phase7a2" \
  --phase7b-dir "$RUN_ROOT/phase7b" \
  --output-dir "$RUN_ROOT/phase7c" \
  --run-group all

python "$REPO_ROOT/src/lih_phase7d_frozen_qiskit_confirmation.py" \
  --phase7a-dir "$RUN_ROOT/phase7a" \
  --phase7a1-dir "$RUN_ROOT/phase7a1" \
  --phase7a2-dir "$RUN_ROOT/phase7a2" \
  --phase7c-dir "$RUN_ROOT/phase7c" \
  --output-dir "$RUN_ROOT/phase7d"

printf 'Phase-7 pipeline completed under %s\n' "$RUN_ROOT"

