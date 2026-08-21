#!/usr/bin/env python3
"""Phase 7A.2: audit the rotated matched 2e,5o LiH/6-31G Hamiltonian.

This program consumes the completed Phase-7A.1 orbital-embedding diagnostic,
reconstructs and freezes its rotated orbitals, reconciles the independent
PySCF and Qiskit molecular-orbital gauges, and builds the rotated active-space
Hamiltonian through Qiskit Nature. An independent PySCF CASCI calculation
cross-checks the total energy, while a frozen-core 2e,10o CASCI calculation
quantifies active-space truncation.

This is an audit-only phase. It performs no VQE or numerical optimization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import lih_phase7a_cross_basis_transfer as p7
    import lih_phase7a1_orbital_embedding_diagnostic as p7a1
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place this script beside lih_phase7a_cross_basis_transfer.py, "
        "lih_phase7a1_orbital_embedding_diagnostic.py, "
        "lih_phase3_geometry_transfer.py, "
        "lih_phase2b_exhaustive_support.py, and "
        "lih_reference_failure_diagnostics.py."
    ) from exc


SCHEMA_VERSION = 3
EXPECTED_PHASE7A1_STATUS = "complete_no_hamiltonian_no_exact_energy_no_vqe"
EXPECTED_PHASE7A1_GATE = (
    "build_phase7a2_rotated_matched_2e5o_hamiltonian_audit"
)
SOURCE_BASIS = "sto3g"
TARGET_BASIS = "6-31g"
ACTIVE_ELECTRONS = 2
ACTIVE_ORBITALS = 5
FULL_NONCORE_ORBITALS = 10
FULL_TARGET_ORBITALS = 11
EXPECTED_QUBITS = 10
EXPECTED_FULL_UCCSD_OPERATORS = 24
EXPECTED_COMPACT_OPERATORS = 10

STRICT_EQUIVALENCE_HA = 1e-6
CHEMICAL_ACCURACY_HA = 0.0016
GOOD_ACCURACY_HA = 0.005
ACCEPTABLE_ACCURACY_HA = 0.01


@dataclass
class InputBundle:
    phase7a_audit: dict[str, Any]
    phase7a_support: dict[str, Any]
    phase7a1_protocol: dict[str, Any]
    phase7a1_conclusions: dict[str, Any]
    rotation_frame: pd.DataFrame


@dataclass
class FrozenGeometry:
    bond_length: float
    target: Any
    active_weights_direct_mo: np.ndarray
    full_rotation_direct_mo: np.ndarray
    saved_active_weights_phase7a1_mo: np.ndarray
    raw_unaligned_rotation_weight_residual: float
    saved_active_orthonormality_residual: float
    target_mo_energy_spectrum_residual_ha: float
    target_mo_occupation_spectrum_residual: float
    embedding_quality_residual: float
    active_orthonormality_residual: float
    full_orthogonality_residual: float
    rotated_embedding_quality: float


@dataclass
class QiskitAuditSystem:
    bond_length: float
    raw_problem: Any
    rotated_full_problem: Any
    active_problem: Any
    mapper: Any
    qubit_op: Any
    constants: dict[str, float]
    total_offset_ha: float
    exact_active_electronic_ha: float
    exact_total_ha: float
    hf_active_electronic_ha: float
    hf_total_qubit_ha: float
    hf_total_driver_ha: float
    driver_coefficients: np.ndarray
    direct_to_driver_gauge: np.ndarray
    full_rotation_driver_mo: np.ndarray
    gauge_orthogonality_residual: float
    physical_rotated_orbital_residual: float
    hermiticity_residual: float
    fingerprint: str


def geometry_key(value: float) -> str:
    return f"{float(value):.6f}"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")


def validate_fingerprint(payload: dict[str, Any], field: str) -> None:
    supplied = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not supplied or supplied != p7.fingerprint(unsigned):
        raise RuntimeError(f"Invalid or modified fingerprint field: {field}")


def required_rotation_columns() -> list[str]:
    return [
        "bond_length_angstrom",
        "geometry_key",
        "target_full_mo_index",
        "target_mo_energy_ha",
        "target_mo_occupation",
        "rotated_active_orbital",
        "rotation_weight",
    ]


def load_inputs(phase7a_dir: Path, phase7a1_dir: Path) -> InputBundle:
    phase7a_audit_path = phase7a_dir / "phase7a_audit.json"
    phase7a_support_path = phase7a_dir / "phase7a_frozen_support.json"
    phase7a1_protocol_path = phase7a1_dir / "phase7a1_protocol.json"
    phase7a1_conclusions_path = (
        phase7a1_dir / "phase7a1_embedding_conclusions.json"
    )
    rotation_path = (
        phase7a1_dir / "phase7a1_rotated_orbital_coefficients.csv"
    )
    for path in (
        phase7a_audit_path,
        phase7a_support_path,
        phase7a1_protocol_path,
        phase7a1_conclusions_path,
        rotation_path,
    ):
        require_file(path)

    phase7a_audit = json.loads(phase7a_audit_path.read_text(encoding="utf-8"))
    phase7a_support = json.loads(
        phase7a_support_path.read_text(encoding="utf-8")
    )
    phase7a1_protocol = json.loads(
        phase7a1_protocol_path.read_text(encoding="utf-8")
    )
    phase7a1_conclusions = json.loads(
        phase7a1_conclusions_path.read_text(encoding="utf-8")
    )
    rotation_frame = pd.read_csv(rotation_path)

    if phase7a_audit.get("analysis_status") != "audit_failed_do_not_run_vqe":
        raise RuntimeError("Phase 7A.2 requires the documented failed Phase-7A audit.")
    if phase7a_audit.get("support_fingerprint") != phase7a_support.get(
        "support_fingerprint"
    ):
        raise RuntimeError("Phase-7A audit and support fingerprints differ.")
    validate_fingerprint(phase7a_support, "support_fingerprint")

    if phase7a1_conclusions.get("analysis_status") != EXPECTED_PHASE7A1_STATUS:
        raise RuntimeError("Phase 7A.1 did not complete under the audited status.")
    if (
        phase7a1_conclusions.get("recommended_next_gate")
        != EXPECTED_PHASE7A1_GATE
    ):
        raise RuntimeError("Phase 7A.1 did not authorize this Hamiltonian audit.")
    if not bool(phase7a1_conclusions.get("phase7a_failure_reproduced", False)):
        raise RuntimeError("Phase-7A failure was not reproduced by Phase 7A.1.")
    if not bool(
        phase7a1_conclusions.get(
            "rotated_embedding_all_audit_geometries_reliable", False
        )
    ):
        raise RuntimeError("Phase-7A.1 rotated embeddings are not all reliable.")
    if not bool(
        phase7a1_conclusions.get(
            "rotated_embedding_all_geometry_transitions_reliable", False
        )
    ):
        raise RuntimeError("Phase-7A.1 rotated geometry continuity failed.")
    for field in (
        "no_target_active_hamiltonian_built",
        "no_exact_target_energy_evaluated",
        "no_vqe_performed",
    ):
        if not bool(phase7a1_conclusions.get(field, False)):
            raise RuntimeError(f"Phase-7A.1 input does not certify {field}.")

    validate_fingerprint(phase7a1_protocol, "diagnostic_protocol_fingerprint")
    if phase7a1_conclusions.get(
        "diagnostic_protocol_fingerprint"
    ) != phase7a1_protocol.get("diagnostic_protocol_fingerprint"):
        raise RuntimeError("Phase-7A.1 protocol and conclusions fingerprints differ.")
    if phase7a1_conclusions.get(
        "phase7a_support_fingerprint"
    ) != phase7a_support.get("support_fingerprint"):
        raise RuntimeError("Phase-7A.1 and Phase-7A support fingerprints differ.")
    if not bool(phase7a1_conclusions.get("threshold_was_not_changed", False)):
        raise RuntimeError("Phase-7A.1 did not preserve the overlap threshold.")

    if list(rotation_frame.columns) != required_rotation_columns():
        raise RuntimeError(
            "Phase-7A.1 rotation CSV columns or ordering differ from the "
            "audited schema."
        )
    if rotation_frame.duplicated(
        subset=[
            "geometry_key",
            "target_full_mo_index",
            "rotated_active_orbital",
        ]
    ).any():
        raise RuntimeError("Phase-7A.1 rotation CSV contains duplicate coefficients.")
    if not np.isfinite(rotation_frame["rotation_weight"].to_numpy(float)).all():
        raise RuntimeError("Phase-7A.1 rotation CSV contains nonfinite weights.")

    expected_bonds = sorted(
        float(value) for value in phase7a_support["target_bond_lengths_angstrom"]
    )
    actual_bonds = sorted(
        float(value) for value in rotation_frame["bond_length_angstrom"].unique()
    )
    if expected_bonds != actual_bonds:
        raise RuntimeError("Rotation-coefficient geometry coverage is incomplete.")
    expected_rows = len(expected_bonds) * FULL_TARGET_ORBITALS * ACTIVE_ORBITALS
    if len(rotation_frame) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rotation rows; found {len(rotation_frame)}."
        )
    return InputBundle(
        phase7a_audit=phase7a_audit,
        phase7a_support=phase7a_support,
        phase7a1_protocol=phase7a1_protocol,
        phase7a1_conclusions=phase7a1_conclusions,
        rotation_frame=rotation_frame,
    )


def rotation_csv_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = frame.sort_values(
        ["bond_length_angstrom", "target_full_mo_index", "rotated_active_orbital"]
    )
    return [
        {
            "bond_length_angstrom": float(row.bond_length_angstrom),
            "target_full_mo_index": int(row.target_full_mo_index),
            "rotated_active_orbital": int(row.rotated_active_orbital),
            "rotation_weight": float(row.rotation_weight),
        }
        for row in ordered.itertuples(index=False)
    ]


def weights_from_frame(frame: pd.DataFrame, bond_length: float) -> np.ndarray:
    subset = frame[
        np.isclose(
            frame["bond_length_angstrom"].to_numpy(float),
            float(bond_length),
            atol=1e-9,
        )
    ]
    weights = np.full(
        (FULL_TARGET_ORBITALS, ACTIVE_ORBITALS),
        np.nan,
        dtype=float,
    )
    for row in subset.itertuples(index=False):
        full_mo = int(row.target_full_mo_index)
        active = int(row.rotated_active_orbital)
        if full_mo not in range(FULL_TARGET_ORBITALS):
            raise RuntimeError("Rotation CSV contains an invalid full-MO index.")
        if active not in range(ACTIVE_ORBITALS):
            raise RuntimeError("Rotation CSV contains an invalid active-orbital index.")
        weights[full_mo, active] = float(row.rotation_weight)
    if not np.isfinite(weights).all():
        raise RuntimeError(f"Incomplete rotation matrix at R={bond_length}.")
    return weights


def rotation_metadata_from_frame(
    frame: pd.DataFrame,
    bond_length: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the Phase-7A.1 target spectrum without assuming its MO gauge.

    The rotation CSV repeats each target-MO energy and occupation once per
    rotated active orbital.  Those metadata are gauge invariant within an
    exactly degenerate block even though the saved rotation weights are not.
    """
    subset = frame[
        np.isclose(
            frame["bond_length_angstrom"].to_numpy(float),
            float(bond_length),
            atol=1e-9,
        )
    ]
    energies = np.full(FULL_TARGET_ORBITALS, np.nan, dtype=float)
    occupations = np.full(FULL_TARGET_ORBITALS, np.nan, dtype=float)
    for full_mo in range(FULL_TARGET_ORBITALS):
        rows = subset[subset["target_full_mo_index"] == full_mo]
        if len(rows) != ACTIVE_ORBITALS:
            raise RuntimeError(
                f"Incomplete repeated target-MO metadata at R={bond_length}, "
                f"MO={full_mo}."
            )
        energy_values = rows["target_mo_energy_ha"].to_numpy(float)
        occupation_values = rows["target_mo_occupation"].to_numpy(float)
        if (
            float(np.ptp(energy_values)) > tolerance
            or float(np.ptp(occupation_values)) > tolerance
        ):
            raise RuntimeError(
                f"Inconsistent repeated target-MO metadata at R={bond_length}, "
                f"MO={full_mo}."
            )
        energies[full_mo] = float(energy_values[0])
        occupations[full_mo] = float(occupation_values[0])
    if not np.isfinite(energies).all() or not np.isfinite(occupations).all():
        raise RuntimeError(f"Nonfinite target-MO metadata at R={bond_length}.")
    return energies, occupations


def occupation_resolved_spectrum_residual(
    saved_energies: np.ndarray,
    saved_occupations: np.ndarray,
    current_energies: np.ndarray,
    current_occupations: np.ndarray,
) -> tuple[float, float]:
    """Compare canonical spectra while allowing rotations/reordering in blocks."""
    energy_residuals: list[float] = []
    occupation_residuals: list[float] = []
    for nominal_occupation in (2.0, 0.0):
        saved_mask = np.isclose(saved_occupations, nominal_occupation, atol=1e-8)
        current_mask = np.isclose(
            current_occupations, nominal_occupation, atol=1e-8
        )
        if int(np.sum(saved_mask)) != int(np.sum(current_mask)):
            raise RuntimeError("Phase-7A.1 and current RHF occupation blocks differ.")
        saved_block = np.sort(saved_energies[saved_mask])
        current_block = np.sort(current_energies[current_mask])
        if saved_block.shape != current_block.shape:
            raise RuntimeError("Phase-7A.1 and current RHF spectra differ in size.")
        if saved_block.size:
            energy_residuals.append(
                float(np.max(np.abs(saved_block - current_block)))
            )
        occupation_residuals.append(
            float(
                max(
                    np.max(np.abs(saved_occupations[saved_mask] - nominal_occupation)),
                    np.max(
                        np.abs(
                            current_occupations[current_mask]
                            - nominal_occupation
                        )
                    ),
                )
            )
        )
    return max(energy_residuals, default=0.0), max(
        occupation_residuals, default=0.0
    )


def canonicalize_column_signs(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    result = np.asarray(matrix, dtype=float).copy()
    for column in range(result.shape[1]):
        nonzero = np.flatnonzero(np.abs(result[:, column]) > tolerance)
        if len(nonzero) and result[int(nonzero[0]), column] < 0:
            result[:, column] *= -1.0
    return result


def complete_full_rotation(
    active_weights: np.ndarray,
    target: p7.TargetRHFData,
    tolerance: float,
) -> tuple[np.ndarray, float, float]:
    occupations = np.asarray(target.mean_field.mo_occ, dtype=float)
    if active_weights.shape != (FULL_TARGET_ORBITALS, ACTIVE_ORBITALS):
        raise RuntimeError("Unexpected active rotation shape.")
    core = list(range(target.num_frozen_core_orbitals))
    occupied = [
        index
        for index in range(target.num_frozen_core_orbitals, len(occupations))
        if np.isclose(occupations[index], 2.0, atol=1e-8)
    ]
    virtuals = [
        index for index, value in enumerate(occupations)
        if np.isclose(value, 0.0, atol=1e-8)
    ]
    if core != [0] or occupied != [1] or len(virtuals) != 9:
        raise RuntimeError("Unexpected frozen-core 6-31G occupation structure.")
    if np.max(np.abs(active_weights[0, :])) > tolerance:
        raise RuntimeError("A rotated active orbital contains frozen-core weight.")
    expected_occupied = np.zeros(FULL_TARGET_ORBITALS)
    expected_occupied[occupied[0]] = 1.0
    if min(
        np.max(np.abs(active_weights[:, 0] - expected_occupied)),
        np.max(np.abs(active_weights[:, 0] + expected_occupied)),
    ) > tolerance:
        raise RuntimeError("The rotated active occupied orbital changed the RHF reference.")
    virtual_weights = active_weights[np.ix_(virtuals, range(1, 5))]
    if np.max(np.abs(active_weights[occupied[0], 1:])) > tolerance:
        raise RuntimeError("A rotated virtual orbital contains occupied-MO weight.")
    active_residual = float(
        np.max(np.abs(active_weights.T @ active_weights - np.eye(5)))
    )
    if active_residual > tolerance:
        raise RuntimeError("Phase-7A.1 active rotation is not orthonormal.")

    q_complete, _ = np.linalg.qr(virtual_weights, mode="complete")
    complement = canonicalize_column_signs(q_complete[:, 4:])
    full_rotation = np.zeros((FULL_TARGET_ORBITALS, FULL_TARGET_ORBITALS))
    full_rotation[core[0], 0] = 1.0
    full_rotation[:, 1:6] = active_weights
    full_rotation[np.ix_(virtuals, range(6, 11))] = complement
    full_residual = float(
        np.max(
            np.abs(
                full_rotation.T @ full_rotation
                - np.eye(FULL_TARGET_ORBITALS)
            )
        )
    )
    if full_residual > tolerance:
        raise RuntimeError("Completed full target rotation is not orthogonal.")
    return full_rotation, active_residual, full_residual


def reconstruct_frozen_geometries(
    bundle: InputBundle,
    reconstruction_tolerance: float,
) -> tuple[dict[str, FrozenGeometry], list[dict[str, Any]]]:
    threshold = float(bundle.phase7a_support["orbital_overlap_threshold"])
    degeneracy_tolerance = float(
        bundle.phase7a_support["degeneracy_tolerance_ha"]
    )
    tracking_bonds = sorted(
        float(value)
        for value in bundle.phase7a_support[
            "source_tracking_bond_lengths_angstrom"
        ]
    )
    audit_bonds = sorted(
        float(value)
        for value in bundle.phase7a_support["target_bond_lengths_angstrom"]
    )
    source_snapshots, source_groups, _, source_summary = p7a1.build_snapshots(
        tracking_bonds,
        SOURCE_BASIS,
        ACTIVE_ORBITALS,
        threshold,
        degeneracy_tolerance,
    )
    if not bool(source_summary["tracking_reliable"].all()):
        raise RuntimeError("Reconstructed source-orbital tracking is unreliable.")

    conclusion_by_key = {
        str(item["geometry_key"]): item
        for item in bundle.phase7a1_conclusions["geometry_results"]
    }
    frozen: dict[str, FrozenGeometry] = {}
    rows: list[dict[str, Any]] = []
    for bond in audit_bonds:
        key = geometry_key(bond)
        target = p7.build_target_rhf(bond, TARGET_BASIS)
        cross_overlap = p7a1.cross_overlap_matrix(source_snapshots[key], target)
        reconstructed = p7a1.build_rotated_embedding(
            bond,
            cross_overlap,
            target,
            source_groups,
            threshold,
        )
        saved_weights = weights_from_frame(bundle.rotation_frame, bond)
        raw_unaligned_residual = float(
            np.max(
                np.abs(
                    saved_weights
                    - reconstructed.weights_in_full_target_mos
                )
            )
        )

        # Phase 7A.1 and Phase 7A.2 are separate RHF runs.  Canonical virtual
        # orbitals may therefore undergo an arbitrary orthogonal rotation
        # inside a degenerate or nearly degenerate block.  The raw MO-basis
        # weights are gauge dependent and must not be compared elementwise as
        # a pass/fail condition.  Validate the saved matrix internally, verify
        # the gauge-invariant target spectrum and embedding quality, and use
        # the freshly reconstructed weights in this run's exact RHF gauge.
        _, saved_active_residual, _ = complete_full_rotation(
            saved_weights,
            target,
            reconstruction_tolerance,
        )
        saved_energies, saved_occupations = rotation_metadata_from_frame(
            bundle.rotation_frame,
            bond,
            reconstruction_tolerance,
        )
        spectrum_residual, occupation_residual = (
            occupation_resolved_spectrum_residual(
                saved_energies,
                saved_occupations,
                np.asarray(target.mean_field.mo_energy, dtype=float),
                np.asarray(target.mean_field.mo_occ, dtype=float),
            )
        )
        if spectrum_residual > reconstruction_tolerance:
            raise RuntimeError(
                f"Phase-7A.1 target MO spectrum failed reproduction at "
                f"R={bond}: {spectrum_residual:.3e} Ha."
            )
        if occupation_residual > reconstruction_tolerance:
            raise RuntimeError(
                f"Phase-7A.1 target occupations failed reproduction at "
                f"R={bond}: {occupation_residual:.3e}."
            )
        full_rotation, active_residual, full_residual = complete_full_rotation(
            reconstructed.weights_in_full_target_mos,
            target,
            reconstruction_tolerance,
        )
        expected_quality = float(
            conclusion_by_key[key]["rotated_2e5o_quality"]
        )
        quality_residual = abs(
            reconstructed.minimum_assigned_group_quality - expected_quality
        )
        if quality_residual > reconstruction_tolerance:
            raise RuntimeError(
                f"Rotated embedding quality failed reproduction at R={bond}."
            )
        frozen[key] = FrozenGeometry(
            bond_length=bond,
            # Reuse this exact RHF object for the Hamiltonian audit. Rebuilding
            # it could choose a different gauge inside a degenerate virtual
            # subspace even when the underlying physics is unchanged.
            target=target,
            active_weights_direct_mo=(
                reconstructed.weights_in_full_target_mos
            ),
            full_rotation_direct_mo=full_rotation,
            saved_active_weights_phase7a1_mo=saved_weights,
            raw_unaligned_rotation_weight_residual=raw_unaligned_residual,
            saved_active_orthonormality_residual=saved_active_residual,
            target_mo_energy_spectrum_residual_ha=spectrum_residual,
            target_mo_occupation_spectrum_residual=occupation_residual,
            embedding_quality_residual=quality_residual,
            active_orthonormality_residual=active_residual,
            full_orthogonality_residual=full_residual,
            rotated_embedding_quality=(
                reconstructed.minimum_assigned_group_quality
            ),
        )
        rows.append(
            {
                "bond_length_angstrom": bond,
                "geometry_key": key,
                "raw_rotation_weight_residual_in_unaligned_mo_gauges": (
                    raw_unaligned_residual
                ),
                "raw_rotation_weight_equality_required": False,
                "saved_rotation_active_orthonormality_residual": (
                    saved_active_residual
                ),
                "target_mo_energy_spectrum_residual_ha": spectrum_residual,
                "target_mo_occupation_spectrum_residual": (
                    occupation_residual
                ),
                "active_orthonormality_residual": active_residual,
                "full_rotation_orthogonality_residual": full_residual,
                "rotated_embedding_quality": (
                    reconstructed.minimum_assigned_group_quality
                ),
                "quality_residual_vs_phase7a1": quality_residual,
                "reconstruction_passed": bool(
                    quality_residual <= reconstruction_tolerance
                    and spectrum_residual <= reconstruction_tolerance
                    and occupation_residual <= reconstruction_tolerance
                    and saved_active_residual <= reconstruction_tolerance
                    and active_residual <= reconstruction_tolerance
                    and full_residual <= reconstruction_tolerance
                ),
            }
        )
    return frozen, rows


def compact_excitations() -> tuple[
    list[diag.Excitation], list[diag.Excitation], list[diag.Excitation]
]:
    singles, doubles, full_pool = p7.excitation_pools()
    indices = p7.compact_reference_indices()
    compact = [full_pool[index] for index in indices]
    if len(full_pool) != EXPECTED_FULL_UCCSD_OPERATORS:
        raise RuntimeError("Rotated full UCCSD pool is not full24.")
    if len(compact) != EXPECTED_COMPACT_OPERATORS or len(set(compact)) != 10:
        raise RuntimeError("Rotated compact support is not ten unique operators.")
    if [p7.operator_label(index) for index in indices] != list(p7.COMPACT_LABELS):
        raise RuntimeError("Rotated compact operator ordering changed.")
    return full_pool, compact, [*singles, *doubles]


def build_frozen_support_payload(
    bundle: InputBundle,
    frozen: dict[str, FrozenGeometry],
) -> dict[str, Any]:
    full_pool, compact, regenerated_pool = compact_excitations()
    if full_pool != regenerated_pool:
        raise RuntimeError("Independent full24 pool regeneration changed.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": "Phase 7A.2 rotated Hamiltonian audit",
        "selection_frozen_before_target_exact_energy_evaluation": True,
        "phase7a_support_fingerprint": bundle.phase7a_support[
            "support_fingerprint"
        ],
        "phase7a1_diagnostic_protocol_fingerprint": (
            bundle.phase7a1_protocol["diagnostic_protocol_fingerprint"]
        ),
        "phase7a1_rotation_csv_fingerprint": p7.fingerprint(
            rotation_csv_payload(bundle.rotation_frame)
        ),
        "source_basis": SOURCE_BASIS,
        "target_basis": TARGET_BASIS,
        "active_electrons": ACTIVE_ELECTRONS,
        "active_spatial_orbitals": ACTIVE_ORBITALS,
        "full_target_spatial_orbitals": FULL_TARGET_ORBITALS,
        "full_noncore_spatial_orbitals": FULL_NONCORE_ORBITALS,
        "rotation_completion_rule": (
            "Reconstruct the Phase-7A.1 block-preserving maximum-overlap "
            "embedding in the current RHF MO gauge; use the core canonical "
            "MO and five reconstructed active columns, then a "
            "sign-canonicalized complete-QR complement of the target virtual "
            "block. Raw Phase-7A.1 MO-basis weights are retained only as "
            "gauge-dependent provenance."
        ),
        "cross_run_mo_gauge_policy": (
            "Separate RHF runs may rotate degenerate virtual orbitals. "
            "Elementwise equality of raw MO-basis rotation weights is not a "
            "scientific pass/fail condition; occupation-resolved canonical "
            "spectra, saved-matrix structure, and rotated-embedding quality "
            "are reproduced instead."
        ),
        "qiskit_gauge_rule": (
            "Re-express the frozen physical rotation in the Qiskit-driver MO "
            "gauge using the full canonical-MO overlap matrix."
        ),
        "target_bond_lengths_angstrom": [
            frozen[key].bond_length for key in sorted(frozen, key=float)
        ],
        "geometry_rotations": {
            key: {
                "bond_length_angstrom": item.bond_length,
                "direct_target_mo_coefficients_in_ao_basis": (
                    np.asarray(
                        item.target.mean_field.mo_coeff, dtype=float
                    ).tolist()
                ),
                "active_weights_in_direct_target_mo_basis": (
                    item.active_weights_direct_mo.tolist()
                ),
                "active_orbital_coefficients_in_ao_basis": (
                    np.asarray(item.target.mean_field.mo_coeff, dtype=float)
                    @ item.active_weights_direct_mo
                ).tolist(),
                "completed_full_rotation_in_direct_target_mo_basis": (
                    item.full_rotation_direct_mo.tolist()
                ),
                "rotated_embedding_quality": item.rotated_embedding_quality,
                "phase7a1_raw_unaligned_weight_residual": (
                    item.raw_unaligned_rotation_weight_residual
                ),
                "target_mo_energy_spectrum_residual_ha": (
                    item.target_mo_energy_spectrum_residual_ha
                ),
                "embedding_quality_residual": (
                    item.embedding_quality_residual
                ),
            }
            for key, item in sorted(frozen.items(), key=lambda pair: float(pair[0]))
        },
        "compact_operator_labels": list(p7.COMPACT_LABELS),
        "compact_reference_global_uccsd_indices": p7.compact_reference_indices(),
        "compact_excitations": [
            diag.excitation_to_json(item) for item in compact
        ],
        "full24_excitations": [
            diag.excitation_to_json(item) for item in full_pool
        ],
        "energy_thresholds_ha": {
            "strict_equivalence": STRICT_EQUIVALENCE_HA,
            "chemical": CHEMICAL_ACCURACY_HA,
            "good": GOOD_ACCURACY_HA,
            "acceptable": ACCEPTABLE_ACCURACY_HA,
        },
        "no_vqe_performed": True,
        "claim_boundary": (
            "This frozen support defines a rotated matched 2e,5o target "
            "Hamiltonian audit. It does not establish VQE accuracy, operator-"
            "support transfer, expanded-active-space equivalence, or transfer "
            "to another molecule."
        ),
    }
    payload["rotated_support_fingerprint"] = p7.fingerprint(payload)
    return payload


def write_or_validate_frozen_support(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("rotated_support_fingerprint") != payload.get(
            "rotated_support_fingerprint"
        ):
            raise RuntimeError(
                f"Existing rotated support {path} differs. Use a new output "
                "directory rather than overwriting a frozen support."
            )
        return
    write_json(path, payload)


def write_or_validate_protocol(path: Path, payload: dict[str, Any]) -> None:
    """Freeze a preregistered protocol instead of silently replacing it."""
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != payload.get(
            "protocol_fingerprint"
        ):
            raise RuntimeError(
                f"Existing preregistered protocol {path} differs. Use a new "
                "output directory rather than overwriting it."
            )
        return
    write_json(path, payload)


def extract_driver_coefficients(schema: Any) -> np.ndarray:
    flat_value = getattr(schema.wavefunction, "scf_orbitals_a", None)
    if flat_value is None:
        raise RuntimeError("Qiskit PySCF schema lacks MO coefficients.")
    nmo = int(schema.properties.calcinfo_nmo)
    flat = np.asarray(flat_value, dtype=float)
    if flat.size % nmo:
        raise RuntimeError("Qiskit PySCF MO coefficient array has invalid size.")
    coefficients = flat.reshape(flat.size // nmo, nmo)
    if coefficients.shape != (FULL_TARGET_ORBITALS, FULL_TARGET_ORBITALS):
        raise RuntimeError(
            f"Expected an 11x11 Qiskit-driver MO matrix; found "
            f"{coefficients.shape}."
        )
    return coefficients


def sparse_hermiticity_residual(qubit_op: Any) -> float:
    matrix = qubit_op.to_matrix(sparse=True)
    residual = matrix - matrix.getH()
    if residual.nnz == 0:
        return 0.0
    return float(np.max(np.abs(residual.data)))


def dense_array(value: Any) -> np.ndarray:
    """Return a numerical ndarray from a Qiskit Tensor or sparse container."""
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "array"):
        value = value.array
    result = np.asarray(value)
    if result.dtype == object:
        raise RuntimeError("A Qiskit integral tensor did not convert to numbers.")
    return result


def unfold_problem_two_body_integrals(problem: Any) -> Any:
    """Make symmetry-compressed ERIs explicit before BasisTransformer einsum.

    Qiskit Nature can store PySCF two-electron integrals as S4/S8 tensors.
    NumPy 2.x dispatch through these compressed Tensor subclasses is not
    reliable in every supported environment.  Rebuilding ElectronicIntegrals
    from explicit four-index arrays is representation-only: it does not alter
    the Hamiltonian.
    """
    from qiskit_nature.second_q.operators import ElectronicIntegrals
    from qiskit_nature.second_q.operators.symmetric_two_body import unfold

    integrals = problem.hamiltonian.electronic_integrals

    def one_body(polynomial: Any) -> np.ndarray | None:
        value = polynomial.get("+-", None)
        return None if value is None else dense_array(value)

    def two_body(polynomial: Any) -> np.ndarray | None:
        value = polynomial.get("++--", None)
        if value is None:
            return None
        result = dense_array(unfold(value))
        if result.ndim != 4:
            raise RuntimeError(
                "Qiskit two-electron integrals did not unfold to rank four."
            )
        return result

    h1_a = one_body(integrals.alpha)
    h2_aa = two_body(integrals.alpha)
    if h1_a is None or h2_aa is None:
        raise RuntimeError("Qiskit alpha-spin electronic integrals are incomplete.")

    h1_b = None if integrals.beta.is_empty() else one_body(integrals.beta)
    h2_bb = None if integrals.beta.is_empty() else two_body(integrals.beta)
    h2_ba = (
        None
        if integrals.beta_alpha.is_empty()
        else two_body(integrals.beta_alpha)
    )
    explicit = ElectronicIntegrals.from_raw_integrals(
        h1_a,
        h2_aa,
        h1_b=h1_b,
        h2_bb=h2_bb,
        h2_ba=h2_ba,
        validate=True,
        auto_index_order=True,
    )
    explicit_h2 = dense_array(explicit.alpha["++--"])
    if explicit_h2.ndim != 4:
        raise RuntimeError(
            "Explicit Qiskit alpha-spin two-electron integrals are not rank four."
        )

    result = copy.deepcopy(problem)
    result.hamiltonian.electronic_integrals = explicit
    return result


def build_qiskit_system(
    frozen: FrozenGeometry,
    target: p7.TargetRHFData,
    audit_tolerance: float,
) -> QiskitAuditSystem:
    from qiskit.quantum_info import Statevector
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import ElectronicIntegrals
    from qiskit_nature.second_q.problems import ElectronicBasis
    from qiskit_nature.second_q.transformers import (
        ActiveSpaceTransformer,
        BasisTransformer,
    )
    from qiskit_nature.units import DistanceUnit

    bond = frozen.bond_length
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {bond}",
        basis=TARGET_BASIS,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
        conv_tol=1e-10,
    )
    driver.run_pyscf()
    schema = driver.to_qcschema(include_dipole=False)
    raw_problem = driver.to_problem(
        basis=ElectronicBasis.MO,
        include_dipole=False,
    )
    raw_problem = unfold_problem_two_body_integrals(raw_problem)
    driver_coefficients = extract_driver_coefficients(schema)
    direct_coefficients = np.asarray(target.mean_field.mo_coeff, dtype=float)
    ao_overlap = np.asarray(target.mean_field.get_ovlp(), dtype=float)
    gauge = direct_coefficients.T @ ao_overlap @ driver_coefficients
    gauge_residual = float(
        np.max(np.abs(gauge.T @ gauge - np.eye(FULL_TARGET_ORBITALS)))
    )
    if gauge_residual > audit_tolerance:
        raise RuntimeError(
            f"Direct and Qiskit target MO bases are inconsistent at R={bond}."
        )
    driver_rotation = gauge.T @ frozen.full_rotation_direct_mo
    physical_direct = direct_coefficients @ frozen.full_rotation_direct_mo
    physical_driver = driver_coefficients @ driver_rotation
    difference = physical_direct - physical_driver
    orbital_norms = np.sqrt(
        np.maximum(
            0.0,
            np.diag(difference.T @ ao_overlap @ difference),
        )
    )
    physical_residual = float(np.max(orbital_norms))
    if physical_residual > audit_tolerance:
        raise RuntimeError(
            f"Gauge-reconciled physical orbitals disagree at R={bond}."
        )

    coefficients = ElectronicIntegrals.from_raw_integrals(
        driver_rotation,
        h1_b=driver_rotation,
    )
    basis_transformer = BasisTransformer(
        ElectronicBasis.MO,
        ElectronicBasis.MO,
        coefficients,
    )
    rotated_full_problem = basis_transformer.transform(copy.deepcopy(raw_problem))
    active_transformer = ActiveSpaceTransformer(
        num_electrons=(1, 1),
        num_spatial_orbitals=ACTIVE_ORBITALS,
        active_orbitals=[1, 2, 3, 4, 5],
    )
    active_problem = active_transformer.transform(rotated_full_problem)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(active_problem.hamiltonian.second_q_op())
    constants = {
        str(key): float(np.real(value))
        for key, value in active_problem.hamiltonian.constants.items()
    }
    total_offset = float(sum(constants.values()))

    exact_solver = NumPyMinimumEigensolver(
        filter_criterion=active_problem.get_default_filter_criterion()
    )
    exact_result = GroundStateEigensolver(mapper, exact_solver).solve(
        active_problem
    )
    exact_total = diag.total_energy_from_result(exact_result)
    exact_active = diag.raw_eigenvalue_from_result(exact_result)

    hf_circuit = HartreeFock(
        active_problem.num_spatial_orbitals,
        active_problem.num_particles,
        mapper,
    )
    hf_state = Statevector.from_instruction(hf_circuit)
    hf_active = float(np.real(hf_state.expectation_value(qubit_op)))
    hf_total_qubit = hf_active + total_offset
    if active_problem.reference_energy is None:
        raise RuntimeError("Rotated target problem lacks an HF reference energy.")
    hf_total_driver = float(np.real(active_problem.reference_energy))
    hermiticity = sparse_hermiticity_residual(qubit_op)
    configuration = {
        "atom": f"Li 0 0 0; H 0 0 {bond}",
        "basis": TARGET_BASIS,
        "rotation": "phase7a1_block_preserving_maximum_overlap",
        "full_rotation_direct_mo": frozen.full_rotation_direct_mo.tolist(),
        "active_full_rotated_indices": [1, 2, 3, 4, 5],
        "active_particles": [1, 1],
        "active_spatial_orbitals": ACTIVE_ORBITALS,
        "mapper": "JordanWignerMapper",
    }
    return QiskitAuditSystem(
        bond_length=bond,
        raw_problem=raw_problem,
        rotated_full_problem=rotated_full_problem,
        active_problem=active_problem,
        mapper=mapper,
        qubit_op=qubit_op,
        constants=constants,
        total_offset_ha=total_offset,
        exact_active_electronic_ha=exact_active,
        exact_total_ha=exact_total,
        hf_active_electronic_ha=hf_active,
        hf_total_qubit_ha=hf_total_qubit,
        hf_total_driver_ha=hf_total_driver,
        driver_coefficients=driver_coefficients,
        direct_to_driver_gauge=gauge,
        full_rotation_driver_mo=driver_rotation,
        gauge_orthogonality_residual=gauge_residual,
        physical_rotated_orbital_residual=physical_residual,
        hermiticity_residual=hermiticity,
        fingerprint=p7.fingerprint(configuration),
    )


def pyscf_casci_benchmarks(
    target: p7.TargetRHFData,
    frozen: FrozenGeometry,
) -> dict[str, float]:
    from pyscf import mcscf

    direct_coefficients = np.asarray(target.mean_field.mo_coeff, dtype=float)
    rotated_coefficients = direct_coefficients @ frozen.full_rotation_direct_mo

    rotated_casci = mcscf.CASCI(
        target.mean_field,
        ACTIVE_ORBITALS,
        (1, 1),
    )
    rotated_casci.verbose = 0
    rotated_casci.fcisolver.conv_tol = 1e-12
    rotated_total = float(rotated_casci.kernel(rotated_coefficients)[0])

    full_noncore_casci = mcscf.CASCI(
        target.mean_field,
        FULL_NONCORE_ORBITALS,
        (1, 1),
    )
    full_noncore_casci.verbose = 0
    full_noncore_casci.fcisolver.conv_tol = 1e-12
    full_noncore_total = float(
        full_noncore_casci.kernel(direct_coefficients)[0]
    )
    return {
        "pyscf_rotated_2e5o_casci_total_ha": rotated_total,
        "pyscf_full_noncore_2e10o_casci_total_ha": full_noncore_total,
        "rotated_2e5o_error_vs_full_noncore_ha": (
            rotated_total - full_noncore_total
        ),
    }


def build_ansatz(system: QiskitAuditSystem, excitations: Sequence[diag.Excitation]) -> Any:
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

    initial_state = HartreeFock(
        system.active_problem.num_spatial_orbitals,
        system.active_problem.num_particles,
        system.mapper,
    )
    ansatz = UCC(
        num_spatial_orbitals=system.active_problem.num_spatial_orbitals,
        num_particles=system.active_problem.num_particles,
        excitations=diag.custom_excitation_generator(excitations),
        qubit_mapper=system.mapper,
        preserve_spin=True,
        reps=1,
        initial_state=initial_state,
    )
    _ = ansatz.num_parameters
    if ansatz.num_parameters != len(excitations):
        raise RuntimeError("Rotated UCC parameter count differs from its support.")
    return ansatz


def nullable_resources() -> dict[str, int | None]:
    return {
        "logical_depth": None,
        "logical_size": None,
        "compiled_depth": None,
        "compiled_size": None,
        "compiled_cx": None,
    }


def classify_truncation_error(error_ha: float) -> str:
    value = abs(float(error_ha))
    if value <= CHEMICAL_ACCURACY_HA:
        return "chemical"
    if value <= GOOD_ACCURACY_HA:
        return "good"
    if value <= ACCEPTABLE_ACCURACY_HA:
        return "acceptable"
    return "above_acceptable"


def audit_geometry(
    system: QiskitAuditSystem,
    target: p7.TargetRHFData,
    frozen: FrozenGeometry,
    benchmarks: dict[str, float],
    audit_tolerance: float,
    skip_resources: bool,
    seed_transpiler: int,
) -> dict[str, Any]:
    full_pool, compact, _ = compact_excitations()
    qiskit_vs_pyscf = abs(
        system.exact_total_ha
        - benchmarks["pyscf_rotated_2e5o_casci_total_ha"]
    )
    exact_reconstruction = abs(
        system.exact_active_electronic_ha
        + system.total_offset_ha
        - system.exact_total_ha
    )
    hf_reconstruction = abs(
        system.hf_total_qubit_ha - system.hf_total_driver_ha
    )
    direct_hf_residual = abs(
        system.hf_total_driver_ha - float(target.mean_field.e_tot)
    )
    truncation_error = benchmarks["rotated_2e5o_error_vs_full_noncore_ha"]
    if skip_resources:
        full_resources = nullable_resources()
        compact_resources = nullable_resources()
    else:
        full_resources = diag.circuit_resource_metrics(
            build_ansatz(system, full_pool), seed_transpiler
        )
        compact_resources = diag.circuit_resource_metrics(
            build_ansatz(system, compact), seed_transpiler
        )
    checks = {
        "phase7a1_embedding_reconstructed_gauge_invariantly": (
            frozen.saved_active_orthonormality_residual <= audit_tolerance
            and frozen.target_mo_energy_spectrum_residual_ha
            <= audit_tolerance
            and frozen.target_mo_occupation_spectrum_residual
            <= audit_tolerance
            and frozen.embedding_quality_residual <= audit_tolerance
        ),
        "raw_cross_run_mo_weight_equality_not_required": True,
        "active_rotation_orthonormal": (
            frozen.active_orthonormality_residual <= audit_tolerance
        ),
        "completed_full_rotation_orthogonal": (
            frozen.full_orthogonality_residual <= audit_tolerance
        ),
        "direct_and_qiskit_mo_gauges_reconciled": (
            system.gauge_orthogonality_residual <= audit_tolerance
            and system.physical_rotated_orbital_residual <= audit_tolerance
        ),
        "rotated_active_electrons_are_2": (
            int(sum(system.active_problem.num_particles)) == ACTIVE_ELECTRONS
        ),
        "rotated_active_spatial_orbitals_are_5": (
            int(system.active_problem.num_spatial_orbitals) == ACTIVE_ORBITALS
        ),
        "rotated_qubits_are_10": (
            int(system.qubit_op.num_qubits) == EXPECTED_QUBITS
        ),
        "qubit_hamiltonian_is_hermitian": (
            system.hermiticity_residual <= audit_tolerance
        ),
        "exact_total_reconstructs_from_offsets": (
            exact_reconstruction <= audit_tolerance
        ),
        "hf_qubit_matches_driver_reference": (
            hf_reconstruction <= audit_tolerance
        ),
        "qiskit_driver_hf_matches_direct_pyscf": (
            direct_hf_residual <= audit_tolerance
        ),
        "qiskit_exact_matches_independent_pyscf_casci": (
            qiskit_vs_pyscf <= audit_tolerance
        ),
        "rotated_2e5o_not_below_full_noncore_2e10o": (
            truncation_error >= -audit_tolerance
        ),
        "exact_not_above_hf": (
            system.exact_total_ha
            <= system.hf_total_driver_ha + audit_tolerance
        ),
        "full_pool_is_24_and_compact_pool_is_10": (
            len(full_pool) == EXPECTED_FULL_UCCSD_OPERATORS
            and len(compact) == EXPECTED_COMPACT_OPERATORS
        ),
    }
    return {
        "bond_length_angstrom": system.bond_length,
        "geometry_key": geometry_key(system.bond_length),
        "audit_passed": bool(all(checks.values())),
        "checks": checks,
        "rotated_configuration_fingerprint": system.fingerprint,
        "dimensions": {
            "full_target_spatial_orbitals": int(
                system.raw_problem.num_spatial_orbitals
            ),
            "full_noncore_spatial_orbitals": FULL_NONCORE_ORBITALS,
            "active_electrons": int(sum(system.active_problem.num_particles)),
            "active_spatial_orbitals": int(
                system.active_problem.num_spatial_orbitals
            ),
            "spin_orbitals": int(2 * system.active_problem.num_spatial_orbitals),
            "qubits": int(system.qubit_op.num_qubits),
            "pauli_terms": int(len(system.qubit_op)),
            "full_uccsd_operators": len(full_pool),
            "compact_operators": len(compact),
        },
        "energies_ha": {
            "qiskit_rotated_exact_active_electronic": (
                system.exact_active_electronic_ha
            ),
            "qiskit_rotated_constant_offsets": system.constants,
            "qiskit_rotated_total_constant_offset": system.total_offset_ha,
            "qiskit_rotated_exact_total": system.exact_total_ha,
            "qiskit_rotated_hf_total": system.hf_total_driver_ha,
            **benchmarks,
            "active_space_truncation_classification": classify_truncation_error(
                truncation_error
            ),
        },
        "numerical_residuals": {
            "phase7a1_gauge_invariant_reconstruction": max(
                frozen.saved_active_orthonormality_residual,
                frozen.target_mo_energy_spectrum_residual_ha,
                frozen.target_mo_occupation_spectrum_residual,
                frozen.embedding_quality_residual,
            ),
            "phase7a1_raw_rotation_weights_unaligned_mo_gauges": (
                frozen.raw_unaligned_rotation_weight_residual
            ),
            "phase7a1_saved_active_orthonormality": (
                frozen.saved_active_orthonormality_residual
            ),
            "target_mo_energy_spectrum_ha": (
                frozen.target_mo_energy_spectrum_residual_ha
            ),
            "target_mo_occupation_spectrum": (
                frozen.target_mo_occupation_spectrum_residual
            ),
            "rotated_embedding_quality": frozen.embedding_quality_residual,
            "active_orthonormality": frozen.active_orthonormality_residual,
            "full_rotation_orthogonality": frozen.full_orthogonality_residual,
            "qiskit_driver_gauge_orthogonality": (
                system.gauge_orthogonality_residual
            ),
            "physical_rotated_orbitals_direct_vs_qiskit": (
                system.physical_rotated_orbital_residual
            ),
            "qubit_hamiltonian_hermiticity": system.hermiticity_residual,
            "exact_offset_reconstruction_ha": exact_reconstruction,
            "hf_qubit_vs_driver_ha": hf_reconstruction,
            "driver_hf_vs_direct_pyscf_ha": direct_hf_residual,
            "qiskit_exact_vs_pyscf_casci_ha": qiskit_vs_pyscf,
        },
        "circuit_resources": {
            "full24": {
                "num_parameters": len(full_pool),
                **full_resources,
            },
            "transferred_compact10": {
                "num_parameters": len(compact),
                **compact_resources,
            },
        },
    }


def geometry_csv_row(item: dict[str, Any]) -> dict[str, Any]:
    energies = item["energies_ha"]
    residuals = item["numerical_residuals"]
    dimensions = item["dimensions"]
    resources = item["circuit_resources"]
    return {
        "bond_length_angstrom": item["bond_length_angstrom"],
        "geometry_key": item["geometry_key"],
        "audit_passed": item["audit_passed"],
        "rotated_configuration_fingerprint": item[
            "rotated_configuration_fingerprint"
        ],
        "qiskit_rotated_exact_total_ha": energies[
            "qiskit_rotated_exact_total"
        ],
        "pyscf_rotated_2e5o_casci_total_ha": energies[
            "pyscf_rotated_2e5o_casci_total_ha"
        ],
        "pyscf_full_noncore_2e10o_casci_total_ha": energies[
            "pyscf_full_noncore_2e10o_casci_total_ha"
        ],
        "rotated_2e5o_error_vs_full_noncore_ha": energies[
            "rotated_2e5o_error_vs_full_noncore_ha"
        ],
        "active_space_truncation_classification": energies[
            "active_space_truncation_classification"
        ],
        "qiskit_exact_vs_pyscf_casci_ha": residuals[
            "qiskit_exact_vs_pyscf_casci_ha"
        ],
        "rotation_reconstruction_residual": residuals[
            "phase7a1_gauge_invariant_reconstruction"
        ],
        "raw_unaligned_mo_weight_residual": residuals[
            "phase7a1_raw_rotation_weights_unaligned_mo_gauges"
        ],
        "gauge_reconciliation_residual": residuals[
            "physical_rotated_orbitals_direct_vs_qiskit"
        ],
        "hermiticity_residual": residuals[
            "qubit_hamiltonian_hermiticity"
        ],
        "qubits": dimensions["qubits"],
        "pauli_terms": dimensions["pauli_terms"],
        "full24_compiled_cx": resources["full24"]["compiled_cx"],
        "compact10_compiled_cx": resources["transferred_compact10"][
            "compiled_cx"
        ],
    }


def operator_pool_rows() -> list[dict[str, Any]]:
    full_pool, compact, _ = compact_excitations()
    compact_set = set(compact)
    rows: list[dict[str, Any]] = []
    for index, excitation in enumerate(full_pool):
        rows.append(
            {
                "global_uccsd_index": index,
                "operator_label": p7.operator_label(index),
                "operator_type": "single" if index < 8 else "double",
                "excitation_json": json.dumps(
                    diag.excitation_to_json(excitation)
                ),
                "included_in_transferred_compact10": excitation in compact_set,
                "rotated_orbital_order": "phase7a1_source_reference_order",
            }
        )
    return rows


def write_report(path: Path, final: dict[str, Any]) -> None:
    lines = [
        "# LiH Phase 7A.2 Rotated Hamiltonian Audit",
        "",
        f"Analysis status: **{final['analysis_status']}**",
        "",
        "No VQE or optimizer was used.",
        "",
        "## Geometry summary",
        "",
        "| R (Å) | Passed | Qiskit exact | PySCF 2e5o | Full non-core 2e10o | "
        "Truncation error | Classification |",
        "|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for item in final["geometry_audits"]:
        energies = item["energies_ha"]
        lines.append(
            f"| {item['bond_length_angstrom']:.3f} | "
            f"{item['audit_passed']} | "
            f"{energies['qiskit_rotated_exact_total']:.12f} | "
            f"{energies['pyscf_rotated_2e5o_casci_total_ha']:.12f} | "
            f"{energies['pyscf_full_noncore_2e10o_casci_total_ha']:.12f} | "
            f"{energies['rotated_2e5o_error_vs_full_noncore_ha']:.9f} | "
            f"{energies['active_space_truncation_classification']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen support",
            "",
            f"Rotated support fingerprint: "
            f"`{final['rotated_support_fingerprint']}`",
            "",
            "The Phase-7A.1 physical orbitals and compact10 operator support "
            "were frozen before exact target energies were evaluated.",
            "",
            "## Decision",
            "",
            (
                "The construction audit passed. A separately preregistered "
                "Phase-7B VQE comparison may now be designed."
                if final["audit_passed"]
                else "The construction audit failed. Do not run target VQE."
            ),
            "",
            "Active-space truncation versus 2e,10o is reported as scientific "
            "context and is not hidden inside the construction gate.",
            "",
            "## Claim boundary",
            "",
            final["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Phase-7A.1 rotated LiH/6-31G 2e,5o Hamiltonian "
            "through independent Qiskit and PySCF constructions. No VQE runs."
        )
    )
    parser.add_argument(
        "--phase7a-dir",
        type=Path,
        default=Path("lih_phase7a_audit"),
    )
    parser.add_argument(
        "--phase7a1-dir",
        type=Path,
        default=Path("lih_phase7a1_embedding_diagnostic"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lih_phase7a2_hamiltonian_audit"),
    )
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--rotation-reconstruction-tolerance",
        type=float,
        default=1e-10,
    )
    parser.add_argument("--seed-transpiler", type=int, default=20260804)
    parser.add_argument(
        "--skip-circuit-resources",
        action="store_true",
        help="Skip transpilation while retaining every Hamiltonian audit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_tolerance <= 0 or args.rotation_reconstruction_tolerance <= 0:
        raise ValueError("Audit tolerances must be positive.")
    diag.require_quantum_stack()
    bundle = load_inputs(args.phase7a_dir, args.phase7a1_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reconstructing and freezing Phase-7A.1 rotated orbitals...", flush=True)
    frozen, reconstruction_rows = reconstruct_frozen_geometries(
        bundle,
        args.rotation_reconstruction_tolerance,
    )
    frozen_payload = build_frozen_support_payload(bundle, frozen)
    frozen_path = args.output_dir / "phase7a2_frozen_rotated_support.json"
    write_or_validate_frozen_support(frozen_path, frozen_payload)
    write_csv(
        args.output_dir / "phase7a2_rotation_reconstruction.csv",
        reconstruction_rows,
    )
    print(
        "Rotated support frozen before exact-energy evaluation: "
        f"{frozen_payload['rotated_support_fingerprint']}",
        flush=True,
    )

    protocol_payload = {
        "analysis_phase": "Phase 7A.2 rotated Hamiltonian audit",
        "rotated_support_fingerprint": frozen_payload[
            "rotated_support_fingerprint"
        ],
        "phase7a1_diagnostic_protocol_fingerprint": (
            bundle.phase7a1_protocol["diagnostic_protocol_fingerprint"]
        ),
        "target_bond_lengths_angstrom": [
            frozen[key].bond_length for key in sorted(frozen, key=float)
        ],
        "qiskit_construction": (
            "Explicitly unfold symmetry-compressed MO two-electron integrals; "
            "apply the same restricted MO-to-MO rotation to alpha and beta "
            "spins with BasisTransformer; then select explicit indices "
            "[1,2,3,4,5] in ActiveSpaceTransformer"
        ),
        "independent_energy_crosscheck": "PySCF CASCI(2e,5o)",
        "full_noncore_benchmark": "PySCF CASCI(2e,10o)",
        "audit_tolerance": args.audit_tolerance,
        "rotation_reconstruction_tolerance": (
            args.rotation_reconstruction_tolerance
        ),
        "seed_transpiler": args.seed_transpiler,
        "circuit_resources_skipped": args.skip_circuit_resources,
        "active_space_truncation_is_context_not_construction_gate": True,
        "no_vqe_performed": True,
    }
    protocol_fingerprint = p7.fingerprint(protocol_payload)
    write_or_validate_protocol(
        args.output_dir / "phase7a2_preregistered_protocol.json",
        {**protocol_payload, "protocol_fingerprint": protocol_fingerprint},
    )

    print("Building independent Qiskit and PySCF Hamiltonian audits...", flush=True)
    geometry_audits: list[dict[str, Any]] = []
    for key in sorted(frozen, key=float):
        item = frozen[key]
        target = item.target
        system = build_qiskit_system(item, target, args.audit_tolerance)
        benchmarks = pyscf_casci_benchmarks(target, item)
        audit = audit_geometry(
            system,
            target,
            item,
            benchmarks,
            args.audit_tolerance,
            args.skip_circuit_resources,
            args.seed_transpiler,
        )
        geometry_audits.append(audit)

    audit_passed = bool(all(item["audit_passed"] for item in geometry_audits))
    truncation_errors = [
        item["energies_ha"]["rotated_2e5o_error_vs_full_noncore_ha"]
        for item in geometry_audits
    ]
    final = {
        "analysis_status": (
            "audit_passed_rotated_hamiltonian_frozen_no_vqe_performed"
            if audit_passed
            else "audit_failed_do_not_run_vqe"
        ),
        "audit_passed": audit_passed,
        "no_vqe_performed": True,
        "rotated_support_fingerprint": frozen_payload[
            "rotated_support_fingerprint"
        ],
        "protocol_fingerprint": protocol_fingerprint,
        "phase7a1_diagnostic_protocol_fingerprint": (
            bundle.phase7a1_protocol["diagnostic_protocol_fingerprint"]
        ),
        "source_basis": SOURCE_BASIS,
        "target_basis": TARGET_BASIS,
        "active_space": {
            "electrons": ACTIVE_ELECTRONS,
            "spatial_orbitals": ACTIVE_ORBITALS,
            "qubits": EXPECTED_QUBITS,
        },
        "geometry_audits": geometry_audits,
        "active_space_truncation_summary": {
            "maximum_error_vs_full_noncore_ha": float(max(truncation_errors)),
            "all_geometries_chemical_vs_full_noncore": bool(
                all(abs(value) <= CHEMICAL_ACCURACY_HA for value in truncation_errors)
            ),
            "all_geometries_acceptable_vs_full_noncore": bool(
                all(abs(value) <= ACCEPTABLE_ACCURACY_HA for value in truncation_errors)
            ),
            "warning": (
                "This comparison measures active-space truncation. It is not "
                "part of the internal Hamiltonian-construction pass/fail gate."
            ),
        },
        "software_versions": diag.dependency_versions(),
        "claim_boundary": (
            "This audit establishes only internal consistency of the frozen "
            "rotated LiH/6-31G 2e,5o Hamiltonians and compact10 operator "
            "identity. It does not establish VQE accuracy, equality to the "
            "full non-core 2e,10o Hamiltonian, transfer to expanded active "
            "spaces, other basis sets, other molecules, or noisy hardware."
        ),
    }
    write_json(args.output_dir / "phase7a2_hamiltonian_audit.json", final)
    write_csv(
        args.output_dir / "phase7a2_geometry_summary.csv",
        [geometry_csv_row(item) for item in geometry_audits],
    )
    write_csv(
        args.output_dir / "phase7a2_target_operator_pool.csv",
        operator_pool_rows(),
    )
    write_report(
        args.output_dir / "PHASE7A2_HAMILTONIAN_REPORT.md",
        final,
    )

    frame = pd.DataFrame([geometry_csv_row(item) for item in geometry_audits])
    print("\nPhase 7A.2 geometry summary:")
    print(
        frame[
            [
                "bond_length_angstrom",
                "audit_passed",
                "qiskit_rotated_exact_total_ha",
                "pyscf_rotated_2e5o_casci_total_ha",
                "pyscf_full_noncore_2e10o_casci_total_ha",
                "rotated_2e5o_error_vs_full_noncore_ha",
                "active_space_truncation_classification",
                "qiskit_exact_vs_pyscf_casci_ha",
            ]
        ].to_string(index=False)
    )
    print("\nPhase 7A.2 conclusions:")
    print(
        json.dumps(
            {
                "analysis_status": final["analysis_status"],
                "audit_passed": audit_passed,
                "no_vqe_performed": True,
                "rotated_support_fingerprint": final[
                    "rotated_support_fingerprint"
                ],
                "protocol_fingerprint": protocol_fingerprint,
                "all_geometries_chemical_vs_full_noncore": final[
                    "active_space_truncation_summary"
                ]["all_geometries_chemical_vs_full_noncore"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit_passed:
        print(
            "FATAL: Phase 7A.2 audit failed. Do not run target VQE.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
