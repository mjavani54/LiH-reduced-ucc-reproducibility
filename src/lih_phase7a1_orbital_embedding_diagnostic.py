#!/usr/bin/env python3
"""Phase 7A.1: diagnose a failed STO-3G -> 6-31G orbital-transfer audit.

This program performs RHF-orbital diagnostics only. It does not build a target
active-space Hamiltonian, evaluate an exact target energy, or run VQE. It
reproduces the Phase-7A canonical mapping, tracks the target 6-31G non-core
orbitals across geometry, exhaustively scans admissible canonical subspaces,
and tests an occupied/virtual-block-preserving maximum-overlap rotated 2e,5o
embedding.

Run it beside ``lih_phase7a_cross_basis_transfer.py`` and the Phase-3 support
modules used by that script.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    import lih_phase3_geometry_transfer as p3
    import lih_phase7a_cross_basis_transfer as p7
except ImportError as exc:
    raise SystemExit(
        "Place this script beside lih_phase7a_cross_basis_transfer.py, "
        "lih_phase3_geometry_transfer.py, lih_phase2b_exhaustive_support.py, "
        "and lih_reference_failure_diagnostics.py."
    ) from exc


SCHEMA_VERSION = 1
EXPECTED_SOURCE_BASIS = "sto3g"
EXPECTED_TARGET_BASIS = "6-31g"
EXPECTED_ACTIVE_ELECTRONS = 2
EXPECTED_SOURCE_ACTIVE_ORBITALS = 5
EXPECTED_TARGET_NONCORE_ORBITALS = 10
EXPECTED_AUDIT_STATUS = "audit_failed_do_not_run_vqe"


@dataclass
class RotatedEmbedding:
    bond_length: float
    weights_in_full_target_mos: np.ndarray
    active_ao_coefficients: np.ndarray
    cross_overlap: np.ndarray
    minimum_assigned_group_quality: float
    minimum_individual_diagonal_overlap: float
    virtual_gram_minimum_eigenvalue: float
    virtual_gram_condition_number: float
    orthonormality_residual: float
    reliable: bool


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
        raise FileNotFoundError(f"Required Phase-7A output not found: {path}")


def normalized_bonds(values: Iterable[float]) -> list[float]:
    result = sorted({round(float(value), 9) for value in values})
    if not result or any(value <= 0 for value in result):
        raise ValueError("Bond lengths must be positive and nonempty.")
    return result


def validate_phase7a_inputs(
    phase7a_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit_path = phase7a_dir / "phase7a_audit.json"
    support_path = phase7a_dir / "phase7a_frozen_support.json"
    protocol_path = phase7a_dir / "phase7a_preregistered_protocol.json"
    for path in (audit_path, support_path, protocol_path):
        require_file(path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    support = json.loads(support_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    if audit.get("analysis_status") != EXPECTED_AUDIT_STATUS:
        raise RuntimeError(
            "Phase 7A.1 requires a failed Phase-7A audit. It must not replace "
            "an already-passing audit."
        )
    if bool(audit.get("audit_passed", True)):
        raise RuntimeError("The supplied Phase-7A audit is not marked failed.")
    if not bool(audit.get("no_vqe_performed", False)):
        raise RuntimeError("The Phase-7A input does not certify that no VQE ran.")
    if audit.get("support_fingerprint") != support.get("support_fingerprint"):
        raise RuntimeError("Phase-7A audit and frozen-support fingerprints differ.")
    if audit.get("protocol_fingerprint") != protocol.get("protocol_fingerprint"):
        raise RuntimeError("Phase-7A audit and protocol fingerprints differ.")

    support_without_fingerprint = dict(support)
    supplied_fingerprint = support_without_fingerprint.pop(
        "support_fingerprint", None
    )
    if supplied_fingerprint != p7.fingerprint(support_without_fingerprint):
        raise RuntimeError("The Phase-7A frozen-support payload was modified.")

    if support.get("source_basis") != EXPECTED_SOURCE_BASIS:
        raise RuntimeError("The source basis is not the audited STO-3G basis.")
    if support.get("target_basis") != EXPECTED_TARGET_BASIS:
        raise RuntimeError("The target basis is not the audited 6-31G basis.")
    if int(support.get("active_electrons", -1)) != EXPECTED_ACTIVE_ELECTRONS:
        raise RuntimeError("The frozen support does not use two active electrons.")
    if (
        int(support.get("active_spatial_orbitals", -1))
        != EXPECTED_SOURCE_ACTIVE_ORBITALS
    ):
        raise RuntimeError("The frozen support does not use five active orbitals.")

    audit_bonds = normalized_bonds(audit.get("target_bond_lengths_angstrom", []))
    support_bonds = normalized_bonds(
        support.get("target_bond_lengths_angstrom", [])
    )
    if audit_bonds != support_bonds:
        raise RuntimeError("Phase-7A audit and support geometry grids differ.")
    if len(audit.get("geometry_audits", [])) != len(audit_bonds):
        raise RuntimeError("Phase-7A geometry-audit coverage is incomplete.")
    return audit, support, protocol


def write_or_validate_protocol(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("diagnostic_protocol_fingerprint") != value.get(
            "diagnostic_protocol_fingerprint"
        ):
            raise RuntimeError(
                f"Existing diagnostic protocol {path} differs. Use a new "
                "output directory rather than overwriting it."
            )
        return
    write_json(path, value)


def build_protocol(
    audit: dict[str, Any],
    support: dict[str, Any],
    reproduction_tolerance: float,
) -> dict[str, Any]:
    threshold = float(support["orbital_overlap_threshold"])
    value = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": "Phase 7A.1 orbital-embedding diagnostic",
        "phase7a_protocol_fingerprint": audit["protocol_fingerprint"],
        "phase7a_support_fingerprint": audit["support_fingerprint"],
        "source_basis": EXPECTED_SOURCE_BASIS,
        "target_basis": EXPECTED_TARGET_BASIS,
        "target_bond_lengths_angstrom": support[
            "target_bond_lengths_angstrom"
        ],
        "tracking_bond_lengths_angstrom": support[
            "source_tracking_bond_lengths_angstrom"
        ],
        "frozen_orbital_overlap_threshold": threshold,
        "frozen_degeneracy_tolerance_ha": float(
            support["degeneracy_tolerance_ha"]
        ),
        "reproduction_tolerance": reproduction_tolerance,
        "canonical_subspace_scan": {
            "sizes": list(
                range(
                    EXPECTED_SOURCE_ACTIVE_ORBITALS,
                    EXPECTED_TARGET_NONCORE_ORBITALS + 1,
                )
            ),
            "occupied_constraint": (
                "include the sole non-core doubly occupied canonical MO"
            ),
            "virtual_rule": (
                "exhaustively enumerate canonical virtual-MO subsets"
            ),
            "objective": (
                "lexicographically maximize minimum source-degenerate-group "
                "projector singular value, minimum individual projector norm, "
                "then mean projection weight"
            ),
        },
        "rotated_embedding_rule": (
            "Project the tracked source occupied orbital into the target "
            "non-core occupied block and the four tracked source virtual "
            "orbitals into the target virtual block; symmetric-orthonormalize "
            "only within the virtual block."
        ),
        "leakage_controls": {
            "no_threshold_change": True,
            "no_target_active_hamiltonian": True,
            "no_exact_target_energy": True,
            "no_vqe": True,
            "no_optimizer": True,
        },
        "decision_order": [
            "reproduce_failed_phase7a_mapping",
            "test_current_selected_canonical_subspace",
            "test_best_admissible_canonical_five_orbital_subspace",
            "test_geometry_tracked_canonical_support",
            "test_rotated_maximum_overlap_two_electron_five_orbital_embedding",
            "otherwise_measure_minimum_canonical_subspace_size",
        ],
    }
    value["diagnostic_protocol_fingerprint"] = p7.fingerprint(value)
    return value


def build_snapshots(
    bonds: Sequence[float],
    basis: str,
    active_orbitals: int,
    overlap_threshold: float,
    degeneracy_tolerance: float,
) -> tuple[
    dict[str, p3.OrbitalSnapshot],
    list[tuple[int, ...]],
    pd.DataFrame,
    pd.DataFrame,
]:
    snapshots = {
        geometry_key(bond): p3.build_orbital_snapshot(
            bond,
            basis,
            EXPECTED_ACTIVE_ELECTRONS,
            active_orbitals,
        )
        for bond in bonds
    }
    groups, detail, summary = p3.track_orbitals(
        snapshots,
        bonds,
        overlap_threshold,
        degeneracy_tolerance,
    )
    return snapshots, groups, detail, summary


def cross_overlap_matrix(
    source: p3.OrbitalSnapshot,
    target: p7.TargetRHFData,
) -> np.ndarray:
    from pyscf import gto

    if source.reference_to_canonical is None:
        raise RuntimeError("The source orbitals were not geometry tracked.")
    source_tracked = source.active_coefficients[
        :, np.asarray(source.reference_to_canonical, dtype=int)
    ]
    overlap = gto.intor_cross("int1e_ovlp", source.molecule, target.molecule)
    return (
        source_tracked.T
        @ overlap
        @ np.asarray(target.mean_field.mo_coeff, dtype=float)
    )


def group_capture_metrics(
    cross_overlap: np.ndarray,
    groups: Sequence[tuple[int, ...]],
    target_columns: Sequence[int],
) -> dict[str, Any]:
    columns = np.asarray(target_columns, dtype=int)
    qualities: list[float] = []
    group_records: list[dict[str, Any]] = []
    for group_id, group in enumerate(groups):
        rows = np.asarray(group, dtype=int)
        singular_values = np.linalg.svd(
            cross_overlap[np.ix_(rows, columns)],
            compute_uv=False,
        )
        quality = float(np.min(singular_values))
        qualities.append(quality)
        group_records.append(
            {
                "group_id": group_id,
                "reference_orbitals": list(group),
                "singular_values": singular_values.tolist(),
                "minimum_singular_value": quality,
            }
        )
    row_projection_norms = np.linalg.norm(
        cross_overlap[:, columns], axis=1
    )
    return {
        "minimum_group_projector_singular_value": float(min(qualities)),
        "minimum_individual_projector_norm": float(
            np.min(row_projection_norms)
        ),
        "mean_projection_weight": float(
            np.mean(row_projection_norms**2)
        ),
        "individual_projector_norms": row_projection_norms.tolist(),
        "group_records": group_records,
    }


def assigned_metrics(
    cross_overlap: np.ndarray,
    groups: Sequence[tuple[int, ...]],
    reference_to_target: Sequence[int],
) -> dict[str, Any]:
    assigned = np.asarray(reference_to_target, dtype=int)
    diagonal = np.abs(
        cross_overlap[
            np.arange(EXPECTED_SOURCE_ACTIVE_ORBITALS),
            assigned,
        ]
    )
    qualities: list[float] = []
    for group in groups:
        rows = np.asarray(group, dtype=int)
        block = cross_overlap[np.ix_(rows, assigned[rows])]
        qualities.append(float(np.min(np.linalg.svd(block, compute_uv=False))))
    return {
        "minimum_assigned_group_singular_value": float(min(qualities)),
        "minimum_individual_assigned_overlap": float(np.min(diagonal)),
        "individual_assigned_overlaps": diagonal.tolist(),
    }


def assignment_within_five_orbital_subset(
    cross_overlap: np.ndarray,
    occupations: np.ndarray,
    subset: Sequence[int],
) -> list[int]:
    selected = [int(value) for value in subset]
    occupied = [
        index for index in selected if np.isclose(occupations[index], 2.0)
    ]
    virtuals = [
        index for index in selected if np.isclose(occupations[index], 0.0)
    ]
    if len(occupied) != 1 or len(virtuals) != 4:
        raise RuntimeError("A canonical five-orbital subset has wrong occupancy.")
    result = np.full(EXPECTED_SOURCE_ACTIVE_ORBITALS, -1, dtype=int)
    result[0] = occupied[0]
    rows, columns = linear_sum_assignment(
        -np.abs(cross_overlap[np.ix_(range(1, 5), virtuals)])
    )
    for row, column in zip(rows, columns):
        result[int(row) + 1] = virtuals[int(column)]
    if np.any(result < 0):
        raise RuntimeError("Canonical subset assignment is incomplete.")
    return result.tolist()


def scan_canonical_subspaces(
    bond_length: float,
    cross_overlap: np.ndarray,
    target: p7.TargetRHFData,
    groups: Sequence[tuple[int, ...]],
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    occupations = np.asarray(target.mean_field.mo_occ, dtype=float)
    noncore = [
        index
        for index in range(target.num_frozen_core_orbitals, len(occupations))
    ]
    occupied = [
        index for index in noncore if np.isclose(occupations[index], 2.0)
    ]
    virtuals = [
        index for index in noncore if np.isclose(occupations[index], 0.0)
    ]
    if len(noncore) != EXPECTED_TARGET_NONCORE_ORBITALS:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGET_NONCORE_ORBITALS} target non-core "
            f"orbitals; found {len(noncore)}."
        )
    if len(occupied) != 1:
        raise RuntimeError("Expected one non-core doubly occupied target MO.")

    rows: list[dict[str, Any]] = []
    best_by_size: dict[int, dict[str, Any]] = {}
    for size in range(
        EXPECTED_SOURCE_ACTIVE_ORBITALS,
        EXPECTED_TARGET_NONCORE_ORBITALS + 1,
    ):
        best: dict[str, Any] | None = None
        best_objective: tuple[float, float, float, tuple[int, ...]] | None = None
        combinations_evaluated = 0
        for virtual_subset in itertools.combinations(virtuals, size - 1):
            subset = tuple(sorted((occupied[0], *virtual_subset)))
            metrics = group_capture_metrics(cross_overlap, groups, subset)
            objective = (
                metrics["minimum_group_projector_singular_value"],
                metrics["minimum_individual_projector_norm"],
                metrics["mean_projection_weight"],
                tuple(-index for index in subset),
            )
            combinations_evaluated += 1
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best = {"subset": list(subset), **metrics}
        if best is None:
            raise RuntimeError(f"No admissible target subspace of size {size}.")
        best["passes_frozen_threshold"] = bool(
            best["minimum_group_projector_singular_value"] >= threshold
        )
        best_by_size[size] = best
        rows.append(
            {
                "bond_length_angstrom": bond_length,
                "geometry_key": geometry_key(bond_length),
                "canonical_subspace_size": size,
                "best_target_full_mo_indices_json": json.dumps(best["subset"]),
                "minimum_group_projector_singular_value": best[
                    "minimum_group_projector_singular_value"
                ],
                "minimum_individual_projector_norm": best[
                    "minimum_individual_projector_norm"
                ],
                "mean_projection_weight": best["mean_projection_weight"],
                "passes_frozen_threshold": best["passes_frozen_threshold"],
                "num_admissible_subsets_evaluated": combinations_evaluated,
            }
        )
    return rows, best_by_size


def symmetric_inverse_square_root(
    matrix: np.ndarray,
    eigenvalue_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(np.min(eigenvalues)) <= eigenvalue_floor:
        raise RuntimeError(
            "Projected source virtual orbitals are linearly dependent in the "
            "target virtual space."
        )
    inverse_root = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )
    return inverse_root, eigenvalues


def build_rotated_embedding(
    bond_length: float,
    cross_overlap: np.ndarray,
    target: p7.TargetRHFData,
    groups: Sequence[tuple[int, ...]],
    threshold: float,
) -> RotatedEmbedding:
    coefficients = np.asarray(target.mean_field.mo_coeff, dtype=float)
    occupations = np.asarray(target.mean_field.mo_occ, dtype=float)
    noncore = list(
        range(target.num_frozen_core_orbitals, coefficients.shape[1])
    )
    occupied = [
        index for index in noncore if np.isclose(occupations[index], 2.0)
    ]
    virtuals = [
        index for index in noncore if np.isclose(occupations[index], 0.0)
    ]
    if len(occupied) != 1 or len(virtuals) != 9:
        raise RuntimeError("Unexpected 6-31G non-core occupation structure.")

    weights = np.zeros(
        (coefficients.shape[1], EXPECTED_SOURCE_ACTIVE_ORBITALS),
        dtype=float,
    )
    weights[occupied[0], 0] = 1.0
    projected_virtuals = cross_overlap[1:, virtuals].T
    virtual_gram = projected_virtuals.T @ projected_virtuals
    inverse_root, eigenvalues = symmetric_inverse_square_root(virtual_gram)
    virtual_weights = projected_virtuals @ inverse_root
    weights[np.ix_(virtuals, range(1, 5))] = virtual_weights

    rotated_overlap = cross_overlap @ weights
    for index in range(EXPECTED_SOURCE_ACTIVE_ORBITALS):
        if rotated_overlap[index, index] < 0:
            weights[:, index] *= -1.0
    rotated_overlap = cross_overlap @ weights
    assigned = assigned_metrics(
        rotated_overlap,
        groups,
        list(range(EXPECTED_SOURCE_ACTIVE_ORBITALS)),
    )
    orthonormality_residual = float(
        np.max(np.abs(weights.T @ weights - np.eye(5)))
    )
    active_coefficients = coefficients @ weights
    quality = float(assigned["minimum_assigned_group_singular_value"])
    minimum_diagonal = float(assigned["minimum_individual_assigned_overlap"])
    minimum_eigenvalue = float(np.min(eigenvalues))
    condition_number = float(np.max(eigenvalues) / minimum_eigenvalue)
    reliable = bool(
        quality >= threshold
        and orthonormality_residual <= 1e-8
        and minimum_eigenvalue > 1e-12
    )
    return RotatedEmbedding(
        bond_length=bond_length,
        weights_in_full_target_mos=weights,
        active_ao_coefficients=active_coefficients,
        cross_overlap=rotated_overlap,
        minimum_assigned_group_quality=quality,
        minimum_individual_diagonal_overlap=minimum_diagonal,
        virtual_gram_minimum_eigenvalue=minimum_eigenvalue,
        virtual_gram_condition_number=condition_number,
        orthonormality_residual=orthonormality_residual,
        reliable=reliable,
    )


def propagated_target_assignment(
    reference_assignment_full_mo: Sequence[int],
    reference_target: p7.TargetRHFData,
    current_target: p7.TargetRHFData,
    current_snapshot: p3.OrbitalSnapshot,
) -> list[int]:
    if current_snapshot.reference_to_canonical is None:
        raise RuntimeError("Target-basis geometry tracking is missing.")
    if (
        reference_target.num_frozen_core_orbitals
        != current_target.num_frozen_core_orbitals
    ):
        raise RuntimeError("Frozen-core count changed across target geometries.")
    frozen = current_target.num_frozen_core_orbitals
    mapping = np.asarray(current_snapshot.reference_to_canonical, dtype=int)
    result: list[int] = []
    for full_index in reference_assignment_full_mo:
        reference_noncore = int(full_index) - frozen
        if reference_noncore < 0 or reference_noncore >= len(mapping):
            raise RuntimeError("Tracked target assignment includes a core MO.")
        result.append(frozen + int(mapping[reference_noncore]))
    return result


def rotated_continuity_rows(
    bonds: Sequence[float],
    targets: dict[str, p7.TargetRHFData],
    embeddings: dict[str, RotatedEmbedding],
    groups: Sequence[tuple[int, ...]],
    threshold: float,
) -> list[dict[str, Any]]:
    from pyscf import gto

    rows: list[dict[str, Any]] = []
    for previous_bond, current_bond in zip(bonds[:-1], bonds[1:]):
        previous_key = geometry_key(previous_bond)
        current_key = geometry_key(current_bond)
        overlap = gto.intor_cross(
            "int1e_ovlp",
            targets[previous_key].molecule,
            targets[current_key].molecule,
        )
        matrix = (
            embeddings[previous_key].active_ao_coefficients.T
            @ overlap
            @ embeddings[current_key].active_ao_coefficients
        )
        metrics = assigned_metrics(matrix, groups, list(range(5)))
        quality = metrics["minimum_assigned_group_singular_value"]
        rows.append(
            {
                "previous_bond_length_angstrom": previous_bond,
                "current_bond_length_angstrom": current_bond,
                "minimum_assigned_group_singular_value": quality,
                "minimum_individual_diagonal_overlap": metrics[
                    "minimum_individual_assigned_overlap"
                ],
                "passes_frozen_threshold": bool(quality >= threshold),
            }
        )
    return rows


def classify_geometry(
    threshold: float,
    original_assigned: float,
    current_subspace: float,
    best5_assigned: float,
    best5_subspace: float,
    rotated_quality: float,
) -> str:
    if original_assigned >= threshold:
        return "original_mapping_unexpectedly_passes"
    if current_subspace >= threshold:
        return "current_canonical5_requires_internal_rotation"
    if best5_assigned >= threshold:
        return "alternate_canonical5_assignment_repairs_failure"
    if best5_subspace >= threshold:
        return "alternate_canonical5_requires_internal_rotation"
    if rotated_quality >= threshold:
        return "canonical5_selection_insufficient_rotated_2e5o_feasible"
    return "strict_rotated_2e5o_embedding_fails"


def write_report(path: Path, conclusions: dict[str, Any]) -> None:
    lines = [
        "# LiH Phase 7A.1 Orbital-Embedding Diagnostic",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        "No target active Hamiltonian, exact target energy, VQE, or optimizer "
        "was used.",
        "",
        "## Geometry diagnosis",
        "",
        "| R (Å) | Original | Current subspace | Best canonical5 | Rotated5 | "
        "Minimum k | Diagnosis |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in conclusions["geometry_results"]:
        lines.append(
            f"| {item['bond_length_angstrom']:.3f} | "
            f"{item['original_assigned_group_quality']:.6f} | "
            f"{item['current_selected_subspace_quality']:.6f} | "
            f"{item['best_canonical5_subspace_quality']:.6f} | "
            f"{item['rotated_2e5o_quality']:.6f} | "
            f"{item['minimum_canonical_subspace_size_meeting_threshold']} | "
            f"{item['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Recommended next gate: "
            f"**{conclusions['recommended_next_gate']}**",
            "",
            conclusions["decision_explanation"],
            "",
            "The original 0.75 overlap threshold was retained unchanged.",
            "",
            "## Claim boundary",
            "",
            conclusions["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the failed Phase-7A STO-3G -> 6-31G orbital mapping "
            "without building a target active Hamiltonian or running VQE."
        )
    )
    parser.add_argument(
        "--phase7a-dir",
        type=Path,
        default=Path("lih_phase7a_audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lih_phase7a1_embedding_diagnostic"),
    )
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Numerical tolerance for reproducing the prior RHF overlap "
            "diagnostic; this does not change the frozen scientific threshold."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.reproduction_tolerance <= 0:
        raise ValueError("Reproduction tolerance must be positive.")

    audit, support, _ = validate_phase7a_inputs(args.phase7a_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_protocol = build_protocol(
        audit,
        support,
        args.reproduction_tolerance,
    )
    write_or_validate_protocol(
        args.output_dir / "phase7a1_protocol.json",
        diagnostic_protocol,
    )

    threshold = float(support["orbital_overlap_threshold"])
    degeneracy_tolerance = float(support["degeneracy_tolerance_ha"])
    tracking_bonds = normalized_bonds(
        support["source_tracking_bond_lengths_angstrom"]
    )
    audit_bonds = normalized_bonds(support["target_bond_lengths_angstrom"])
    audit_by_key = {
        str(item["geometry_key"]): item for item in audit["geometry_audits"]
    }

    print("Tracking source STO-3G active orbitals...", flush=True)
    source_snapshots, source_groups, source_tracking, source_summary = (
        build_snapshots(
            tracking_bonds,
            EXPECTED_SOURCE_BASIS,
            EXPECTED_SOURCE_ACTIVE_ORBITALS,
            threshold,
            degeneracy_tolerance,
        )
    )
    source_tracking.to_csv(
        args.output_dir / "phase7a1_source_orbital_tracking.csv",
        index=False,
    )
    source_summary.to_csv(
        args.output_dir / "phase7a1_source_orbital_tracking_summary.csv",
        index=False,
    )

    print("Tracking all target 6-31G non-core orbitals...", flush=True)
    target_snapshots, _target_groups, target_tracking, target_summary = (
        build_snapshots(
            tracking_bonds,
            EXPECTED_TARGET_BASIS,
            EXPECTED_TARGET_NONCORE_ORBITALS,
            threshold,
            degeneracy_tolerance,
        )
    )
    target_tracking.to_csv(
        args.output_dir / "phase7a1_target_orbital_tracking.csv",
        index=False,
    )
    target_summary.to_csv(
        args.output_dir / "phase7a1_target_orbital_tracking_summary.csv",
        index=False,
    )

    targets: dict[str, p7.TargetRHFData] = {}
    cross_overlaps: dict[str, np.ndarray] = {}
    independent_mappings: dict[str, p7.CrossBasisMapping] = {}
    rotations: dict[str, RotatedEmbedding] = {}
    scan_rows: list[dict[str, Any]] = []
    scan_best: dict[str, dict[int, dict[str, Any]]] = {}
    reproduction_rows: list[dict[str, Any]] = []

    print("Scanning canonical subspaces and rotated embeddings...", flush=True)
    for bond in tracking_bonds:
        key = geometry_key(bond)
        target = p7.build_target_rhf(bond, EXPECTED_TARGET_BASIS)
        matrix = cross_overlap_matrix(source_snapshots[key], target)
        mapping, _ = p7.cross_basis_mapping(
            source_snapshots[key],
            target,
            source_groups,
            threshold,
        )
        rows, best = scan_canonical_subspaces(
            bond,
            matrix,
            target,
            source_groups,
            threshold,
        )
        rotation = build_rotated_embedding(
            bond,
            matrix,
            target,
            source_groups,
            threshold,
        )
        targets[key] = target
        cross_overlaps[key] = matrix
        independent_mappings[key] = mapping
        rotations[key] = rotation
        scan_rows.extend(rows)
        scan_best[key] = best

        if key in audit_by_key:
            prior = audit_by_key[key]
            prior_selected = [
                int(value) for value in prior["selected_target_full_mo_indices"]
            ]
            prior_quality = float(
                prior["cross_basis_overlap"]["minimum_group_singular_value"]
            )
            reproduced_quality = float(
                min(mapping.group_quality_by_reference_orbital)
            )
            selected_match = (
                prior_selected == mapping.selected_target_full_mo_indices
            )
            quality_residual = abs(reproduced_quality - prior_quality)
            reproduction_rows.append(
                {
                    "bond_length_angstrom": bond,
                    "geometry_key": key,
                    "prior_selected_target_full_mo_indices_json": json.dumps(
                        prior_selected
                    ),
                    "reproduced_selected_target_full_mo_indices_json": json.dumps(
                        mapping.selected_target_full_mo_indices
                    ),
                    "selected_indices_match": selected_match,
                    "prior_minimum_assigned_group_quality": prior_quality,
                    "reproduced_minimum_assigned_group_quality": (
                        reproduced_quality
                    ),
                    "absolute_quality_residual": quality_residual,
                    "reproduction_passed": bool(
                        selected_match
                        and quality_residual <= args.reproduction_tolerance
                    ),
                }
            )

    write_csv(
        args.output_dir / "phase7a1_reproduction_audit.csv",
        reproduction_rows,
    )
    write_csv(
        args.output_dir / "phase7a1_canonical_subspace_scan.csv",
        scan_rows,
    )

    reference_key = geometry_key(p7.REFERENCE_BOND_LENGTH)
    reference_target = targets[reference_key]
    original_reference_assignment = independent_mappings[
        reference_key
    ].reference_to_target_full_mo
    reference_best5_subset = scan_best[reference_key][5]["subset"]
    best5_reference_assignment = assignment_within_five_orbital_subset(
        cross_overlaps[reference_key],
        np.asarray(reference_target.mean_field.mo_occ, dtype=float),
        reference_best5_subset,
    )

    continuity_rows = rotated_continuity_rows(
        tracking_bonds,
        targets,
        rotations,
        source_groups,
        threshold,
    )
    write_csv(
        args.output_dir / "phase7a1_rotated_orbital_continuity.csv",
        continuity_rows,
    )

    rotation_rows: list[dict[str, Any]] = []
    per_orbital_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    geometry_results: list[dict[str, Any]] = []
    for bond in audit_bonds:
        key = geometry_key(bond)
        target = targets[key]
        matrix = cross_overlaps[key]
        mapping = independent_mappings[key]
        rotation = rotations[key]
        current_selected = mapping.selected_target_full_mo_indices
        current_subspace = group_capture_metrics(
            matrix,
            source_groups,
            current_selected,
        )
        original_assigned = assigned_metrics(
            matrix,
            source_groups,
            mapping.reference_to_target_full_mo,
        )
        best5 = scan_best[key][5]
        best5_assignment = assignment_within_five_orbital_subset(
            matrix,
            np.asarray(target.mean_field.mo_occ, dtype=float),
            best5["subset"],
        )
        best5_assigned = assigned_metrics(
            matrix,
            source_groups,
            best5_assignment,
        )
        tracked_original_assignment = propagated_target_assignment(
            original_reference_assignment,
            reference_target,
            target,
            target_snapshots[key],
        )
        tracked_original = assigned_metrics(
            matrix,
            source_groups,
            tracked_original_assignment,
        )
        tracked_best5_assignment = propagated_target_assignment(
            best5_reference_assignment,
            reference_target,
            target,
            target_snapshots[key],
        )
        tracked_best5 = assigned_metrics(
            matrix,
            source_groups,
            tracked_best5_assignment,
        )
        minimum_size = next(
            (
                size
                for size in range(5, EXPECTED_TARGET_NONCORE_ORBITALS + 1)
                if scan_best[key][size]["passes_frozen_threshold"]
            ),
            None,
        )
        original_quality = original_assigned[
            "minimum_assigned_group_singular_value"
        ]
        current_subspace_quality = current_subspace[
            "minimum_group_projector_singular_value"
        ]
        best5_quality = best5["minimum_group_projector_singular_value"]
        best5_assigned_quality = best5_assigned[
            "minimum_assigned_group_singular_value"
        ]
        diagnosis = classify_geometry(
            threshold,
            original_quality,
            current_subspace_quality,
            best5_assigned_quality,
            best5_quality,
            rotation.minimum_assigned_group_quality,
        )
        result = {
            "bond_length_angstrom": bond,
            "geometry_key": key,
            "original_selected_target_full_mo_indices": current_selected,
            "original_assigned_group_quality": original_quality,
            "current_selected_subspace_quality": current_subspace_quality,
            "best_canonical5_target_full_mo_indices": best5["subset"],
            "best_canonical5_reference_assignment": best5_assignment,
            "best_canonical5_assigned_quality": best5_assigned_quality,
            "best_canonical5_subspace_quality": best5_quality,
            "tracked_original_assignment": tracked_original_assignment,
            "tracked_original_assigned_quality": tracked_original[
                "minimum_assigned_group_singular_value"
            ],
            "tracked_best5_assignment": tracked_best5_assignment,
            "tracked_best5_assigned_quality": tracked_best5[
                "minimum_assigned_group_singular_value"
            ],
            "rotated_2e5o_quality": rotation.minimum_assigned_group_quality,
            "rotated_2e5o_minimum_diagonal_overlap": (
                rotation.minimum_individual_diagonal_overlap
            ),
            "rotated_2e5o_virtual_gram_condition_number": (
                rotation.virtual_gram_condition_number
            ),
            "rotated_2e5o_orthonormality_residual": (
                rotation.orthonormality_residual
            ),
            "rotated_2e5o_reliable": rotation.reliable,
            "minimum_canonical_subspace_size_meeting_threshold": minimum_size,
            "diagnosis": diagnosis,
        }
        geometry_results.append(result)
        geometry_rows.append(
            {
                **result,
                "original_selected_target_full_mo_indices": json.dumps(
                    current_selected
                ),
                "best_canonical5_target_full_mo_indices": json.dumps(
                    best5["subset"]
                ),
                "best_canonical5_reference_assignment": json.dumps(
                    best5_assignment
                ),
                "tracked_original_assignment": json.dumps(
                    tracked_original_assignment
                ),
                "tracked_best5_assignment": json.dumps(
                    tracked_best5_assignment
                ),
            }
        )

        for reference_orbital in range(5):
            per_orbital_rows.append(
                {
                    "bond_length_angstrom": bond,
                    "geometry_key": key,
                    "reference_source_active_orbital": reference_orbital,
                    "original_assigned_target_full_mo": (
                        mapping.reference_to_target_full_mo[reference_orbital]
                    ),
                    "original_assigned_absolute_overlap": abs(
                        matrix[
                            reference_orbital,
                            mapping.reference_to_target_full_mo[
                                reference_orbital
                            ],
                        ]
                    ),
                    "current_selected_subspace_projector_norm": (
                        current_subspace["individual_projector_norms"][
                            reference_orbital
                        ]
                    ),
                    "best_canonical5_subspace_projector_norm": best5[
                        "individual_projector_norms"
                    ][reference_orbital],
                    "all_noncore_subspace_projector_norm": scan_best[key][10][
                        "individual_projector_norms"
                    ][reference_orbital],
                    "rotated_2e5o_absolute_diagonal_overlap": abs(
                        rotation.cross_overlap[
                            reference_orbital, reference_orbital
                        ]
                    ),
                }
            )

        energies = np.asarray(target.mean_field.mo_energy, dtype=float)
        occupations = np.asarray(target.mean_field.mo_occ, dtype=float)
        for full_mo in range(rotation.weights_in_full_target_mos.shape[0]):
            for rotated_orbital in range(5):
                rotation_rows.append(
                    {
                        "bond_length_angstrom": bond,
                        "geometry_key": key,
                        "target_full_mo_index": full_mo,
                        "target_mo_energy_ha": float(energies[full_mo]),
                        "target_mo_occupation": float(occupations[full_mo]),
                        "rotated_active_orbital": rotated_orbital,
                        "rotation_weight": float(
                            rotation.weights_in_full_target_mos[
                                full_mo, rotated_orbital
                            ]
                        ),
                    }
                )

    write_csv(
        args.output_dir / "phase7a1_geometry_summary.csv", geometry_rows
    )
    write_csv(
        args.output_dir / "phase7a1_per_orbital_overlap.csv",
        per_orbital_rows,
    )
    write_csv(
        args.output_dir / "phase7a1_rotated_orbital_coefficients.csv",
        rotation_rows,
    )

    reproduction_passed = bool(
        all(row["reproduction_passed"] for row in reproduction_rows)
    )
    rotated_all_reliable = bool(
        all(item["rotated_2e5o_reliable"] for item in geometry_results)
    )
    continuity_all_reliable = bool(
        all(row["passes_frozen_threshold"] for row in continuity_rows)
    )
    best5_all_pass = bool(
        all(
            item["best_canonical5_subspace_quality"] >= threshold
            for item in geometry_results
        )
    )
    best5_assigned_all_pass = bool(
        all(
            item["best_canonical5_assigned_quality"] >= threshold
            for item in geometry_results
        )
    )
    tracked_best5_all_pass = bool(
        all(
            item["tracked_best5_assigned_quality"] >= threshold
            for item in geometry_results
        )
    )
    source_tracking_all_reliable = bool(
        source_summary["tracking_reliable"].all()
    )
    target_tracking_all_reliable = bool(
        target_summary["tracking_reliable"].all()
    )

    if not reproduction_passed:
        recommended_next_gate = "stop_and_resolve_phase7a_nonreproduction"
        explanation = (
            "The failed Phase-7A mapping could not be reproduced within the "
            "declared numerical tolerance. No replacement protocol is valid "
            "until this discrepancy is resolved."
        )
    elif not source_tracking_all_reliable:
        recommended_next_gate = "stop_and_resolve_source_orbital_tracking"
        explanation = (
            "The STO-3G source-orbital identities are not reliable across the "
            "declared geometry path. Neither a canonical nor a rotated target "
            "repair is interpretable until source tracking is resolved."
        )
    elif best5_assigned_all_pass and tracked_best5_all_pass:
        recommended_next_gate = (
            "preregister_and_audit_overlap_optimized_canonical_2e5o"
        )
        explanation = (
            "A different five-canonical-orbital selection passes the frozen "
            "threshold and remains reliable under target-basis geometry "
            "tracking. It must be refrozen before any target energy is used."
        )
    elif best5_all_pass:
        recommended_next_gate = (
            "preregister_and_audit_rotated_best_canonical5_subspace"
        )
        explanation = (
            "Five canonical orbitals span an adequate subspace, but a "
            "one-to-one canonical assignment does not. The next gate must "
            "construct and audit an internal active-space rotation."
        )
    elif rotated_all_reliable and continuity_all_reliable:
        recommended_next_gate = (
            "build_phase7a2_rotated_matched_2e5o_hamiltonian_audit"
        )
        explanation = (
            "No canonical five-orbital subset is adequate at every geometry, "
            "but the block-preserving maximum-overlap rotated 2e,5o embedding "
            "passes and is continuous. The next step is a separate Hamiltonian "
            "audit using the frozen rotation coefficients; still no VQE."
        )
    else:
        recommended_next_gate = (
            "abandon_strict_2e5o_transfer_and_preregister_expanded_active_space"
        )
        explanation = (
            "Even the rotated five-orbital embedding is not reliable across "
            "the grid. A larger target active space is required, which changes "
            "the scientific question and must be preregistered as a new phase."
        )

    resolved_minimum_sizes = [
        item["minimum_canonical_subspace_size_meeting_threshold"]
        for item in geometry_results
        if item["minimum_canonical_subspace_size_meeting_threshold"] is not None
    ]
    conclusions = {
        "analysis_status": (
            "complete_no_hamiltonian_no_exact_energy_no_vqe"
            if reproduction_passed
            else "failed_phase7a_reproduction"
        ),
        "diagnostic_protocol_fingerprint": diagnostic_protocol[
            "diagnostic_protocol_fingerprint"
        ],
        "phase7a_protocol_fingerprint": audit["protocol_fingerprint"],
        "phase7a_support_fingerprint": audit["support_fingerprint"],
        "frozen_orbital_overlap_threshold": threshold,
        "threshold_was_not_changed": True,
        "phase7a_failure_reproduced": reproduction_passed,
        "source_tracking_all_reliable": source_tracking_all_reliable,
        "target_noncore_tracking_all_reliable": target_tracking_all_reliable,
        "rotated_embedding_all_audit_geometries_reliable": (
            rotated_all_reliable
        ),
        "rotated_embedding_all_geometry_transitions_reliable": (
            continuity_all_reliable
        ),
        "best_canonical5_subspace_all_geometries_pass": best5_all_pass,
        "best_canonical5_assignment_all_geometries_pass": (
            best5_assigned_all_pass
        ),
        "tracked_best_canonical5_all_geometries_pass": (
            tracked_best5_all_pass
        ),
        "maximum_minimum_canonical_subspace_size_required": (
            max(resolved_minimum_sizes) if resolved_minimum_sizes else None
        ),
        "geometry_results": geometry_results,
        "recommended_next_gate": recommended_next_gate,
        "decision_explanation": explanation,
        "no_target_active_hamiltonian_built": True,
        "no_exact_target_energy_evaluated": True,
        "no_vqe_performed": True,
        "claim_boundary": (
            "This diagnostic distinguishes orbital assignment, canonical "
            "subspace selection, geometry tracking, and rotated active-space "
            "embedding only for frozen-core LiH STO-3G -> 6-31G over the "
            "declared geometry path. Passing a rotated embedding does not "
            "establish Hamiltonian consistency, VQE accuracy, transfer to an "
            "expanded active space, or transfer to another molecule."
        ),
    }
    write_json(
        args.output_dir / "phase7a1_embedding_conclusions.json",
        conclusions,
    )
    write_report(
        args.output_dir / "PHASE7A1_EMBEDDING_REPORT.md",
        conclusions,
    )

    summary_frame = pd.DataFrame(geometry_rows)
    print("\nPhase 7A.1 geometry summary:")
    print(
        summary_frame[
            [
                "bond_length_angstrom",
                "original_assigned_group_quality",
                "current_selected_subspace_quality",
                "best_canonical5_subspace_quality",
                "rotated_2e5o_quality",
                "minimum_canonical_subspace_size_meeting_threshold",
                "diagnosis",
            ]
        ].to_string(index=False)
    )
    print("\nPhase 7A.1 conclusions:")
    print(
        json.dumps(
            {
                "analysis_status": conclusions["analysis_status"],
                "phase7a_failure_reproduced": reproduction_passed,
                "recommended_next_gate": recommended_next_gate,
                "no_target_active_hamiltonian_built": True,
                "no_exact_target_energy_evaluated": True,
                "no_vqe_performed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not reproduction_passed:
        print(
            "FATAL: Phase-7A mapping did not reproduce. Stop here.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
