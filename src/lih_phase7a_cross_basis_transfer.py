#!/usr/bin/env python3
"""Phase 7A audit: blind STO-3G -> 6-31G transfer in matched LiH active spaces.

This program is deliberately audit-only. It does not optimize a VQE ansatz.
It tracks the audited STO-3G active orbitals along the established LiH geometry
path, maps those orbitals into the full 6-31G RHF space by occupancy-constrained
maximum overlap, freezes a matched 2-electron/5-orbital support, and only then
builds and exactly audits the target active-space Hamiltonians.

The program targets Qiskit Nature 0.8 and the same Jordan-Wigner conventions as
Phases 1-6. Run it beside ``lih_reference_failure_diagnostics.py``,
``lih_phase2b_exhaustive_support.py``, and
``lih_phase3_geometry_transfer.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place the Phase-7A script beside lih_phase3_geometry_transfer.py, "
        "lih_phase2b_exhaustive_support.py, and "
        "lih_reference_failure_diagnostics.py."
    ) from exc


SOURCE_BASIS = "sto3g"
TARGET_BASIS = "6-31g"
REFERENCE_BOND_LENGTH = 1.595
TARGET_BOND_LENGTHS: tuple[float, ...] = (1.595, 2.5, 3.0)
SOURCE_TRACKING_BOND_LENGTHS: tuple[float, ...] = (1.595, 2.1, 2.5, 3.0)

ACTIVE_ELECTRONS = 2
ACTIVE_SPATIAL_ORBITALS = 5
EXPECTED_QUBITS = 10
EXPECTED_SINGLES = 8
EXPECTED_DOUBLES = 16
EXPECTED_FULL_UCCSD = 24
EXPECTED_REFERENCE_FINGERPRINT = (
    "9d3762a8e0f742ce0cebb8ca38ae5204e72ea66b031c0b04980399644b9a4e51"
)

ACTIVE_SINGLE_LOCAL_IDS: tuple[int, ...] = (0, 3, 4, 7)
ACTIVE_DOUBLE_LOCAL_IDS: tuple[int, ...] = (0, 3, 5, 10, 12, 15)
COMPACT_LABELS: tuple[str, ...] = (
    "S0",
    "S3",
    "S4",
    "S7",
    "D0",
    "D3",
    "D5",
    "D10",
    "D12",
    "D15",
)

STRICT_EQUIVALENCE_HA = 1e-6
CHEMICAL_ACCURACY_HA = 0.0016
GOOD_ACCURACY_HA = 0.005
ACCEPTABLE_ACCURACY_HA = 0.01


@dataclass
class TargetRHFData:
    bond_length: float
    molecule: Any
    mean_field: Any
    num_frozen_core_orbitals: int


@dataclass
class CrossBasisMapping:
    bond_length: float
    source_reference_to_source_canonical: list[int]
    reference_to_target_full_mo: list[int]
    reference_to_target_active_position: list[int]
    selected_target_full_mo_indices: list[int]
    signed_overlaps: list[float]
    absolute_overlaps: list[float]
    group_quality_by_reference_orbital: list[float]
    target_full_mo_occupations: list[float]
    source_tracking_reliable: bool
    occupation_pattern_ok: bool
    mapping_reliable: bool


@dataclass
class ActiveSystemData:
    bond_length: float
    active_orbitals: list[int]
    raw_problem: Any
    problem: Any
    mapper: Any
    qubit_op: Any
    constants: dict[str, float]
    total_offset_ha: float
    exact_active_electronic_ha: float
    exact_total_ha: float
    hf_active_electronic_ha: float
    hf_total_qubit_ha: float
    hf_total_driver_ha: float
    singles: list[diag.Excitation]
    doubles: list[diag.Excitation]
    full_pool: list[diag.Excitation]
    fingerprint: str
    driver_orbital_energy_max_abs_difference_ha: float | None
    selected_driver_gauge_min_abs_diagonal_overlap: float | None
    selected_driver_gauge_max_abs_offdiagonal_overlap: float | None


def geometry_key(bond_length: float) -> str:
    return f"{float(bond_length):.6f}"


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


def fingerprint(value: Any) -> str:
    raw = json.dumps(
        json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalized_values(values: Iterable[float]) -> list[float]:
    result = sorted({round(float(value), 9) for value in values})
    if not result or any(value <= 0 for value in result):
        raise ValueError("At least one positive bond length is required.")
    return result


def load_phase6_conclusions(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Phase-6 conclusions not found: {path}. Run the Phase-6 "
            "consolidation first."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("analysis_status") != (
        "complete_with_reproducible_best_basin_rescues"
    ):
        raise RuntimeError(
            "Phase 7A requires the completed consolidated Phase-6 result."
        )
    loo = value.get("consolidated_leave_one_out_results", {})
    if int(loo.get("num_tests", -1)) != 30:
        raise RuntimeError("Phase-6 conclusions do not contain all 30 tests.")
    if not bool(loo.get("all_tests_scientifically_resolved", False)):
        raise RuntimeError("Phase-6 conclusions are not scientifically resolved.")
    rule = value.get("candidate_invariant_rule", {})
    labels = tuple(rule.get("compact_operator_labels", []))
    if labels != COMPACT_LABELS:
        raise RuntimeError(
            f"Phase-6 compact labels {labels} differ from {COMPACT_LABELS}."
        )
    required_chemical = tuple(
        loo.get(
            "operators_required_for_chemical_accuracy_at_any_tested_geometry",
            [],
        )
    )
    if set(required_chemical) != {"D0", "D3", "D12", "D15"}:
        raise RuntimeError(
            "Phase-6 chemical-critical operator set differs from the audited set."
        )
    if not str(value.get("consolidation_fingerprint", "")):
        raise RuntimeError("Phase-6 conclusions lack a consolidation fingerprint.")
    return value


def build_target_rhf(bond_length: float, basis: str) -> TargetRHFData:
    from pyscf import gto, scf

    molecule = gto.M(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-10
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.conv_tol = 1e-10
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(
            f"Target-basis RHF failed at R={bond_length} Angstrom."
        )
    frozen_core = (int(molecule.nelectron) - ACTIVE_ELECTRONS) // 2
    if frozen_core < 0 or 2 * frozen_core + ACTIVE_ELECTRONS != molecule.nelectron:
        raise RuntimeError("The target frozen-core electron count is inconsistent.")
    return TargetRHFData(
        bond_length=bond_length,
        molecule=molecule,
        mean_field=mean_field,
        num_frozen_core_orbitals=frozen_core,
    )


def cross_basis_mapping(
    source: p3.OrbitalSnapshot,
    target: TargetRHFData,
    reference_groups: Sequence[tuple[int, ...]],
    overlap_threshold: float,
) -> tuple[CrossBasisMapping, list[dict[str, Any]]]:
    from pyscf import gto

    if source.reference_to_canonical is None:
        raise RuntimeError("Source STO-3G orbital tracking has not been applied.")
    source_mapping = np.asarray(source.reference_to_canonical, dtype=int)
    source_tracked = source.active_coefficients[:, source_mapping]
    target_coefficients = np.asarray(target.mean_field.mo_coeff, dtype=float)
    target_energies = np.asarray(target.mean_field.mo_energy, dtype=float)
    target_occupations = np.asarray(target.mean_field.mo_occ, dtype=float)
    cross_overlap = gto.intor_cross(
        "int1e_ovlp",
        source.molecule,
        target.molecule,
    )
    overlaps = source_tracked.T @ cross_overlap @ target_coefficients

    occupied_candidates = [
        index
        for index, occupation in enumerate(target_occupations)
        if index >= target.num_frozen_core_orbitals
        and np.isclose(occupation, 2.0, atol=1e-8)
    ]
    virtual_candidates = [
        index
        for index, occupation in enumerate(target_occupations)
        if np.isclose(occupation, 0.0, atol=1e-8)
    ]
    if len(occupied_candidates) != 1:
        raise RuntimeError(
            f"Expected one target valence occupied MO, found {occupied_candidates}."
        )
    if len(virtual_candidates) < ACTIVE_SPATIAL_ORBITALS - 1:
        raise RuntimeError("The target basis has too few virtual orbitals.")

    reference_to_target = np.full(ACTIVE_SPATIAL_ORBITALS, -1, dtype=int)
    reference_to_target[0] = occupied_candidates[0]
    virtual_overlap = np.abs(overlaps[1:, virtual_candidates])
    rows, columns = linear_sum_assignment(-virtual_overlap)
    if len(rows) != ACTIVE_SPATIAL_ORBITALS - 1:
        raise RuntimeError("Cross-basis virtual-orbital assignment is incomplete.")
    for row, column in zip(rows, columns):
        reference_to_target[int(row) + 1] = virtual_candidates[int(column)]
    if np.any(reference_to_target < 0):
        raise RuntimeError("Cross-basis orbital assignment is incomplete.")
    if len(set(reference_to_target.tolist())) != ACTIVE_SPATIAL_ORBITALS:
        raise RuntimeError("Cross-basis orbital assignment is not one-to-one.")

    signed = overlaps[np.arange(ACTIVE_SPATIAL_ORBITALS), reference_to_target]
    absolute = np.abs(signed)
    group_quality = np.empty(ACTIVE_SPATIAL_ORBITALS, dtype=float)
    group_by_orbital: dict[int, tuple[int, tuple[int, ...]]] = {}
    for group_id, group in enumerate(reference_groups):
        group_rows = np.asarray(group, dtype=int)
        group_columns = reference_to_target[group_rows]
        block = overlaps[np.ix_(group_rows, group_columns)]
        quality = float(np.min(np.linalg.svd(block, compute_uv=False)))
        group_quality[group_rows] = quality
        for orbital in group:
            group_by_orbital[int(orbital)] = (group_id, tuple(group))

    selected = sorted(int(value) for value in reference_to_target)
    reference_to_active = [
        selected.index(int(target_index)) for target_index in reference_to_target
    ]
    mapped_occupations = target_occupations[reference_to_target]
    occupation_ok = bool(
        np.isclose(mapped_occupations[0], 2.0, atol=1e-8)
        and np.allclose(mapped_occupations[1:], 0.0, atol=1e-8)
        and selected[0] >= target.num_frozen_core_orbitals
        and not set(selected).intersection(
            range(target.num_frozen_core_orbitals)
        )
    )
    reliable = bool(
        source.tracking_reliable
        and occupation_ok
        and float(np.min(group_quality)) >= overlap_threshold
    )

    detail_rows: list[dict[str, Any]] = []
    for reference_orbital in range(ACTIVE_SPATIAL_ORBITALS):
        group_id, group = group_by_orbital[reference_orbital]
        source_canonical = int(source_mapping[reference_orbital])
        target_full = int(reference_to_target[reference_orbital])
        detail_rows.append(
            {
                "bond_length_angstrom": target.bond_length,
                "geometry_key": geometry_key(target.bond_length),
                "reference_active_orbital": reference_orbital,
                "source_current_canonical_active_orbital": source_canonical,
                "target_full_mo_index": target_full,
                "target_matched_active_position": reference_to_active[
                    reference_orbital
                ],
                "source_orbital_energy_ha": float(
                    source.active_energies[source_canonical]
                ),
                "target_orbital_energy_ha": float(target_energies[target_full]),
                "source_orbital_occupation": float(
                    source.active_occupations[source_canonical]
                ),
                "target_orbital_occupation": float(
                    target_occupations[target_full]
                ),
                "signed_cross_basis_overlap": float(signed[reference_orbital]),
                "absolute_cross_basis_overlap": float(
                    absolute[reference_orbital]
                ),
                "reference_degenerate_group_id": group_id,
                "reference_degenerate_group_json": json.dumps(list(group)),
                "matched_group_min_singular_value": float(
                    group_quality[reference_orbital]
                ),
                "source_tracking_reliable": bool(source.tracking_reliable),
                "cross_basis_mapping_reliable": reliable,
            }
        )

    result = CrossBasisMapping(
        bond_length=target.bond_length,
        source_reference_to_source_canonical=source_mapping.tolist(),
        reference_to_target_full_mo=reference_to_target.tolist(),
        reference_to_target_active_position=reference_to_active,
        selected_target_full_mo_indices=selected,
        signed_overlaps=signed.tolist(),
        absolute_overlaps=absolute.tolist(),
        group_quality_by_reference_orbital=group_quality.tolist(),
        target_full_mo_occupations=target_occupations.tolist(),
        source_tracking_reliable=bool(source.tracking_reliable),
        occupation_pattern_ok=occupation_ok,
        mapping_reliable=reliable,
    )
    return result, detail_rows


def excitation_pools() -> tuple[
    list[diag.Excitation],
    list[diag.Excitation],
    list[diag.Excitation],
]:
    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )

    singles = [
        diag.canonical_excitation(item)
        for item in generate_fermionic_excitations(
            num_excitations=1,
            num_spatial_orbitals=ACTIVE_SPATIAL_ORBITALS,
            num_particles=(1, 1),
            preserve_spin=True,
        )
    ]
    doubles = [
        diag.canonical_excitation(item)
        for item in generate_fermionic_excitations(
            num_excitations=2,
            num_spatial_orbitals=ACTIVE_SPATIAL_ORBITALS,
            num_particles=(1, 1),
            preserve_spin=True,
        )
    ]
    full = [*singles, *doubles]
    if len(singles) != EXPECTED_SINGLES or len(doubles) != EXPECTED_DOUBLES:
        raise RuntimeError(
            f"Expected 8 singles and 16 doubles, found "
            f"{len(singles)} and {len(doubles)}."
        )
    if len(set(full)) != EXPECTED_FULL_UCCSD:
        raise RuntimeError("The generated full24 pool contains duplicates.")
    return singles, doubles, full


def operator_label(global_index: int) -> str:
    if global_index < EXPECTED_SINGLES:
        return f"S{global_index}"
    return f"D{global_index - EXPECTED_SINGLES}"


def compact_reference_indices() -> list[int]:
    return [
        *ACTIVE_SINGLE_LOCAL_IDS,
        *(EXPECTED_SINGLES + index for index in ACTIVE_DOUBLE_LOCAL_IDS),
    ]


def translate_spin_orbital(index: int, spatial_mapping: Sequence[int]) -> int:
    spin = int(index) // ACTIVE_SPATIAL_ORBITALS
    spatial = int(index) % ACTIVE_SPATIAL_ORBITALS
    return int(spatial_mapping[spatial]) + spin * ACTIVE_SPATIAL_ORBITALS


def translate_excitation(
    excitation: diag.Excitation,
    spatial_mapping: Sequence[int],
) -> diag.Excitation:
    occupied, virtual = excitation
    return diag.canonical_excitation(
        (
            tuple(
                translate_spin_orbital(index, spatial_mapping)
                for index in occupied
            ),
            tuple(
                translate_spin_orbital(index, spatial_mapping)
                for index in virtual
            ),
        )
    )


def freeze_operator_support(
    phase6: dict[str, Any],
    mappings: dict[str, CrossBasisMapping],
    full_pool: Sequence[diag.Excitation],
    overlap_threshold: float,
    degeneracy_tolerance_ha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[diag.Excitation]]]:
    reference_indices = compact_reference_indices()
    if [operator_label(index) for index in reference_indices] != list(COMPACT_LABELS):
        raise RuntimeError("The hard-coded compact10 ordering is inconsistent.")

    mapping_rows: list[dict[str, Any]] = []
    mapped_by_geometry: dict[str, list[diag.Excitation]] = {}
    frozen_geometry: dict[str, Any] = {}
    for key in sorted(mappings, key=float):
        mapping = mappings[key]
        mapped_excitations: list[diag.Excitation] = []
        mapped_ids: list[int] = []
        for reference_index in reference_indices:
            reference_excitation = full_pool[reference_index]
            mapped_excitation = translate_excitation(
                reference_excitation,
                mapping.reference_to_target_active_position,
            )
            if mapped_excitation not in full_pool:
                raise RuntimeError(
                    f"Mapped excitation is absent from full24 at R={key}: "
                    f"{mapped_excitation}."
                )
            mapped_id = list(full_pool).index(mapped_excitation)
            mapped_excitations.append(mapped_excitation)
            mapped_ids.append(mapped_id)
            mapping_rows.append(
                {
                    "bond_length_angstrom": mapping.bond_length,
                    "geometry_key": key,
                    "reference_operator_label": operator_label(reference_index),
                    "reference_global_uccsd_index": reference_index,
                    "reference_excitation_json": json.dumps(
                        diag.excitation_to_json(reference_excitation)
                    ),
                    "target_operator_label": operator_label(mapped_id),
                    "target_global_uccsd_index": mapped_id,
                    "target_excitation_json": json.dumps(
                        diag.excitation_to_json(mapped_excitation)
                    ),
                    "operator_type": (
                        "single"
                        if reference_index < EXPECTED_SINGLES
                        else "double"
                    ),
                    "operator_ordering": "phase6_compact10_reference_order",
                }
            )
        if len(set(mapped_excitations)) != len(COMPACT_LABELS):
            raise RuntimeError(f"Mapped compact10 is not unique at R={key}.")
        mapped_by_geometry[key] = mapped_excitations
        frozen_geometry[key] = {
            "bond_length_angstrom": mapping.bond_length,
            "selected_target_full_mo_indices": (
                mapping.selected_target_full_mo_indices
            ),
            "reference_to_target_active_position": (
                mapping.reference_to_target_active_position
            ),
            "reference_operator_labels": list(COMPACT_LABELS),
            "mapped_target_global_uccsd_ids": mapped_ids,
            "mapped_target_operator_labels": [
                operator_label(index) for index in mapped_ids
            ],
            "mapped_target_excitations": [
                diag.excitation_to_json(item) for item in mapped_excitations
            ],
        }

    payload = {
        "schema_version": 1,
        "analysis_phase": "Phase 7A matched-active-space cross-basis audit",
        "selection_frozen_before_target_energy_evaluation": True,
        "source_basis": SOURCE_BASIS,
        "target_basis": TARGET_BASIS,
        "active_electrons": ACTIVE_ELECTRONS,
        "active_spatial_orbitals": ACTIVE_SPATIAL_ORBITALS,
        "target_bond_lengths_angstrom": list(TARGET_BOND_LENGTHS),
        "source_tracking_bond_lengths_angstrom": list(
            SOURCE_TRACKING_BOND_LENGTHS
        ),
        "phase6_consolidation_fingerprint": phase6[
            "consolidation_fingerprint"
        ],
        "source_tracking_method": p3.TRACKING_METHOD,
        "cross_basis_tracking_method": (
            "occupancy_constrained_maximum_absolute_cross_basis_mo_overlap"
        ),
        "orbital_overlap_threshold": overlap_threshold,
        "degeneracy_tolerance_ha": degeneracy_tolerance_ha,
        "geometry_supports": frozen_geometry,
        "evaluation_thresholds_ha": {
            "strict_equivalence_vs_full24": STRICT_EQUIVALENCE_HA,
            "chemical_vs_exact": CHEMICAL_ACCURACY_HA,
            "good_vs_exact": GOOD_ACCURACY_HA,
            "acceptable_vs_exact": ACCEPTABLE_ACCURACY_HA,
        },
        "preregistered_future_variants": {
            "full24": "All eight singles and sixteen doubles.",
            "transferred_compact10": (
                "The frozen mapped Phase-6 compact10 support."
            ),
            "mp2_type_matched10": (
                "Four singles and six doubles ranked only by preregistered "
                "HF/MP2 diagnostics; selection must be frozen before VQE."
            ),
            "random_type_matched10": (
                "Random four-single/six-double supports with declared seeds."
            ),
        },
        "preregistered_future_restart_policy": {
            "base_restarts": 2,
            "rescue_restarts": 3,
            "rescue_triggers": [
                "optimizer_failure",
                "nonfinite_energy",
                "variational_violation",
                "restart_energy_disagreement_above_1e-7_ha",
                "chemical_or_strict_classification_disagreement",
            ],
        },
        "claim_boundary": (
            "This frozen support tests a basis change within matched LiH "
            "2e,5o active spaces. It does not yet test an expanded active "
            "space or a different molecule."
        ),
    }
    payload["support_fingerprint"] = fingerprint(payload)
    return payload, mapping_rows, mapped_by_geometry


def write_or_validate_frozen_support(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("support_fingerprint") != payload["support_fingerprint"]:
            raise RuntimeError(
                f"Existing frozen support {path} has a different fingerprint. "
                "Use a new output directory; do not overwrite a preregistered "
                "support."
            )
        return
    write_json(path, payload)


def build_active_system(
    bond_length: float,
    active_orbitals: Sequence[int],
    target_rhf: TargetRHFData,
) -> ActiveSystemData:
    from qiskit.quantum_info import Statevector
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    from qiskit_nature.units import DistanceUnit

    selected = [int(value) for value in active_orbitals]
    if selected != sorted(selected) or len(set(selected)) != ACTIVE_SPATIAL_ORBITALS:
        raise ValueError("Target active-orbital indices must be sorted and unique.")
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis=TARGET_BASIS,
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
        conv_tol=1e-10,
    )
    driver.run_pyscf()
    driver_schema = driver.to_qcschema(include_dipole=False)
    raw_problem = driver.to_problem(include_dipole=False)
    if max(selected) >= int(raw_problem.num_spatial_orbitals):
        raise RuntimeError("A selected target MO index exceeds the driver space.")
    transformer = ActiveSpaceTransformer(
        num_electrons=(1, 1),
        num_spatial_orbitals=ACTIVE_SPATIAL_ORBITALS,
        active_orbitals=selected,
    )
    problem = transformer.transform(raw_problem)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    constants = {
        str(key): float(np.real(value))
        for key, value in problem.hamiltonian.constants.items()
    }
    total_offset = float(sum(constants.values()))

    exact_solver = NumPyMinimumEigensolver(
        filter_criterion=problem.get_default_filter_criterion()
    )
    exact_result = GroundStateEigensolver(mapper, exact_solver).solve(problem)
    exact_total = diag.total_energy_from_result(exact_result)
    exact_active = diag.raw_eigenvalue_from_result(exact_result)

    hf_circuit = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    )
    hf_state = Statevector.from_instruction(hf_circuit)
    hf_active = float(np.real(hf_state.expectation_value(qubit_op)))
    hf_total_qubit = hf_active + total_offset
    if problem.reference_energy is None:
        raise RuntimeError("The target active problem has no HF reference energy.")
    hf_total_driver = float(np.real(problem.reference_energy))

    singles = [
        diag.canonical_excitation(item)
        for item in generate_fermionic_excitations(
            num_excitations=1,
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            preserve_spin=True,
        )
    ]
    doubles = diag.candidate_double_excitations(problem, preserve_spin=True)
    full_pool = [*singles, *doubles]

    orbital_difference: float | None = None
    driver_energies = getattr(raw_problem, "orbital_energies", None)
    if driver_energies is not None:
        driver_array = np.asarray(driver_energies, dtype=float)
        direct_array = np.asarray(target_rhf.mean_field.mo_energy, dtype=float)
        if driver_array.shape == direct_array.shape:
            orbital_difference = float(np.max(np.abs(driver_array - direct_array)))

    gauge_min_diagonal: float | None = None
    gauge_max_offdiagonal: float | None = None
    driver_coefficients_flat = getattr(
        driver_schema.wavefunction,
        "scf_orbitals_a",
        None,
    )
    if driver_coefficients_flat is not None:
        num_driver_mos = int(driver_schema.properties.calcinfo_nmo)
        flat = np.asarray(driver_coefficients_flat, dtype=float)
        if flat.size % num_driver_mos == 0:
            driver_coefficients = flat.reshape(
                flat.size // num_driver_mos,
                num_driver_mos,
            )
            direct_coefficients = np.asarray(
                target_rhf.mean_field.mo_coeff,
                dtype=float,
            )
            if driver_coefficients.shape == direct_coefficients.shape:
                ao_overlap = np.asarray(
                    target_rhf.mean_field.get_ovlp(),
                    dtype=float,
                )
                gauge_overlap = (
                    direct_coefficients.T
                    @ ao_overlap
                    @ driver_coefficients
                )
                selected_array = np.asarray(selected, dtype=int)
                gauge_min_diagonal = float(
                    np.min(
                        np.abs(
                            gauge_overlap[selected_array, selected_array]
                        )
                    )
                )
                selected_rows = np.abs(gauge_overlap[selected_array, :]).copy()
                for row, orbital_index in enumerate(selected):
                    selected_rows[row, orbital_index] = 0.0
                gauge_max_offdiagonal = float(np.max(selected_rows))

    configuration = {
        "atom": f"Li 0 0 0; H 0 0 {bond_length}",
        "basis": TARGET_BASIS,
        "charge": 0,
        "spin": 0,
        "unit": "ANGSTROM",
        "active_electrons_alpha_beta": [1, 1],
        "active_spatial_orbitals": ACTIVE_SPATIAL_ORBITALS,
        "active_full_mo_indices": selected,
        "transformer": "ActiveSpaceTransformer",
        "mapper": "JordanWignerMapper",
        "preserve_spin": True,
    }
    system_fingerprint = fingerprint(
        {
            "configuration": configuration,
            "full_uccsd_pool": [
                diag.excitation_to_json(item) for item in full_pool
            ],
        }
    )
    return ActiveSystemData(
        bond_length=bond_length,
        active_orbitals=selected,
        raw_problem=raw_problem,
        problem=problem,
        mapper=mapper,
        qubit_op=qubit_op,
        constants=constants,
        total_offset_ha=total_offset,
        exact_active_electronic_ha=exact_active,
        exact_total_ha=exact_total,
        hf_active_electronic_ha=hf_active,
        hf_total_qubit_ha=hf_total_qubit,
        hf_total_driver_ha=hf_total_driver,
        singles=singles,
        doubles=doubles,
        full_pool=full_pool,
        fingerprint=system_fingerprint,
        driver_orbital_energy_max_abs_difference_ha=orbital_difference,
        selected_driver_gauge_min_abs_diagonal_overlap=gauge_min_diagonal,
        selected_driver_gauge_max_abs_offdiagonal_overlap=(
            gauge_max_offdiagonal
        ),
    )


def build_ansatz(system: ActiveSystemData, excitations: Sequence[diag.Excitation]) -> Any:
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

    initial_state = HartreeFock(
        system.problem.num_spatial_orbitals,
        system.problem.num_particles,
        system.mapper,
    )
    ansatz = UCC(
        num_spatial_orbitals=system.problem.num_spatial_orbitals,
        num_particles=system.problem.num_particles,
        excitations=diag.custom_excitation_generator(excitations),
        qubit_mapper=system.mapper,
        preserve_spin=True,
        reps=1,
        initial_state=initial_state,
    )
    _ = ansatz.num_parameters
    if ansatz.num_parameters != len(excitations):
        raise RuntimeError(
            f"Ansatz has {ansatz.num_parameters} parameters; expected "
            f"{len(excitations)}."
        )
    return ansatz


def nullable_resources() -> dict[str, int | None]:
    return {
        "logical_depth": None,
        "logical_size": None,
        "compiled_depth": None,
        "compiled_size": None,
        "compiled_cx": None,
    }


def audit_active_system(
    system: ActiveSystemData,
    source_system: diag.SystemData,
    source_snapshot: p3.OrbitalSnapshot,
    target_rhf: TargetRHFData,
    mapping: CrossBasisMapping,
    mapped_compact: Sequence[diag.Excitation],
    audit_tolerance_ha: float,
    skip_resources: bool,
    seed_transpiler: int,
) -> dict[str, Any]:
    exact_reconstruction = abs(
        system.exact_active_electronic_ha
        + system.total_offset_ha
        - system.exact_total_ha
    )
    hf_reconstruction = abs(
        system.hf_total_qubit_ha - system.hf_total_driver_ha
    )
    direct_target_hf_residual = abs(
        system.hf_total_driver_ha - float(target_rhf.mean_field.e_tot)
    )
    source_hf_residual = abs(
        source_system.hf_total_driver_ha - source_snapshot.scf_total_energy_ha
    )
    orbital_energy_residual = (
        system.driver_orbital_energy_max_abs_difference_ha
    )
    gauge_min_diagonal = (
        system.selected_driver_gauge_min_abs_diagonal_overlap
    )
    gauge_max_offdiagonal = (
        system.selected_driver_gauge_max_abs_offdiagonal_overlap
    )
    checks = {
        "source_hamiltonian_audit_passed": bool(
            diag.validate_reference(source_system, [], audit_tolerance_ha)[
                "audit_passed"
            ]
        ),
        "source_orbital_tracking_reliable": bool(
            source_snapshot.tracking_reliable
        ),
        "cross_basis_mapping_reliable": bool(mapping.mapping_reliable),
        "cross_basis_occupation_pattern_ok": bool(
            mapping.occupation_pattern_ok
        ),
        "target_active_electrons_are_2": int(sum(system.problem.num_particles))
        == ACTIVE_ELECTRONS,
        "target_active_spatial_orbitals_are_5": int(
            system.problem.num_spatial_orbitals
        )
        == ACTIVE_SPATIAL_ORBITALS,
        "target_num_qubits_is_10": int(system.qubit_op.num_qubits)
        == EXPECTED_QUBITS,
        "target_pool_is_full24": len(system.full_pool) == EXPECTED_FULL_UCCSD,
        "target_compact_support_has_10_unique_operators": (
            len(mapped_compact) == len(COMPACT_LABELS)
            and len(set(mapped_compact)) == len(COMPACT_LABELS)
        ),
        "target_exact_total_reconstructs_from_offset": (
            exact_reconstruction <= audit_tolerance_ha
        ),
        "target_hf_qubit_matches_driver": (
            hf_reconstruction <= audit_tolerance_ha
        ),
        "target_driver_hf_matches_mapping_scf": (
            direct_target_hf_residual <= audit_tolerance_ha
        ),
        "source_driver_hf_matches_tracking_scf": (
            source_hf_residual <= audit_tolerance_ha
        ),
        "target_exact_not_above_hf": (
            system.exact_total_ha
            <= system.hf_total_driver_ha + audit_tolerance_ha
        ),
        "target_driver_orbital_order_matches_mapping_scf": (
            orbital_energy_residual is not None
            and orbital_energy_residual <= audit_tolerance_ha
        ),
        "target_driver_mo_gauge_matches_mapping_scf": (
            gauge_min_diagonal is not None
            and gauge_max_offdiagonal is not None
            and 1.0 - gauge_min_diagonal <= audit_tolerance_ha
            and gauge_max_offdiagonal <= audit_tolerance_ha
        ),
    }
    if skip_resources:
        full_resources = nullable_resources()
        compact_resources = nullable_resources()
    else:
        full_resources = diag.circuit_resource_metrics(
            build_ansatz(system, system.full_pool), seed_transpiler
        )
        compact_resources = diag.circuit_resource_metrics(
            build_ansatz(system, mapped_compact), seed_transpiler
        )
    return {
        "audit_passed": bool(all(checks.values())),
        "checks": checks,
        "bond_length_angstrom": system.bond_length,
        "geometry_key": geometry_key(system.bond_length),
        "source_configuration_fingerprint": source_system.fingerprint,
        "target_configuration_fingerprint": system.fingerprint,
        "selected_target_full_mo_indices": system.active_orbitals,
        "reference_to_target_active_position": (
            mapping.reference_to_target_active_position
        ),
        "source_dimensions": {
            "active_electrons": int(sum(source_system.problem.num_particles)),
            "active_spatial_orbitals": int(
                source_system.problem.num_spatial_orbitals
            ),
            "qubits": int(source_system.qubit_op.num_qubits),
        },
        "target_full_basis_dimensions": {
            "electrons": int(sum(system.raw_problem.num_particles)),
            "spatial_orbitals": int(system.raw_problem.num_spatial_orbitals),
            "frozen_core_orbitals": target_rhf.num_frozen_core_orbitals,
        },
        "target_matched_active_dimensions": {
            "active_electrons": int(sum(system.problem.num_particles)),
            "active_spatial_orbitals": int(system.problem.num_spatial_orbitals),
            "spin_orbitals": int(2 * system.problem.num_spatial_orbitals),
            "qubits": int(system.qubit_op.num_qubits),
            "pauli_terms": int(len(system.qubit_op)),
            "singles": len(system.singles),
            "doubles": len(system.doubles),
            "full_uccsd_operators": len(system.full_pool),
            "transferred_compact_operators": len(mapped_compact),
        },
        "energies_ha": {
            "source_sto3g_exact_total": source_system.exact_total_ha,
            "source_sto3g_hf_total": source_system.hf_total_driver_ha,
            "target_631g_exact_active_electronic": (
                system.exact_active_electronic_ha
            ),
            "target_631g_constant_offsets": system.constants,
            "target_631g_total_constant_offset": system.total_offset_ha,
            "target_631g_exact_total": system.exact_total_ha,
            "target_631g_hf_total": system.hf_total_driver_ha,
            "target_631g_hf_error_vs_exact": (
                system.hf_total_driver_ha - system.exact_total_ha
            ),
        },
        "numerical_residuals_ha": {
            "target_exact_offset_reconstruction": exact_reconstruction,
            "target_hf_qubit_vs_driver": hf_reconstruction,
            "target_driver_hf_vs_mapping_scf": direct_target_hf_residual,
            "source_driver_hf_vs_tracking_scf": source_hf_residual,
            "target_driver_orbital_energies_vs_mapping_scf": (
                orbital_energy_residual
            ),
            "one_minus_selected_driver_gauge_min_abs_diagonal_overlap": (
                None
                if gauge_min_diagonal is None
                else 1.0 - gauge_min_diagonal
            ),
            "selected_driver_gauge_max_abs_offdiagonal_overlap": (
                gauge_max_offdiagonal
            ),
        },
        "cross_basis_overlap": {
            "minimum_individual_absolute_overlap": min(
                mapping.absolute_overlaps
            ),
            "minimum_group_singular_value": min(
                mapping.group_quality_by_reference_orbital
            ),
        },
        "circuit_resources": {
            "full24": {
                "num_parameters": len(system.full_pool),
                **full_resources,
            },
            "transferred_compact10": {
                "num_parameters": len(mapped_compact),
                **compact_resources,
            },
        },
    }


def geometry_csv_row(audit: dict[str, Any]) -> dict[str, Any]:
    target_full = audit["target_full_basis_dimensions"]
    target_active = audit["target_matched_active_dimensions"]
    energies = audit["energies_ha"]
    overlaps = audit["cross_basis_overlap"]
    resources = audit["circuit_resources"]
    return {
        "bond_length_angstrom": audit["bond_length_angstrom"],
        "geometry_key": audit["geometry_key"],
        "audit_passed": audit["audit_passed"],
        "source_configuration_fingerprint": audit[
            "source_configuration_fingerprint"
        ],
        "target_configuration_fingerprint": audit[
            "target_configuration_fingerprint"
        ],
        "selected_target_full_mo_indices_json": json.dumps(
            audit["selected_target_full_mo_indices"]
        ),
        "reference_to_target_active_position_json": json.dumps(
            audit["reference_to_target_active_position"]
        ),
        "target_full_spatial_orbitals": target_full["spatial_orbitals"],
        "target_active_electrons": target_active["active_electrons"],
        "target_active_spatial_orbitals": target_active[
            "active_spatial_orbitals"
        ],
        "target_qubits": target_active["qubits"],
        "target_pauli_terms": target_active["pauli_terms"],
        "target_exact_total_energy_ha": energies["target_631g_exact_total"],
        "target_hf_total_energy_ha": energies["target_631g_hf_total"],
        "target_hf_error_exact_ha": energies[
            "target_631g_hf_error_vs_exact"
        ],
        "minimum_individual_absolute_cross_basis_overlap": overlaps[
            "minimum_individual_absolute_overlap"
        ],
        "minimum_group_cross_basis_singular_value": overlaps[
            "minimum_group_singular_value"
        ],
        "full24_parameters": resources["full24"]["num_parameters"],
        "full24_compiled_depth": resources["full24"]["compiled_depth"],
        "full24_compiled_cx": resources["full24"]["compiled_cx"],
        "compact10_parameters": resources["transferred_compact10"][
            "num_parameters"
        ],
        "compact10_compiled_depth": resources["transferred_compact10"][
            "compiled_depth"
        ],
        "compact10_compiled_cx": resources["transferred_compact10"][
            "compiled_cx"
        ],
    }


def target_pool_rows(
    systems: dict[str, ActiveSystemData],
    mapped_by_geometry: dict[str, list[diag.Excitation]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(systems, key=float):
        system = systems[key]
        selected = set(mapped_by_geometry[key])
        for global_index, excitation in enumerate(system.full_pool):
            rows.append(
                {
                    "bond_length_angstrom": system.bond_length,
                    "geometry_key": key,
                    "target_operator_label": operator_label(global_index),
                    "target_global_uccsd_index": global_index,
                    "operator_type": (
                        "single"
                        if global_index < EXPECTED_SINGLES
                        else "double"
                    ),
                    "target_excitation_json": json.dumps(
                        diag.excitation_to_json(excitation)
                    ),
                    "included_in_transferred_compact10": excitation in selected,
                }
            )
    return rows


def write_report(path: Path, final: dict[str, Any]) -> None:
    lines = [
        "# LiH Phase 7A Cross-Basis Audit Report",
        "",
        f"Analysis status: **{final['analysis_status']}**",
        "",
        "No VQE optimization was performed by this audit.",
        "",
        "## Audit summary",
        "",
        "| R (Å) | Passed | Target MOs | Min overlap | Min group quality | "
        "Exact (Ha) | HF error (Ha) |",
        "|---:|:---:|---|---:|---:|---:|---:|",
    ]
    for audit in final["geometry_audits"]:
        overlaps = audit["cross_basis_overlap"]
        energies = audit["energies_ha"]
        lines.append(
            f"| {audit['bond_length_angstrom']:.3f} | "
            f"{audit['audit_passed']} | "
            f"{audit['selected_target_full_mo_indices']} | "
            f"{overlaps['minimum_individual_absolute_overlap']:.6f} | "
            f"{overlaps['minimum_group_singular_value']:.6f} | "
            f"{energies['target_631g_exact_total']:.12f} | "
            f"{energies['target_631g_hf_error_vs_exact']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen support",
            "",
            f"Support fingerprint: `{final['support_fingerprint']}`",
            "",
            "The transferred compact10 support was selected from RHF orbital "
            "overlaps and frozen before target exact energies were evaluated.",
            "",
            "## Gate decision",
            "",
            (
                "The audit passed. Phase 7A VQE comparisons may proceed."
                if final["audit_passed"]
                else "The audit failed. Do not run Phase 7A VQE comparisons."
            ),
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
            "Audit leakage-free transfer of the LiH STO-3G compact10 support "
            "into matched 6-31G 2e,5o active spaces. No VQE is run."
        )
    )
    parser.add_argument(
        "--phase6-conclusions",
        type=Path,
        default=Path("lih_phase6_final/phase6_final_conclusions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lih_phase7a_audit"),
    )
    parser.add_argument("--source-basis", default=SOURCE_BASIS)
    parser.add_argument("--target-basis", default=TARGET_BASIS)
    parser.add_argument("--orbital-overlap-threshold", type=float, default=0.75)
    parser.add_argument("--degeneracy-tolerance", type=float, default=1e-6)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument("--seed-transpiler", type=int, default=20260804)
    parser.add_argument(
        "--skip-circuit-resources",
        action="store_true",
        help="Skip circuit transpilation while retaining every scientific audit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source_basis.lower().replace("-", "") != SOURCE_BASIS:
        raise ValueError("Phase 7A source basis is frozen as STO-3G.")
    if args.target_basis.lower() != TARGET_BASIS:
        raise ValueError("Phase 7A target basis is frozen as 6-31G.")
    if (
        args.orbital_overlap_threshold <= 0
        or args.orbital_overlap_threshold > 1
    ):
        raise ValueError("Orbital overlap threshold must lie in (0, 1].")
    if args.degeneracy_tolerance <= 0 or args.audit_tolerance <= 0:
        raise ValueError("Audit tolerances must be positive.")

    diag.require_quantum_stack()
    phase6 = load_phase6_conclusions(args.phase6_conclusions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracking_bonds = normalized_values(SOURCE_TRACKING_BOND_LENGTHS)
    target_bonds = normalized_values(TARGET_BOND_LENGTHS)
    print("Building and tracking source STO-3G orbitals...", flush=True)
    source_snapshots = {
        geometry_key(bond): p3.build_orbital_snapshot(
            bond,
            SOURCE_BASIS,
            ACTIVE_ELECTRONS,
            ACTIVE_SPATIAL_ORBITALS,
        )
        for bond in tracking_bonds
    }
    reference_groups, source_tracking, source_tracking_summary = p3.track_orbitals(
        source_snapshots,
        tracking_bonds,
        args.orbital_overlap_threshold,
        args.degeneracy_tolerance,
    )
    source_tracking.to_csv(
        args.output_dir / "phase7a_source_orbital_tracking.csv", index=False
    )
    source_tracking_summary.to_csv(
        args.output_dir / "phase7a_source_orbital_tracking_summary.csv",
        index=False,
    )

    print("Matching STO-3G orbitals into the full 6-31G space...", flush=True)
    target_rhf: dict[str, TargetRHFData] = {}
    mappings: dict[str, CrossBasisMapping] = {}
    orbital_rows: list[dict[str, Any]] = []
    for bond in target_bonds:
        key = geometry_key(bond)
        target = build_target_rhf(bond, TARGET_BASIS)
        mapping, details = cross_basis_mapping(
            source_snapshots[key],
            target,
            reference_groups,
            args.orbital_overlap_threshold,
        )
        target_rhf[key] = target
        mappings[key] = mapping
        orbital_rows.extend(details)
    write_csv(
        args.output_dir / "phase7a_cross_basis_orbital_mapping.csv",
        orbital_rows,
    )

    _, _, reference_full_pool = excitation_pools()
    frozen_support, operator_rows, mapped_by_geometry = freeze_operator_support(
        phase6,
        mappings,
        reference_full_pool,
        args.orbital_overlap_threshold,
        args.degeneracy_tolerance,
    )
    frozen_path = args.output_dir / "phase7a_frozen_support.json"
    write_or_validate_frozen_support(frozen_path, frozen_support)
    write_csv(
        args.output_dir / "phase7a_transferred_operator_mapping.csv",
        operator_rows,
    )
    print(
        "Transferred support frozen before target-energy evaluation: "
        f"{frozen_support['support_fingerprint']}",
        flush=True,
    )

    print("Building and exactly auditing matched 6-31G Hamiltonians...", flush=True)
    systems: dict[str, ActiveSystemData] = {}
    geometry_audits: list[dict[str, Any]] = []
    for bond in target_bonds:
        key = geometry_key(bond)
        source_system = diag.build_system(p3.build_config(bond, SOURCE_BASIS))
        if (
            abs(bond - REFERENCE_BOND_LENGTH) <= 1e-9
            and source_system.fingerprint != EXPECTED_REFERENCE_FINGERPRINT
        ):
            raise RuntimeError(
                "The equilibrium STO-3G fingerprint differs from Phase 1-6."
            )
        target_system = build_active_system(
            bond,
            mappings[key].selected_target_full_mo_indices,
            target_rhf[key],
        )
        if target_system.full_pool != reference_full_pool:
            raise RuntimeError(
                f"Target full24 ordering changed at R={bond}."
            )
        audit = audit_active_system(
            target_system,
            source_system,
            source_snapshots[key],
            target_rhf[key],
            mappings[key],
            mapped_by_geometry[key],
            args.audit_tolerance,
            args.skip_circuit_resources,
            args.seed_transpiler,
        )
        systems[key] = target_system
        geometry_audits.append(audit)

    audit_passed = bool(all(item["audit_passed"] for item in geometry_audits))
    protocol_payload = {
        "analysis_phase": "Phase 7A cross-basis matched-active-space audit",
        "support_fingerprint": frozen_support["support_fingerprint"],
        "phase6_consolidation_fingerprint": phase6[
            "consolidation_fingerprint"
        ],
        "target_configuration_fingerprints": {
            key: systems[key].fingerprint for key in sorted(systems, key=float)
        },
        "source_basis": SOURCE_BASIS,
        "target_basis": TARGET_BASIS,
        "target_bond_lengths_angstrom": target_bonds,
        "source_tracking_bond_lengths_angstrom": tracking_bonds,
        "orbital_overlap_threshold": args.orbital_overlap_threshold,
        "degeneracy_tolerance_ha": args.degeneracy_tolerance,
        "audit_tolerance_ha": args.audit_tolerance,
        "seed_transpiler": args.seed_transpiler,
        "circuit_resources_skipped": args.skip_circuit_resources,
        "no_vqe_performed": True,
    }
    protocol_fingerprint = fingerprint(protocol_payload)
    final = {
        "analysis_status": (
            "audit_passed_support_frozen_no_vqe_performed"
            if audit_passed
            else "audit_failed_do_not_run_vqe"
        ),
        "audit_passed": audit_passed,
        "no_vqe_performed": True,
        "support_fingerprint": frozen_support["support_fingerprint"],
        "protocol_fingerprint": protocol_fingerprint,
        "phase6_consolidation_fingerprint": phase6[
            "consolidation_fingerprint"
        ],
        "source_basis": SOURCE_BASIS,
        "target_basis": TARGET_BASIS,
        "active_space": {
            "electrons": ACTIVE_ELECTRONS,
            "spatial_orbitals": ACTIVE_SPATIAL_ORBITALS,
        },
        "target_bond_lengths_angstrom": target_bonds,
        "source_tracking_bond_lengths_angstrom": tracking_bonds,
        "reference_degeneracy_groups": [list(group) for group in reference_groups],
        "geometry_audits": geometry_audits,
        "software_versions": diag.dependency_versions(),
        "claim_boundary": (
            "This audit establishes only that the Phase-6 compact10 support "
            "can be mapped without leakage into matched 2e,5o LiH/6-31G "
            "active spaces and that the resulting Hamiltonians are internally "
            "consistent. It reports no VQE transfer result and makes no claim "
            "about expanded active spaces or other molecules."
        ),
    }
    write_json(args.output_dir / "phase7a_audit.json", final)
    write_json(
        args.output_dir / "phase7a_preregistered_protocol.json",
        {**protocol_payload, "protocol_fingerprint": protocol_fingerprint},
    )
    write_csv(
        args.output_dir / "phase7a_geometry_audit.csv",
        [geometry_csv_row(item) for item in geometry_audits],
    )
    write_csv(
        args.output_dir / "phase7a_target_operator_pool.csv",
        target_pool_rows(systems, mapped_by_geometry),
    )
    write_report(args.output_dir / "PHASE7A_AUDIT_REPORT.md", final)

    frame = pd.DataFrame([geometry_csv_row(item) for item in geometry_audits])
    print("\nPhase 7A audit summary:")
    print(
        frame[
            [
                "bond_length_angstrom",
                "audit_passed",
                "selected_target_full_mo_indices_json",
                "target_full_spatial_orbitals",
                "target_exact_total_energy_ha",
                "target_hf_error_exact_ha",
                "minimum_group_cross_basis_singular_value",
                "full24_compiled_cx",
                "compact10_compiled_cx",
            ]
        ].to_string(index=False)
    )
    print(
        json.dumps(
            {
                "analysis_status": final["analysis_status"],
                "audit_passed": audit_passed,
                "no_vqe_performed": True,
                "support_fingerprint": final["support_fingerprint"],
                "protocol_fingerprint": final["protocol_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit_passed:
        print(
            "FATAL: Phase 7A audit failed. Do not run target VQE.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
