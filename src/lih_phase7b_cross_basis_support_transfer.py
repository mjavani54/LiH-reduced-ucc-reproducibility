#!/usr/bin/env python3
"""Phase 7B: test cross-basis transfer of a frozen LiH operator support.

This program consumes the accepted Phase-7A.2 rotated LiH/6-31G 2e,5o
Hamiltonian audit.  It rebuilds (but does not diagonalize) the three audited
Hamiltonians, verifies them against the accepted fingerprints and scalar
audit values, freezes all comparison supports, and then runs statevector VQE.

The primary comparison is transferred compact10 versus full24.  A
Hamiltonian-derived Epstein--Nesbet-like perturbative10 support and twenty
composition-matched random10 supports are controls.  The exact 2e,5o values
are read from the accepted Phase-7A.2 audit; no eigensolver is called here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import lih_phase7a_cross_basis_transfer as p7
    import lih_phase7a2_rotated_hamiltonian_audit as p7a2
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place this script beside the Phase-7A, Phase-7A.1, Phase-7A.2, "
        "Phase-3, Phase-2B, and reference-diagnostic Python modules."
    ) from exc


SCHEMA_VERSION = 1
ANALYSIS_PHASE = "Phase 7B cross-basis support transfer"
EXPECTED_PHASE7A2_STATUS = (
    "audit_passed_rotated_hamiltonian_frozen_no_vqe_performed"
)
EXPECTED_BOND_LENGTHS = (1.595, 2.5, 3.0)
EXPECTED_FULL_OPERATORS = 24
EXPECTED_COMPACT_OPERATORS = 10
EXPECTED_RANDOM_SUPPORTS = 20
EXPECTED_RANDOM_SINGLES = 4
EXPECTED_RANDOM_DOUBLES = 6
REFERENCE_BOND_LENGTH = 1.595

STRICT_EQUIVALENCE_HA = 1e-6
CHEMICAL_ACCURACY_HA = 0.0016
GOOD_ACCURACY_HA = 0.005
ACCEPTABLE_ACCURACY_HA = 0.01


@dataclass(frozen=True)
class Variant:
    name: str
    selected_ids: tuple[int, ...]
    family: str
    scientific_role: str


@dataclass
class BuiltSystem:
    optimization_system: diag.SystemData
    bond_length: float
    raw_problem: Any
    rotated_full_problem: Any
    driver_coefficients: np.ndarray
    direct_to_driver_gauge: np.ndarray
    full_rotation_driver_mo: np.ndarray
    gauge_orthogonality_residual: float
    physical_rotated_orbital_residual: float
    hermiticity_residual: float
    current_gauge_configuration_fingerprint: str


@dataclass
class Phase7BInputs:
    phase7a2_audit: dict[str, Any]
    phase7a2_frozen_support: dict[str, Any]
    phase7a2_protocol: dict[str, Any]
    audits_by_geometry: dict[str, dict[str, Any]]


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
        return None if not np.isfinite(value) else float(value)
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
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")


def validate_signed_payload(payload: dict[str, Any], field: str) -> None:
    supplied = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not supplied or supplied != p7.fingerprint(unsigned):
        raise RuntimeError(f"Invalid or modified fingerprint field: {field}")


def write_or_validate_signed_payload(
    path: Path,
    payload: dict[str, Any],
    fingerprint_field: str,
) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_signed_payload(existing, fingerprint_field)
        if existing[fingerprint_field] != payload[fingerprint_field]:
            raise RuntimeError(
                f"Existing preregistration {path} has a different fingerprint. "
                "Use a new output directory; do not overwrite it."
            )
        return
    write_json(path, payload)


def load_phase7a2_inputs(phase7a2_dir: Path) -> Phase7BInputs:
    audit_path = phase7a2_dir / "phase7a2_hamiltonian_audit.json"
    support_path = phase7a2_dir / "phase7a2_frozen_rotated_support.json"
    protocol_path = phase7a2_dir / "phase7a2_preregistered_protocol.json"
    for path in (audit_path, support_path, protocol_path):
        require_file(path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    support = json.loads(support_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_signed_payload(support, "rotated_support_fingerprint")
    validate_signed_payload(protocol, "protocol_fingerprint")

    if audit.get("analysis_status") != EXPECTED_PHASE7A2_STATUS:
        raise RuntimeError("Phase 7A.2 did not authorize VQE follow-up.")
    if not bool(audit.get("audit_passed", False)):
        raise RuntimeError("Phase 7A.2 audit_passed is not true.")
    if not bool(audit.get("no_vqe_performed", False)):
        raise RuntimeError("Phase 7A.2 provenance unexpectedly reports VQE.")
    if audit.get("protocol_fingerprint") != protocol.get("protocol_fingerprint"):
        raise RuntimeError("Phase-7A.2 audit and protocol fingerprints differ.")
    if audit.get("rotated_support_fingerprint") != support.get(
        "rotated_support_fingerprint"
    ):
        raise RuntimeError("Phase-7A.2 audit and support fingerprints differ.")
    audits = audit.get("geometry_audits", [])
    audits_by_geometry = {
        geometry_key(item["bond_length_angstrom"]): item for item in audits
    }
    expected_keys = {geometry_key(value) for value in EXPECTED_BOND_LENGTHS}
    if set(audits_by_geometry) != expected_keys:
        raise RuntimeError("Phase 7A.2 does not contain the three fixed geometries.")
    if not all(bool(item.get("audit_passed", False)) for item in audits):
        raise RuntimeError("At least one Phase-7A.2 geometry audit failed.")
    return Phase7BInputs(audit, support, protocol, audits_by_geometry)


def reconcile_with_accepted_physical_orbitals(
    frozen: dict[str, p7a2.FrozenGeometry],
    accepted_support: dict[str, Any],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Express the accepted physical orbitals in each new RHF MO gauge.

    The Phase-7A.2 fingerprint signs the MO coefficients and rotation matrix
    from that specific RHF run.  A later RHF run may choose a different basis
    inside a degenerate virtual subspace, so re-hashing its raw MO-basis
    matrices is invalid.  The AO-basis physical orbitals are invariant to that
    gauge change and are therefore the authoritative reconstruction target.
    """
    rotations = accepted_support.get("geometry_rotations", {})
    if set(rotations) != set(frozen):
        raise RuntimeError(
            "Accepted Phase-7A.2 rotation geometries differ from the "
            "reconstructed geometry set."
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(frozen, key=float):
        item = frozen[key]
        record = rotations[key]
        if (
            abs(float(record["bond_length_angstrom"]) - item.bond_length)
            > 1e-9
        ):
            raise RuntimeError(
                f"Accepted rotation geometry metadata changed for key {key}."
            )
        saved_direct = np.asarray(
            record["direct_target_mo_coefficients_in_ao_basis"], dtype=float
        )
        saved_rotation = np.asarray(
            record["completed_full_rotation_in_direct_target_mo_basis"],
            dtype=float,
        )
        saved_active_ao = np.asarray(
            record["active_orbital_coefficients_in_ao_basis"], dtype=float
        )
        expected_shape = (
            p7a2.FULL_TARGET_ORBITALS,
            p7a2.FULL_TARGET_ORBITALS,
        )
        if saved_direct.shape != expected_shape or saved_rotation.shape != expected_shape:
            raise RuntimeError(
                f"Accepted Phase-7A.2 rotation has an invalid shape at R={item.bond_length}."
            )
        if saved_active_ao.shape != (
            p7a2.FULL_TARGET_ORBITALS,
            p7a2.ACTIVE_ORBITALS,
        ):
            raise RuntimeError(
                f"Accepted active AO orbitals have an invalid shape at R={item.bond_length}."
            )

        saved_physical_full = saved_direct @ saved_rotation
        saved_active_residual = float(
            np.max(np.abs(saved_physical_full[:, 1:6] - saved_active_ao))
        )
        if saved_active_residual > tolerance:
            raise RuntimeError(
                f"Accepted active/full orbital records disagree at "
                f"R={item.bond_length}: {saved_active_residual:.3e}."
            )

        current_direct = np.asarray(item.target.mean_field.mo_coeff, dtype=float)
        ao_overlap = np.asarray(item.target.mean_field.get_ovlp(), dtype=float)
        saved_physical_orthogonality = float(
            np.max(
                np.abs(
                    saved_physical_full.T
                    @ ao_overlap
                    @ saved_physical_full
                    - np.eye(p7a2.FULL_TARGET_ORBITALS)
                )
            )
        )
        if saved_physical_orthogonality > tolerance:
            raise RuntimeError(
                f"Accepted physical orbitals are not orthonormal in the "
                f"current AO metric at R={item.bond_length}: "
                f"{saved_physical_orthogonality:.3e}."
            )

        current_rotation = (
            current_direct.T @ ao_overlap @ saved_physical_full
        )
        current_rotation_orthogonality = float(
            np.max(
                np.abs(
                    current_rotation.T
                    @ current_rotation
                    - np.eye(p7a2.FULL_TARGET_ORBITALS)
                )
            )
        )
        reconstructed_physical = current_direct @ current_rotation
        physical_difference = reconstructed_physical - saved_physical_full
        physical_orbital_norms = np.sqrt(
            np.maximum(
                0.0,
                np.diag(
                    physical_difference.T
                    @ ao_overlap
                    @ physical_difference
                ),
            )
        )
        physical_residual = float(np.max(physical_orbital_norms))
        if max(current_rotation_orthogonality, physical_residual) > tolerance:
            raise RuntimeError(
                f"Accepted physical orbitals failed gauge reconciliation at "
                f"R={item.bond_length}."
            )

        # Freeze the accepted Phase-7A.2 physical orbitals in this run's RHF
        # gauge.  Do not substitute a newly completed QR complement.
        item.full_rotation_direct_mo = current_rotation
        item.active_weights_direct_mo = current_rotation[:, 1:6]
        item.active_orthonormality_residual = float(
            np.max(
                np.abs(
                    item.active_weights_direct_mo.T
                    @ item.active_weights_direct_mo
                    - np.eye(p7a2.ACTIVE_ORBITALS)
                )
            )
        )
        item.full_orthogonality_residual = current_rotation_orthogonality
        rows.append(
            {
                "geometry_key": key,
                "accepted_saved_active_vs_full_physical_residual": (
                    saved_active_residual
                ),
                "accepted_physical_orbital_orthogonality_residual": (
                    saved_physical_orthogonality
                ),
                "current_gauge_rotation_orthogonality_residual": (
                    current_rotation_orthogonality
                ),
                "accepted_physical_orbital_reconstruction_residual": (
                    physical_residual
                ),
                "accepted_phase7a2_physical_orbitals_recovered": True,
            }
        )
    return rows


def build_optimization_system_without_diagonalization(
    frozen: p7a2.FrozenGeometry,
    accepted_audit: dict[str, Any],
    accepted_rotation_record: dict[str, Any],
    audit_tolerance: float,
) -> BuiltSystem:
    """Rebuild an audited system without evaluating an exact eigenpair."""
    from qiskit.quantum_info import Statevector
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

    bond = float(frozen.bond_length)
    driver = PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {bond}",
        basis=p7a2.TARGET_BASIS,
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
    raw_problem = p7a2.unfold_problem_two_body_integrals(raw_problem)
    driver_coefficients = p7a2.extract_driver_coefficients(schema)
    direct_coefficients = np.asarray(frozen.target.mean_field.mo_coeff, dtype=float)
    ao_overlap = np.asarray(frozen.target.mean_field.get_ovlp(), dtype=float)
    gauge = direct_coefficients.T @ ao_overlap @ driver_coefficients
    gauge_residual = float(
        np.max(np.abs(gauge.T @ gauge - np.eye(p7a2.FULL_TARGET_ORBITALS)))
    )
    if gauge_residual > audit_tolerance:
        raise RuntimeError(f"Direct/Qiskit MO gauge audit failed at R={bond}.")

    driver_rotation = gauge.T @ frozen.full_rotation_direct_mo
    physical_direct = direct_coefficients @ frozen.full_rotation_direct_mo
    physical_driver = driver_coefficients @ driver_rotation
    difference = physical_direct - physical_driver
    orbital_norms = np.sqrt(
        np.maximum(0.0, np.diag(difference.T @ ao_overlap @ difference))
    )
    physical_residual = float(np.max(orbital_norms))
    if physical_residual > audit_tolerance:
        raise RuntimeError(f"Physical rotated-orbital audit failed at R={bond}.")

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
        num_spatial_orbitals=p7a2.ACTIVE_ORBITALS,
        active_orbitals=[1, 2, 3, 4, 5],
    )
    problem = active_transformer.transform(rotated_full_problem)
    mapper = JordanWignerMapper()
    fermionic_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)
    constants = {
        str(key): float(np.real(value))
        for key, value in problem.hamiltonian.constants.items()
    }
    total_offset = float(sum(constants.values()))

    hf_circuit = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    )
    hf_state = Statevector.from_instruction(hf_circuit)
    hf_active = float(np.real(hf_state.expectation_value(qubit_op)))
    hf_total_qubit = hf_active + total_offset
    if problem.reference_energy is None:
        raise RuntimeError("Rotated target problem lacks an HF reference energy.")
    hf_total_driver = float(np.real(problem.reference_energy))
    hermiticity = p7a2.sparse_hermiticity_residual(qubit_op)

    configuration = {
        "atom": f"Li 0 0 0; H 0 0 {bond}",
        "basis": p7a2.TARGET_BASIS,
        "rotation": "phase7a1_block_preserving_maximum_overlap",
        "full_rotation_direct_mo": frozen.full_rotation_direct_mo.tolist(),
        "active_full_rotated_indices": [1, 2, 3, 4, 5],
        "active_particles": [1, 1],
        "active_spatial_orbitals": p7a2.ACTIVE_ORBITALS,
        "mapper": "JordanWignerMapper",
    }
    current_gauge_configuration_fingerprint = p7.fingerprint(configuration)
    accepted_fingerprint = accepted_audit["rotated_configuration_fingerprint"]
    accepted_configuration = {
        **configuration,
        "full_rotation_direct_mo": accepted_rotation_record[
            "completed_full_rotation_in_direct_target_mo_basis"
        ],
    }
    if p7.fingerprint(accepted_configuration) != accepted_fingerprint:
        raise RuntimeError(
            f"Accepted Phase-7A.2 configuration provenance is inconsistent "
            f"at R={bond}."
        )

    energies = accepted_audit["energies_ha"]
    accepted_offset = float(energies["qiskit_rotated_total_constant_offset"])
    accepted_hf = float(energies["qiskit_rotated_hf_total"])
    if abs(total_offset - accepted_offset) > audit_tolerance:
        raise RuntimeError(f"Constant offset changed at R={bond}.")
    if abs(hf_total_qubit - accepted_hf) > audit_tolerance:
        raise RuntimeError(f"Qubit HF energy changed at R={bond}.")
    if abs(hf_total_driver - accepted_hf) > audit_tolerance:
        raise RuntimeError(f"Driver HF energy changed at R={bond}.")
    if hermiticity > audit_tolerance:
        raise RuntimeError(f"Qubit Hamiltonian is not Hermitian at R={bond}.")
    if int(qubit_op.num_qubits) != p7a2.EXPECTED_QUBITS:
        raise RuntimeError(f"Qubit count changed at R={bond}.")

    full_pool, _, _ = p7a2.compact_excitations()
    exact_active = float(energies["qiskit_rotated_exact_active_electronic"])
    exact_total = float(energies["qiskit_rotated_exact_total"])
    config = diag.MolecularConfig(
        atom=f"Li 0 0 0; H 0 0 {bond}",
        basis=p7a2.TARGET_BASIS,
        charge=0,
        spin=0,
        unit="ANGSTROM",
        freeze_core=True,
        mapper="JordanWignerMapper",
        excitation_rank=2,
        preserve_spin=True,
    )
    optimization_system = diag.SystemData(
        config=config,
        problem=problem,
        mapper=mapper,
        fermionic_op=fermionic_op,
        qubit_op=qubit_op,
        constant_offsets=constants,
        total_offset_ha=total_offset,
        exact_active_electronic_ha=exact_active,
        exact_total_ha=exact_total,
        hf_active_electronic_ha=hf_active,
        hf_total_from_qubit_ha=hf_total_qubit,
        hf_total_driver_ha=hf_total_driver,
        candidate_excitations=list(full_pool),
        # Preserve the accepted configuration identity.  The current raw
        # MO-gauge representation may have a different hash while describing
        # the same AO-basis physical orbitals.
        fingerprint=accepted_fingerprint,
    )
    return BuiltSystem(
        optimization_system=optimization_system,
        bond_length=bond,
        raw_problem=raw_problem,
        rotated_full_problem=rotated_full_problem,
        driver_coefficients=driver_coefficients,
        direct_to_driver_gauge=gauge,
        full_rotation_driver_mo=driver_rotation,
        gauge_orthogonality_residual=gauge_residual,
        physical_rotated_orbital_residual=physical_residual,
        hermiticity_residual=hermiticity,
        current_gauge_configuration_fingerprint=(
            current_gauge_configuration_fingerprint
        ),
    )


def hf_basis_index(system: diag.SystemData) -> int:
    from qiskit.quantum_info import Statevector
    from qiskit_nature.second_q.circuit.library import HartreeFock

    circuit = HartreeFock(
        system.problem.num_spatial_orbitals,
        system.problem.num_particles,
        system.mapper,
    )
    data = np.asarray(Statevector.from_instruction(circuit).data)
    index = int(np.argmax(np.abs(data)))
    if not np.isclose(abs(data[index]), 1.0, atol=1e-12):
        raise RuntimeError("Hartree--Fock initial state is not a basis determinant.")
    return index


def excited_basis_index(hf_index: int, excitation: diag.Excitation) -> int:
    occupied, virtual = excitation
    result = int(hf_index)
    for orbital in occupied:
        bit = 1 << int(orbital)
        if not result & bit:
            raise RuntimeError(
                f"Excitation removes unoccupied spin orbital {orbital}."
            )
        result &= ~bit
    for orbital in virtual:
        bit = 1 << int(orbital)
        if result & bit:
            raise RuntimeError(
                f"Excitation fills occupied spin orbital {orbital}."
            )
        result |= bit
    return result


def perturbative_operator_rows(
    equilibrium_system: diag.SystemData,
    denominator_floor_ha: float,
) -> list[dict[str, Any]]:
    """Rank the full24 pool without consulting any exact energy/eigenvector."""
    if denominator_floor_ha <= 0:
        raise ValueError("The perturbative denominator floor must be positive.")
    matrix = equilibrium_system.qubit_op.to_matrix(sparse=True).tocsr()
    hf_index = hf_basis_index(equilibrium_system)
    hf_diagonal = float(np.real(matrix[hf_index, hf_index]))
    rows: list[dict[str, Any]] = []
    for index, excitation in enumerate(equilibrium_system.candidate_excitations):
        excited_index = excited_basis_index(hf_index, excitation)
        coupling = complex(matrix[excited_index, hf_index])
        excited_diagonal = float(np.real(matrix[excited_index, excited_index]))
        signed_gap = excited_diagonal - hf_diagonal
        denominator = max(abs(signed_gap), denominator_floor_ha)
        amplitude_score = abs(coupling) / denominator
        lowering_score = abs(coupling) ** 2 / denominator
        rows.append(
            {
                "global_uccsd_index": index,
                "operator_label": p7.operator_label(index),
                "operator_type": "single" if index < 8 else "double",
                "excitation_json": json.dumps(diag.excitation_to_json(excitation)),
                "hf_basis_index": hf_index,
                "excited_basis_index": excited_index,
                "absolute_hamiltonian_coupling_ha": abs(coupling),
                "signed_diagonal_gap_ha": signed_gap,
                "absolute_denominator_ha": denominator,
                "en_amplitude_score": amplitude_score,
                "en_second_order_lowering_score_ha": lowering_score,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["en_amplitude_score"]),
            int(row["global_uccsd_index"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["en_amplitude_rank"] = rank
        row["included_in_perturbative10"] = rank <= EXPECTED_COMPACT_OPERATORS
    return sorted(ranked, key=lambda row: int(row["global_uccsd_index"]))


def generate_random_supports(
    seed: int,
    excluded_supports: set[tuple[int, ...]],
) -> list[Variant]:
    rng = np.random.default_rng(seed)
    variants: list[Variant] = []
    seen = set(excluded_supports)
    attempts = 0
    while len(variants) < EXPECTED_RANDOM_SUPPORTS:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Unable to generate unique random10 supports.")
        singles = rng.choice(8, size=EXPECTED_RANDOM_SINGLES, replace=False)
        doubles = rng.choice(
            np.arange(8, EXPECTED_FULL_OPERATORS),
            size=EXPECTED_RANDOM_DOUBLES,
            replace=False,
        )
        selected = tuple(sorted(int(value) for value in [*singles, *doubles]))
        if selected in seen:
            continue
        seen.add(selected)
        variants.append(
            Variant(
                name=f"random10_{len(variants):02d}",
                selected_ids=selected,
                family="random",
                scientific_role=(
                    "Composition-matched random-support null control "
                    "(four singles and six doubles)."
                ),
            )
        )
    return variants


def build_variants(
    operator_rows: Sequence[dict[str, Any]],
    random_seed: int,
) -> list[Variant]:
    full_ids = tuple(range(EXPECTED_FULL_OPERATORS))
    compact_ids = tuple(int(value) for value in p7.compact_reference_indices())
    perturbative_ids = tuple(
        sorted(
            int(row["global_uccsd_index"])
            for row in operator_rows
            if bool(row["included_in_perturbative10"])
        )
    )
    if len(full_ids) != 24 or len(compact_ids) != 10 or len(perturbative_ids) != 10:
        raise RuntimeError("A deterministic support has the wrong cardinality.")
    deterministic = [
        Variant(
            "full24",
            full_ids,
            "deterministic",
            "Parent full spin-preserving UCCSD support in the rotated 2e,5o model.",
        ),
        Variant(
            "transferred_compact10",
            compact_ids,
            "deterministic",
            "Phase-6 compact support transferred through the audited orbital embedding.",
        ),
        Variant(
            "perturbative10",
            perturbative_ids,
            "deterministic",
            "Equilibrium Hamiltonian-derived Epstein--Nesbet-like amplitude ranking.",
        ),
    ]
    excluded = {compact_ids, perturbative_ids}
    return [*deterministic, *generate_random_supports(random_seed, excluded)]


def support_payload(
    inputs: Phase7BInputs,
    variants: Sequence[Variant],
    operator_rows: Sequence[dict[str, Any]],
    random_seed: int,
    denominator_floor_ha: float,
) -> dict[str, Any]:
    full_pool, compact, regenerated = p7a2.compact_excitations()
    if full_pool != regenerated or len(full_pool) != EXPECTED_FULL_OPERATORS:
        raise RuntimeError("The full24 pool failed independent regeneration.")
    if tuple(p7.compact_reference_indices()) != tuple(
        index for index, item in enumerate(full_pool) if item in set(compact)
    ):
        raise RuntimeError("The compact10 operator identity changed.")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "phase7a2_protocol_fingerprint": inputs.phase7a2_audit[
            "protocol_fingerprint"
        ],
        "phase7a2_rotated_support_fingerprint": inputs.phase7a2_audit[
            "rotated_support_fingerprint"
        ],
        "support_frozen_before_phase7b_vqe": True,
        "exact_energy_or_eigenvector_used_for_support_selection": False,
        "exact_reference_policy": (
            "Accepted Phase-7A.2 exact 2e,5o totals are audit inputs, but the "
            "support-selection code accesses only the HF determinant and the "
            "Hamiltonian matrix; Phase 7B never diagonalizes a Hamiltonian."
        ),
        "geometries_angstrom": list(EXPECTED_BOND_LENGTHS),
        "operator_ordering": "ascending_global_uccsd_index",
        "full_pool_excitations": [
            diag.excitation_to_json(item) for item in full_pool
        ],
        "perturbative_ranking": {
            "reference_geometry_angstrom": REFERENCE_BOND_LENGTH,
            "method": (
                "abs(<D_mu|H|HF>)/max(abs(<D_mu|H|D_mu>-<HF|H|HF>),floor)"
            ),
            "denominator_floor_ha": denominator_floor_ha,
            "uses_exact_energy": False,
            "uses_exact_eigenvector": False,
            "ordinary_mp2_label_rejected": (
                "The rotated active virtuals are noncanonical, so this control "
                "is not labeled MP2-ranked."
            ),
        },
        "random_control": {
            "seed": random_seed,
            "number_of_supports": EXPECTED_RANDOM_SUPPORTS,
            "singles_per_support": EXPECTED_RANDOM_SINGLES,
            "doubles_per_support": EXPECTED_RANDOM_DOUBLES,
            "sampling": "uniform_without_replacement_within_operator_type",
            "duplicate_and_deterministic_supports_rejected": True,
        },
        "variants": [
            {
                "name": variant.name,
                "family": variant.family,
                "scientific_role": variant.scientific_role,
                "selected_global_uccsd_indices": list(variant.selected_ids),
                "operator_labels": [
                    p7.operator_label(index) for index in variant.selected_ids
                ],
                "excitations": [
                    diag.excitation_to_json(full_pool[index])
                    for index in variant.selected_ids
                ],
            }
            for variant in variants
        ],
        "operator_score_rows": list(operator_rows),
        "claim_boundary": (
            "The experiment tests operator-support transfer only inside the "
            "audited rotated LiH/6-31G 2e,5o Hamiltonians. The Phase-7A.2 "
            "2e,10o truncation errors remain external context."
        ),
    }
    payload["phase7b_support_fingerprint"] = p7.fingerprint(payload)
    return payload


def variants_from_frozen_support(
    support: dict[str, Any],
) -> list[Variant]:
    records = support.get("variants")
    if not isinstance(records, list):
        raise RuntimeError("Frozen Phase-7B support lacks its variant records.")
    full_pool, _, _ = p7a2.compact_excitations()
    variants: list[Variant] = []
    for record in records:
        selected = tuple(
            int(value)
            for value in record["selected_global_uccsd_indices"]
        )
        if (
            selected != tuple(sorted(selected))
            or len(selected) != len(set(selected))
            or any(index < 0 or index >= len(full_pool) for index in selected)
        ):
            raise RuntimeError(
                f"Frozen variant {record.get('name')} has invalid operator IDs."
            )
        expected_labels = [p7.operator_label(index) for index in selected]
        if list(record.get("operator_labels", [])) != expected_labels:
            raise RuntimeError(
                f"Frozen variant {record.get('name')} has changed operator labels."
            )
        expected_excitations = [
            diag.excitation_to_json(full_pool[index]) for index in selected
        ]
        if list(record.get("excitations", [])) != expected_excitations:
            raise RuntimeError(
                f"Frozen variant {record.get('name')} has changed excitations."
            )
        variants.append(
            Variant(
                name=str(record["name"]),
                selected_ids=selected,
                family=str(record["family"]),
                scientific_role=str(record["scientific_role"]),
            )
        )

    expected_names = {
        "full24",
        "transferred_compact10",
        "perturbative10",
        *(f"random10_{index:02d}" for index in range(EXPECTED_RANDOM_SUPPORTS)),
    }
    if (
        len(variants) != 3 + EXPECTED_RANDOM_SUPPORTS
        or {variant.name for variant in variants} != expected_names
    ):
        raise RuntimeError("The frozen Phase-7B variant set is incomplete.")
    by_name = {variant.name: variant for variant in variants}
    if by_name["full24"].selected_ids != tuple(range(EXPECTED_FULL_OPERATORS)):
        raise RuntimeError("The frozen full24 support changed.")
    if by_name["transferred_compact10"].selected_ids != tuple(
        p7.compact_reference_indices()
    ):
        raise RuntimeError("The frozen transferred compact10 support changed.")
    if len(by_name["perturbative10"].selected_ids) != EXPECTED_COMPACT_OPERATORS:
        raise RuntimeError("The frozen perturbative10 support changed cardinality.")
    for index in range(EXPECTED_RANDOM_SUPPORTS):
        random_variant = by_name[f"random10_{index:02d}"]
        if (
            len(random_variant.selected_ids) != EXPECTED_COMPACT_OPERATORS
            or sum(value < 8 for value in random_variant.selected_ids)
            != EXPECTED_RANDOM_SINGLES
            or sum(value >= 8 for value in random_variant.selected_ids)
            != EXPECTED_RANDOM_DOUBLES
        ):
            raise RuntimeError(
                f"Frozen random support {random_variant.name} changed composition."
            )
    random_variants = [
        variant for variant in variants if variant.family == "random"
    ]
    if len({variant.selected_ids for variant in random_variants}) != len(
        random_variants
    ):
        raise RuntimeError("The frozen random-control set contains duplicates.")
    deterministic_ten_operator_supports = {
        by_name["transferred_compact10"].selected_ids,
        by_name["perturbative10"].selected_ids,
    }
    if any(
        variant.selected_ids in deterministic_ten_operator_supports
        for variant in random_variants
    ):
        raise RuntimeError(
            "A frozen random control duplicates a deterministic ten-operator "
            "support."
        )
    if (
        by_name["transferred_compact10"].selected_ids
        == by_name["perturbative10"].selected_ids
    ):
        print(
            "NOTE: perturbative10 independently selected the same operator "
            "support as transferred_compact10.",
            flush=True,
        )
    return variants


def load_or_create_frozen_support(
    path: Path,
    inputs: Phase7BInputs,
    equilibrium_system: diag.SystemData,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Variant]]:
    if path.is_file():
        print("Loading the previously frozen Phase-7B supports...", flush=True)
        support = json.loads(path.read_text(encoding="utf-8"))
        validate_signed_payload(support, "phase7b_support_fingerprint")
        if support.get("analysis_phase") != ANALYSIS_PHASE:
            raise RuntimeError("Existing support is not a Phase-7B artifact.")
        if support.get("phase7a2_protocol_fingerprint") != inputs.phase7a2_audit.get(
            "protocol_fingerprint"
        ):
            raise RuntimeError(
                "Frozen Phase-7B support belongs to a different Phase-7A.2 "
                "protocol."
            )
        if support.get(
            "phase7a2_rotated_support_fingerprint"
        ) != inputs.phase7a2_audit.get("rotated_support_fingerprint"):
            raise RuntimeError(
                "Frozen Phase-7B support belongs to different rotated orbitals."
            )
        random_control = support.get("random_control", {})
        if int(random_control.get("seed", -1)) != args.random_support_seed:
            raise RuntimeError(
                "The requested random-support seed differs from the frozen support."
            )
        ranking = support.get("perturbative_ranking", {})
        frozen_floor = float(ranking.get("denominator_floor_ha", np.nan))
        if not np.isclose(
            frozen_floor,
            args.perturbative_denominator_floor,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "The requested perturbative denominator floor differs from "
                "the frozen support."
            )
        operator_rows = list(support.get("operator_score_rows", []))
        if len(operator_rows) != EXPECTED_FULL_OPERATORS:
            raise RuntimeError("Frozen Phase-7B operator scores are incomplete.")
        variants = variants_from_frozen_support(support)
        return support, operator_rows, variants

    operator_rows = perturbative_operator_rows(
        equilibrium_system,
        args.perturbative_denominator_floor,
    )
    variants = build_variants(operator_rows, args.random_support_seed)
    if len(variants) != 3 + EXPECTED_RANDOM_SUPPORTS:
        raise RuntimeError("The frozen variant set is incomplete.")
    support = support_payload(
        inputs,
        variants,
        operator_rows,
        args.random_support_seed,
        args.perturbative_denominator_floor,
    )
    write_or_validate_signed_payload(
        path,
        support,
        "phase7b_support_fingerprint",
    )
    return support, operator_rows, variants


def protocol_payload(
    support: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "phase7b_support_fingerprint": support["phase7b_support_fingerprint"],
        "phase7a2_protocol_fingerprint": support[
            "phase7a2_protocol_fingerprint"
        ],
        "phase7a2_rotated_support_fingerprint": support[
            "phase7a2_rotated_support_fingerprint"
        ],
        "optimizer": args.optimizer,
        "maxiter": args.maxiter,
        "optimizer_tolerance": args.optimizer_tolerance,
        "initial_scale": args.initial_scale,
        "parameter_seed": args.parameter_seed,
        "base_restarts": args.base_restarts,
        "rescue_restarts": args.rescue_restarts,
        "rescue_disagreement_tolerance_ha": args.disagreement_tolerance,
        "rescue_triggers": [
            "incomplete_base_restarts",
            "nonfinite_base_energy",
            "optimizer_failure",
            "variational_violation",
            "energy_disagreement",
            "chemical_classification_disagreement",
        ],
        "analysis_cohort": (
            "rescue_only if rescue is triggered; otherwise base restarts"
        ),
        "energy_thresholds_ha": {
            "strict_full24_equivalence": STRICT_EQUIVALENCE_HA,
            "chemical_accuracy": CHEMICAL_ACCURACY_HA,
            "good_accuracy": GOOD_ACCURACY_HA,
            "acceptable_accuracy": ACCEPTABLE_ACCURACY_HA,
        },
        "strict_equivalence_requires_each_energy_spread_at_most_threshold": True,
        "execution_controls_excluded_from_protocol_identity": [
            "run_group",
            "resume",
            "freeze_only",
            "skip_circuit_resources",
            "output_dir",
        ],
        "primary_reference": "accepted Phase-7A.2 rotated 2e,5o exact total",
        "full_noncore_2e10o_role": "active-space-truncation context only",
        "noiseless_statevector_only": True,
    }
    payload["phase7b_protocol_fingerprint"] = p7.fingerprint(payload)
    return payload


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def rescue_reasons(
    base_runs: pd.DataFrame,
    base_restarts: int,
    disagreement_tolerance_ha: float,
) -> list[str]:
    reasons: list[str] = []
    if len(base_runs) != base_restarts:
        return ["incomplete_base_restarts"]
    energies = finite_series(base_runs, "vqe_total_energy_ha")
    if energies.isna().any():
        reasons.append("nonfinite_base_energy")
    if not base_runs["optimizer_success"].map(as_bool).all():
        reasons.append("optimizer_failure")
    if base_runs["variational_violation"].map(as_bool).any():
        reasons.append("variational_violation")
    if (
        energies.notna().all()
        and float(energies.max() - energies.min()) > disagreement_tolerance_ha
    ):
        reasons.append("energy_disagreement")
    errors = finite_series(base_runs, "absolute_error_ha")
    if errors.notna().all():
        classes = errors <= CHEMICAL_ACCURACY_HA
        if classes.nunique() > 1:
            reasons.append("chemical_classification_disagreement")
    return reasons


def assess_analysis_runs(
    analysis_runs: pd.DataFrame,
    expected_runs: int,
    disagreement_tolerance_ha: float,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if len(analysis_runs) != expected_runs:
        return False, ["incomplete_analysis_restarts"]
    energies = finite_series(analysis_runs, "vqe_total_energy_ha")
    if energies.isna().any():
        warnings.append("nonfinite_energy")
    if not analysis_runs["optimizer_success"].map(as_bool).all():
        warnings.append("optimizer_failure")
    if analysis_runs["variational_violation"].map(as_bool).any():
        warnings.append("variational_violation")
    if (
        energies.notna().all()
        and float(energies.max() - energies.min()) > disagreement_tolerance_ha
    ):
        warnings.append("energy_disagreement")
    errors = finite_series(analysis_runs, "absolute_error_ha")
    if errors.notna().all() and (errors <= CHEMICAL_ACCURACY_HA).nunique() > 1:
        warnings.append("chemical_classification_disagreement")
    return not warnings, warnings


def nullable_resources() -> dict[str, int | None]:
    return {
        "logical_depth": None,
        "logical_size": None,
        "compiled_depth": None,
        "compiled_size": None,
        "compiled_cx": None,
    }


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def normalize_checkpoint_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Restore canonical string identities after a CSV round trip.

    pandas infers values such as ``1.595000`` as floats.  Without explicit
    normalization, a resumed row becomes ``"1.595"`` and no longer matches
    the canonical geometry key ``"1.595000"``.
    """
    required = {
        "protocol_fingerprint",
        "bond_length_angstrom",
        "geometry_key",
        "variant",
        "restart",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "Existing Phase-7B checkpoint is missing columns: "
            + ", ".join(missing)
        )
    result = frame.copy()
    bonds = pd.to_numeric(result["bond_length_angstrom"], errors="coerce")
    saved_keys = pd.to_numeric(result["geometry_key"], errors="coerce")
    if bonds.isna().any() or saved_keys.isna().any():
        raise RuntimeError("Existing checkpoint contains an invalid geometry key.")
    if float(np.max(np.abs(bonds - saved_keys))) > 5e-7:
        raise RuntimeError(
            "Existing checkpoint geometry keys disagree with its bond lengths."
        )
    result["bond_length_angstrom"] = bonds.astype(float)
    result["geometry_key"] = bonds.map(geometry_key)
    result["protocol_fingerprint"] = result[
        "protocol_fingerprint"
    ].astype(str)
    result["variant"] = result["variant"].astype(str)
    restarts = pd.to_numeric(result["restart"], errors="coerce")
    if restarts.isna().any() or not np.allclose(
        restarts,
        np.rint(restarts),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("Existing checkpoint contains an invalid restart ID.")
    result["restart"] = np.rint(restarts).astype(int)
    duplicate_key = [
        "protocol_fingerprint",
        "geometry_key",
        "variant",
        "restart",
    ]
    duplicate_mask = result.duplicated(duplicate_key, keep=False)
    duplicates_removed = 0
    if duplicate_mask.any():
        duplicated = result.loc[duplicate_mask].copy()
        if "vqe_total_energy_ha" in duplicated:
            for identity, group in duplicated.groupby(
                duplicate_key, dropna=False, sort=False
            ):
                energies = pd.to_numeric(
                    group["vqe_total_energy_ha"], errors="coerce"
                ).dropna()
                if (
                    len(energies) > 1
                    and float(energies.max() - energies.min()) > 1e-7
                ):
                    raise RuntimeError(
                        "Repeated checkpoint evaluations disagree by more "
                        f"than 1e-7 Ha for identity {identity}."
                    )
        before = len(result)
        result = result.drop_duplicates(duplicate_key, keep="last").copy()
        duplicates_removed = before - len(result)
    return result, duplicates_removed


def failure_row(base: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **base,
        "optimizer_success": False,
        "optimizer_status": -999,
        "optimizer_message": "exception",
        "num_objective_evaluations": 0,
        "num_iterations": 0,
        "elapsed_seconds": 0.0,
        "vqe_total_energy_ha": np.nan,
        "signed_error_ha": np.nan,
        "absolute_error_ha": np.nan,
        "variational_violation": False,
        "chemical_accuracy": False,
        "good_accuracy": False,
        "acceptable_accuracy": False,
        "optimal_parameters_json": None,
        "run_exception": repr(exc),
    }


def select_variants_for_group(
    variants: Sequence[Variant],
    run_group: str,
) -> list[Variant]:
    if run_group == "all":
        return list(variants)
    if run_group == "deterministic":
        return [item for item in variants if item.family == "deterministic"]
    if run_group == "random":
        return [item for item in variants if item.family == "random"]
    raise ValueError(f"Unknown run group: {run_group}")


def run_experiment(
    systems: dict[str, BuiltSystem],
    variants: Sequence[Variant],
    variants_to_run: Sequence[Variant],
    output_dir: Path,
    protocol_fingerprint: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    runs_path = output_dir / "phase7b_restart_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if runs_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"Checkpoint already exists: {runs_path}. Enable --resume or "
                "use a new output directory."
            )
        previous, duplicates_removed = normalize_checkpoint_frame(
            pd.read_csv(runs_path)
        )
        if duplicates_removed:
            backup_path = runs_path.with_name(
                "phase7b_restart_runs_pre_normalization_backup.csv"
            )
            if backup_path.exists():
                suffix = 1
                while backup_path.with_name(
                    f"phase7b_restart_runs_pre_normalization_backup_{suffix}.csv"
                ).exists():
                    suffix += 1
                backup_path = backup_path.with_name(
                    f"phase7b_restart_runs_pre_normalization_backup_{suffix}.csv"
                )
            shutil.copy2(runs_path, backup_path)
            checkpoint(runs_path, previous.to_dict(orient="records"))
            print(
                f"Normalized checkpoint geometry keys and removed "
                f"{duplicates_removed} repeated run identities; raw backup: "
                f"{backup_path}",
                flush=True,
            )
        fingerprints = set(previous["protocol_fingerprint"].dropna().astype(str))
        if fingerprints and fingerprints != {protocol_fingerprint}:
            raise RuntimeError("Existing checkpoint has a different protocol.")
        rows = previous.to_dict(orient="records")
        completed = {
            (str(row["geometry_key"]), str(row["variant"]), int(row["restart"]))
            for row in rows
        }
        print(f"Resuming with {len(completed)} completed Phase-7B attempts.")

    variant_indices = {item.name: index for index, item in enumerate(variants)}
    resource_cache: dict[tuple[int, ...], dict[str, int | None]] = {}

    def current_runs(key: str, variant_name: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        return frame[
            (frame["geometry_key"].astype(str) == key)
            & (frame["variant"].astype(str) == variant_name)
        ].copy()

    for geometry_index, bond in enumerate(EXPECTED_BOND_LENGTHS):
        key = geometry_key(bond)
        built = systems[key]
        system = built.optimization_system
        for variant in variants_to_run:
            variant_index = variant_indices[variant.name]
            existing = current_runs(key, variant.name)
            if not existing.empty:
                base_existing = existing[
                    pd.to_numeric(existing["restart"], errors="coerce")
                    < args.base_restarts
                ]
                if len(base_existing) == args.base_restarts:
                    reasons = rescue_reasons(
                        base_existing,
                        args.base_restarts,
                        args.disagreement_tolerance,
                    )
                    expected = args.base_restarts + (
                        args.rescue_restarts if reasons else 0
                    )
                    if all(
                        (key, variant.name, restart) in completed
                        for restart in range(expected)
                    ):
                        continue

            excitations = [
                system.candidate_excitations[index]
                for index in variant.selected_ids
            ]
            ansatz = diag.build_selected_ansatz(system, excitations)
            if int(ansatz.num_parameters) != len(variant.selected_ids):
                raise RuntimeError(
                    f"Parameter count changed for {variant.name} at R={bond}."
                )
            if variant.selected_ids not in resource_cache:
                resource_cache[variant.selected_ids] = (
                    nullable_resources()
                    if args.skip_circuit_resources
                    else diag.circuit_resource_metrics(
                        ansatz,
                        args.seed_transpiler + variant_index,
                    )
                )
            resources = resource_cache[variant.selected_ids]
            common = {
                "protocol_fingerprint": protocol_fingerprint,
                "phase7b_support_fingerprint": args.support_fingerprint,
                "configuration_fingerprint": system.fingerprint,
                "geometry_index": geometry_index,
                "bond_length_angstrom": bond,
                "geometry_key": key,
                "variant": variant.name,
                "variant_family": variant.family,
                "scientific_role": variant.scientific_role,
                "selected_count": len(variant.selected_ids),
                "selected_ids_json": json.dumps(list(variant.selected_ids)),
                "operator_labels_json": json.dumps(
                    [p7.operator_label(index) for index in variant.selected_ids]
                ),
                "selected_excitations_json": json.dumps(
                    [diag.excitation_to_json(item) for item in excitations]
                ),
                "num_parameters": int(ansatz.num_parameters),
                "exact_total_energy_ha": system.exact_total_ha,
                "exact_reference_source": "accepted_phase7a2_audit",
                "hf_total_energy_ha": system.hf_total_driver_ha,
                "optimizer": args.optimizer,
                "maxiter": args.maxiter,
                "optimizer_tolerance": args.optimizer_tolerance,
                "initial_scale": args.initial_scale,
                **resources,
            }

            def execute_restart(
                restart: int,
                stage: str,
                reasons: Sequence[str] | None,
            ) -> None:
                completed_key = (key, variant.name, restart)
                if completed_key in completed:
                    return
                restart_seed = (
                    args.parameter_seed
                    + 1_000_003 * geometry_index
                    + 100_003 * variant_index
                    + 10_007 * restart
                )
                rng = np.random.default_rng(restart_seed)
                initial = diag.initial_point_for_restart(
                    restart,
                    ansatz.num_parameters,
                    rng,
                    args.initial_scale,
                )
                base = {
                    **common,
                    "restart": restart,
                    "restart_stage": stage,
                    "initialization": "zeros" if restart == 0 else "uniform_random",
                    "initialization_seed": None if restart == 0 else restart_seed,
                    "rescue_trigger_reasons": (
                        None if not reasons else json.dumps(list(reasons))
                    ),
                    "evaluation_mode": "noiseless_statevector_vqe",
                }
                print(
                    f"R={bond:.3f} variant={variant.name} "
                    f"{stage}_restart={restart + 1} "
                    f"parameters={ansatz.num_parameters}",
                    flush=True,
                )
                try:
                    outcome = diag.run_one_optimization(
                        system,
                        ansatz,
                        initial,
                        args.optimizer,
                        args.maxiter,
                        args.optimizer_tolerance,
                    )
                    row = {**base, **outcome, "run_exception": None}
                except Exception as exc:
                    print(
                        f"ERROR at R={bond}, {variant.name}, restart "
                        f"{restart}: {exc}",
                        file=sys.stderr,
                    )
                    row = failure_row(base, exc)
                rows.append(row)
                completed.add(completed_key)
                checkpoint(runs_path, rows)

            for restart in range(args.base_restarts):
                execute_restart(restart, "base", None)

            base_frame = current_runs(key, variant.name)
            base_frame = base_frame[
                pd.to_numeric(base_frame["restart"], errors="coerce")
                < args.base_restarts
            ].sort_values("restart")
            reasons = rescue_reasons(
                base_frame,
                args.base_restarts,
                args.disagreement_tolerance,
            )
            if reasons:
                print(
                    f"R={bond:.3f} variant={variant.name} rescue: "
                    + ", ".join(reasons),
                    flush=True,
                )
                for restart in range(
                    args.base_restarts,
                    args.base_restarts + args.rescue_restarts,
                ):
                    execute_restart(restart, "rescue", reasons)

    frame = pd.DataFrame(rows)
    return frame[
        frame["protocol_fingerprint"].astype(str) == protocol_fingerprint
    ].copy()


def summarize_runs(
    runs: pd.DataFrame,
    systems: dict[str, BuiltSystem],
    variants: Sequence[Variant],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for geometry_index, bond in enumerate(EXPECTED_BOND_LENGTHS):
        key = geometry_key(bond)
        system = systems[key].optimization_system
        for variant in variants:
            group = runs[
                (runs["geometry_key"].astype(str) == key)
                & (runs["variant"].astype(str) == variant.name)
            ].copy()
            group = group.sort_values("restart") if not group.empty else group
            base = {
                "geometry_index": geometry_index,
                "bond_length_angstrom": bond,
                "geometry_key": key,
                "configuration_fingerprint": system.fingerprint,
                "variant": variant.name,
                "variant_family": variant.family,
                "scientific_role": variant.scientific_role,
                "selected_count": len(variant.selected_ids),
                "selected_ids_json": json.dumps(list(variant.selected_ids)),
                "exact_total_energy_ha": system.exact_total_ha,
                "hf_total_energy_ha": system.hf_total_driver_ha,
                "hf_error_ha": system.hf_total_driver_ha - system.exact_total_ha,
            }
            if group.empty:
                rows.append(
                    {
                        **base,
                        "completed_attempts": 0,
                        "expected_attempts": args.base_restarts,
                        "analysis_restart_scope": "not_run",
                        "analysis_restarts": 0,
                        "rescue_triggered": False,
                        "analysis_complete": False,
                        "analysis_stable": False,
                        "stability_warnings_json": json.dumps(["not_run"]),
                        "classification": "not_run",
                    }
                )
                continue

            rescue = group[group["restart_stage"].astype(str) == "rescue"].copy()
            rescue_triggered = not rescue.empty
            if rescue_triggered:
                analysis_runs = rescue
                expected_analysis = args.rescue_restarts
                analysis_scope = "rescue_only"
            else:
                analysis_runs = group[
                    group["restart_stage"].astype(str) == "base"
                ].copy()
                expected_analysis = args.base_restarts
                analysis_scope = "base"
            stable, warnings = assess_analysis_runs(
                analysis_runs,
                expected_analysis,
                args.disagreement_tolerance,
            )
            expected_attempts = args.base_restarts + (
                args.rescue_restarts if rescue_triggered else 0
            )
            complete = len(group) == expected_attempts
            valid = analysis_runs[
                finite_series(analysis_runs, "absolute_error_ha").notna()
            ].copy()
            errors = finite_series(valid, "absolute_error_ha")
            energies = finite_series(valid, "vqe_total_energy_ha")
            robust_chemical: bool | float = np.nan
            if stable and not errors.empty:
                robust_chemical = bool((errors <= CHEMICAL_ACCURACY_HA).all())
            if not complete:
                classification = "incomplete"
            elif not stable:
                classification = "unresolved_optimizer_behavior"
            elif bool(robust_chemical):
                classification = "stable_chemical_accuracy"
            else:
                classification = "stable_nonchemical"
            rescue_values = group["rescue_trigger_reasons"].dropna().astype(str)
            first = group.iloc[0]
            rows.append(
                {
                    **base,
                    "completed_attempts": int(len(group)),
                    "expected_attempts": expected_attempts,
                    "analysis_restart_scope": analysis_scope,
                    "analysis_restarts": int(len(analysis_runs)),
                    "rescue_triggered": rescue_triggered,
                    "rescue_trigger_reasons": (
                        rescue_values.iloc[0] if not rescue_values.empty else None
                    ),
                    "analysis_complete": complete,
                    "analysis_stable": stable,
                    "stability_warnings_json": json.dumps(warnings),
                    "classification": classification,
                    "best_total_energy_ha": (
                        float(energies.min()) if not energies.empty else np.nan
                    ),
                    "median_total_energy_ha": (
                        float(energies.median()) if not energies.empty else np.nan
                    ),
                    "best_error_exact_ha": (
                        float(errors.min()) if not errors.empty else np.nan
                    ),
                    "median_error_exact_ha": (
                        float(errors.median()) if not errors.empty else np.nan
                    ),
                    "worst_error_exact_ha": (
                        float(errors.max()) if not errors.empty else np.nan
                    ),
                    "energy_spread_ha": (
                        float(energies.max() - energies.min())
                        if not energies.empty
                        else np.nan
                    ),
                    "chemical_success_rate": (
                        float((errors <= CHEMICAL_ACCURACY_HA).mean())
                        if not errors.empty
                        else np.nan
                    ),
                    "robust_chemical_accuracy": robust_chemical,
                    "optimizer_success_rate": (
                        float(analysis_runs["optimizer_success"].map(as_bool).mean())
                        if not analysis_runs.empty
                        else np.nan
                    ),
                    "median_iterations": (
                        float(
                            pd.to_numeric(
                                valid["num_iterations"], errors="coerce"
                            ).median()
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "median_elapsed_seconds": (
                        float(
                            pd.to_numeric(
                                valid["elapsed_seconds"], errors="coerce"
                            ).median()
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "total_elapsed_seconds_all_attempts": float(
                        pd.to_numeric(
                            group["elapsed_seconds"], errors="coerce"
                        ).fillna(0).sum()
                    ),
                    "logical_depth": first.get("logical_depth"),
                    "compiled_depth": first.get("compiled_depth"),
                    "compiled_cx": first.get("compiled_cx"),
                    "num_parameters": int(first["num_parameters"]),
                    "any_variational_violation": bool(
                        group["variational_violation"].map(as_bool).any()
                    ),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["geometry_index", "variant"])
    return add_full24_comparisons(summary)


def add_full24_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for column in (
        "full24_best_total_energy_ha",
        "signed_best_energy_difference_vs_full24_ha",
        "absolute_best_energy_difference_vs_full24_ha",
        "compiled_cx_reduction_vs_full24",
        "num_parameters_reduction_vs_full24",
    ):
        result[column] = np.nan
    result["strict_equivalence_numerically_resolved"] = False
    result["matches_full24_within_1e-6_ha"] = False
    result["transfer_classification"] = "not_run"

    for key, indices in result.groupby("geometry_key").groups.items():
        geometry = result.loc[indices]
        full_rows = geometry[geometry["variant"] == "full24"]
        if full_rows.empty or full_rows.iloc[0]["classification"] == "not_run":
            result.loc[indices, "transfer_classification"] = "full24_not_run"
            continue
        full = full_rows.iloc[0]
        if not as_bool(full["analysis_stable"]):
            result.loc[indices, "transfer_classification"] = "full24_unresolved"
            continue
        full_energy = float(full["best_total_energy_ha"])
        full_spread = float(full["energy_spread_ha"])
        full_chemical = as_bool(full["robust_chemical_accuracy"])
        for index in indices:
            row = result.loc[index]
            result.at[index, "full24_best_total_energy_ha"] = full_energy
            if not as_bool(row["analysis_stable"]):
                result.at[index, "transfer_classification"] = "variant_unresolved"
                continue
            difference = float(row["best_total_energy_ha"]) - full_energy
            absolute = abs(difference)
            row_spread = float(row["energy_spread_ha"])
            precision_resolved = bool(
                max(full_spread, row_spread) <= STRICT_EQUIVALENCE_HA
            )
            strict_match = bool(
                precision_resolved and absolute <= STRICT_EQUIVALENCE_HA
            )
            result.at[index, "signed_best_energy_difference_vs_full24_ha"] = difference
            result.at[index, "absolute_best_energy_difference_vs_full24_ha"] = absolute
            result.at[index, "strict_equivalence_numerically_resolved"] = (
                precision_resolved
            )
            result.at[index, "matches_full24_within_1e-6_ha"] = strict_match
            for resource in ("compiled_cx", "num_parameters"):
                denominator = full.get(resource)
                value = row.get(resource)
                if pd.notna(denominator) and float(denominator) and pd.notna(value):
                    result.at[index, f"{resource}_reduction_vs_full24"] = (
                        float(denominator) - float(value)
                    ) / float(denominator)

            if row["variant"] == "full24":
                classification = (
                    "full24_chemical_accuracy"
                    if full_chemical
                    else "full24_parent_ansatz_failure"
                )
            elif not full_chemical:
                classification = "parent_ansatz_failure"
            elif as_bool(row["robust_chemical_accuracy"]) and strict_match:
                classification = "strict_support_transfer"
            elif as_bool(row["robust_chemical_accuracy"]) and not precision_resolved:
                classification = "chemical_transfer_strict_test_unresolved"
            elif as_bool(row["robust_chemical_accuracy"]):
                classification = "chemical_support_transfer"
            else:
                classification = "support_transfer_failure"
            result.at[index, "transfer_classification"] = classification
    return result


def random_support_summary(
    summary: pd.DataFrame,
    variants: Sequence[Variant],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant.family != "random":
            continue
        group = summary[summary["variant"] == variant.name].sort_values(
            "geometry_index"
        )
        complete = bool(
            len(group) == len(EXPECTED_BOND_LENGTHS)
            and group["analysis_complete"].map(as_bool).all()
        )
        stable = bool(complete and group["analysis_stable"].map(as_bool).all())
        valid = group[group["analysis_stable"].map(as_bool)]
        rows.append(
            {
                "variant": variant.name,
                "selected_ids_json": json.dumps(list(variant.selected_ids)),
                "analysis_complete": complete,
                "all_geometries_stable": stable,
                "all_geometries_chemical_accuracy": bool(
                    stable and valid["robust_chemical_accuracy"].map(as_bool).all()
                ),
                "all_geometries_strict_full24_match": bool(
                    stable
                    and valid["matches_full24_within_1e-6_ha"].map(as_bool).all()
                ),
                "worst_best_error_exact_ha": (
                    float(valid["best_error_exact_ha"].max())
                    if not valid.empty
                    else np.nan
                ),
                "worst_absolute_best_energy_difference_vs_full24_ha": (
                    float(
                        valid[
                            "absolute_best_energy_difference_vs_full24_ha"
                        ].max()
                    )
                    if not valid.empty
                    else np.nan
                ),
            }
        )
    return rows


def make_conclusions(
    protocol: dict[str, Any],
    inputs: Phase7BInputs,
    summary: pd.DataFrame,
    random_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = len(EXPECTED_BOND_LENGTHS) * (
        3 + EXPECTED_RANDOM_SUPPORTS
    )
    execution_complete = bool(
        len(summary) == expected_rows
        and summary["analysis_complete"].map(as_bool).all()
    )
    optimizer_resolved = bool(
        execution_complete and summary["analysis_stable"].map(as_bool).all()
    )

    def variant_result(name: str) -> dict[str, Any]:
        group = summary[summary["variant"] == name].sort_values("geometry_index")
        stable = group[group["analysis_stable"].map(as_bool)]
        return {
            "num_geometries_complete": int(
                group["analysis_complete"].map(as_bool).sum()
            ),
            "num_geometries_stable": int(len(stable)),
            "all_geometries_chemical_accuracy": bool(
                len(stable) == len(EXPECTED_BOND_LENGTHS)
                and stable["robust_chemical_accuracy"].map(as_bool).all()
            ),
            "all_geometries_strict_full24_match": bool(
                len(stable) == len(EXPECTED_BOND_LENGTHS)
                and stable["matches_full24_within_1e-6_ha"].map(as_bool).all()
            ),
            "worst_best_error_exact_ha": (
                float(stable["best_error_exact_ha"].max())
                if not stable.empty
                else None
            ),
            "geometry_classifications": {
                str(row["geometry_key"]): row["transfer_classification"]
                for _, row in group.iterrows()
            },
        }

    full = variant_result("full24")
    compact = variant_result("transferred_compact10")
    perturbative = variant_result("perturbative10")
    compact_support_rows = summary[
        summary["variant"] == "transferred_compact10"
    ]
    perturbative_support_rows = summary[summary["variant"] == "perturbative10"]
    compact_support_ids = (
        json.loads(str(compact_support_rows.iloc[0]["selected_ids_json"]))
        if not compact_support_rows.empty
        else []
    )
    perturbative_support_ids = (
        json.loads(str(perturbative_support_rows.iloc[0]["selected_ids_json"]))
        if not perturbative_support_rows.empty
        else []
    )
    compact_equals_perturbative = bool(
        compact_support_ids
        and compact_support_ids == perturbative_support_ids
    )
    random_complete = bool(
        len(random_rows) == EXPECTED_RANDOM_SUPPORTS
        and all(as_bool(row["analysis_complete"]) for row in random_rows)
    )
    random_stable_rows = [
        row for row in random_rows if as_bool(row["all_geometries_stable"])
    ]
    compact_worst = compact["worst_best_error_exact_ha"]
    compact_beats = None
    compact_percentile = None
    if compact_worst is not None and random_stable_rows:
        compact_beats = int(
            sum(
                float(row["worst_best_error_exact_ha"]) > float(compact_worst)
                for row in random_stable_rows
            )
        )
        compact_percentile = compact_beats / len(random_stable_rows)

    if not execution_complete:
        status = "partial_protocol_execution"
        primary = "incomplete"
    elif not optimizer_resolved:
        status = "optimizer_unresolved"
        primary = "unresolved_optimizer_behavior"
    elif not full["all_geometries_chemical_accuracy"]:
        status = "complete_parent_ansatz_failure"
        primary = "parent_ansatz_failure"
    elif compact["all_geometries_strict_full24_match"]:
        status = "complete_strict_support_transfer"
        primary = "strict_support_transfer"
    elif compact["all_geometries_chemical_accuracy"]:
        status = "complete_chemical_support_transfer"
        primary = "chemical_support_transfer"
    else:
        status = "complete_support_transfer_failure"
        primary = "support_transfer_failure"

    truncation_context = {
        key: {
            "rotated_2e5o_exact_total_ha": item["energies_ha"][
                "qiskit_rotated_exact_total"
            ],
            "full_noncore_2e10o_casci_total_ha": item["energies_ha"][
                "pyscf_full_noncore_2e10o_casci_total_ha"
            ],
            "rotated_2e5o_error_vs_full_noncore_ha": item["energies_ha"][
                "rotated_2e5o_error_vs_full_noncore_ha"
            ],
        }
        for key, item in sorted(inputs.audits_by_geometry.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "analysis_status": status,
        "primary_transfer_classification": primary,
        "protocol_fingerprint": protocol["phase7b_protocol_fingerprint"],
        "support_fingerprint": protocol["phase7b_support_fingerprint"],
        "phase7a2_protocol_fingerprint": inputs.phase7a2_audit[
            "protocol_fingerprint"
        ],
        "phase7a2_rotated_support_fingerprint": inputs.phase7a2_audit[
            "rotated_support_fingerprint"
        ],
        "execution_complete": execution_complete,
        "all_optimizers_resolved": optimizer_resolved,
        "exact_reference_recomputed_in_phase7b": False,
        "full24": full,
        "transferred_compact10": compact,
        "perturbative10": perturbative,
        "deterministic_support_identity": {
            "transferred_compact10_global_uccsd_indices": compact_support_ids,
            "perturbative10_global_uccsd_indices": perturbative_support_ids,
            "compact10_equals_perturbative10": compact_equals_perturbative,
            "interpretation": (
                "The independently defined perturbative ranking selected the "
                "same ten-operator support as the transferred compact rule. "
                "Their separate VQE rows test optimizer repeatability, not "
                "distinct ansatz expressivity."
                if compact_equals_perturbative
                else "The transferred and perturbatively ranked supports are "
                "distinct ansatz controls."
            ),
        },
        "random_control": {
            "execution_complete": random_complete,
            "num_supports_stable_all_geometries": len(random_stable_rows),
            "num_supports_chemical_all_geometries": int(
                sum(
                    as_bool(row["all_geometries_chemical_accuracy"])
                    for row in random_stable_rows
                )
            ),
            "num_supports_strict_all_geometries": int(
                sum(
                    as_bool(row["all_geometries_strict_full24_match"])
                    for row in random_stable_rows
                )
            ),
            "compact_beats_random_count_on_worst_geometry_error": compact_beats,
            "compact_empirical_percentile_among_stable_random_supports": (
                compact_percentile
            ),
            "interpretation_limit": (
                "Twenty fixed random supports are a descriptive null control, "
                "not a high-power inferential study."
            ),
        },
        "active_space_truncation_context": truncation_context,
        "claim_boundary": (
            "A positive result supports transfer of the compact operator support "
            "within the audited rotated LiH/6-31G 2e,5o model only. It does not "
            "show chemical accuracy against the full non-core 2e,10o model, "
            "another molecule, another active space, or noisy hardware."
        ),
    }


def fmt(value: Any, digits: int = 8) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    conclusions: dict[str, Any],
    summary: pd.DataFrame,
) -> None:
    deterministic = summary[
        summary["variant"].isin(
            ["full24", "transferred_compact10", "perturbative10"]
        )
    ].sort_values(["geometry_index", "variant"])
    lines = [
        "# LiH Phase 7B Cross-Basis Support Transfer",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        f"Primary classification: **{conclusions['primary_transfer_classification']}**",
        "",
        "## Deterministic comparison",
        "",
        "| R (Å) | Variant | Best error vs rotated exact (Ha) | "
        "Best-energy difference vs full24 (Ha) | Robust chemical | Classification |",
        "|---:|---|---:|---:|:---:|---|",
    ]
    for _, row in deterministic.iterrows():
        lines.append(
            f"| {float(row['bond_length_angstrom']):.3f} | {row['variant']} | "
            f"{fmt(row.get('best_error_exact_ha'))} | "
            f"{fmt(row.get('signed_best_energy_difference_vs_full24_ha'))} | "
            f"{row.get('robust_chemical_accuracy', 'n/a')} | "
            f"{row.get('transfer_classification')} |"
        )
    identity = conclusions["deterministic_support_identity"]
    lines.extend(
        [
            "",
            "## Deterministic support identity",
            "",
            identity["interpretation"],
        ]
    )
    random = conclusions["random_control"]
    lines.extend(
        [
            "",
            "## Random-support control",
            "",
            f"Stable across all geometries: "
            f"{random['num_supports_stable_all_geometries']}/{EXPECTED_RANDOM_SUPPORTS}.",
            "",
            f"Chemical across all geometries: "
            f"{random['num_supports_chemical_all_geometries']}/{EXPECTED_RANDOM_SUPPORTS}.",
            "",
            f"Compact empirical percentile: "
            f"{fmt(random['compact_empirical_percentile_among_stable_random_supports'], 3)}.",
            "",
            random["interpretation_limit"],
            "",
            "## Reference boundary",
            "",
            "Phase 7B reads the accepted Phase-7A.2 rotated 2e,5o exact values; "
            "it does not run an eigensolver. The 2e,10o values are reported only "
            "as active-space-truncation context.",
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
            "Test the frozen Phase-6 compact10 support in the accepted "
            "Phase-7A.2 rotated LiH/6-31G 2e,5o Hamiltonians."
        )
    )
    parser.add_argument(
        "--phase7a-dir", type=Path, default=Path("lih_phase7a_audit")
    )
    parser.add_argument(
        "--phase7a1-dir",
        type=Path,
        default=Path("lih_phase7a1_embedding_diagnostic"),
    )
    parser.add_argument(
        "--phase7a2-dir",
        type=Path,
        default=Path("lih_phase7a2_hamiltonian_audit"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("lih_phase7b_support_transfer")
    )
    parser.add_argument(
        "--run-group",
        choices=["deterministic", "random", "all"],
        default="deterministic",
        help=(
            "Execution staging only; all supports are frozen regardless. "
            "Default: deterministic."
        ),
    )
    parser.add_argument("--optimizer", choices=["SLSQP", "COBYLA", "L-BFGS-B"], default="SLSQP")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-9)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--parameter-seed", type=int, default=20260810)
    parser.add_argument("--random-support-seed", type=int, default=271828)
    parser.add_argument("--base-restarts", type=int, default=2)
    parser.add_argument("--rescue-restarts", type=int, default=3)
    parser.add_argument("--disagreement-tolerance", type=float, default=1e-5)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--rotation-reconstruction-tolerance", type=float, default=1e-10
    )
    parser.add_argument("--perturbative-denominator-floor", type=float, default=1e-8)
    parser.add_argument("--seed-transpiler", type=int, default=20260810)
    parser.add_argument(
        "--skip-circuit-resources",
        action="store_true",
        help="Skip transpilation; energy comparisons remain unchanged.",
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Audit Hamiltonians and freeze all supports, then stop before VQE.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume matching checkpoints (default: true).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.base_restarts < 2:
        raise ValueError("At least two base restarts are required.")
    if args.rescue_restarts < 1:
        raise ValueError("At least one rescue restart is required.")
    if args.maxiter < 1:
        raise ValueError("maxiter must be positive.")
    for name in (
        "optimizer_tolerance",
        "initial_scale",
        "disagreement_tolerance",
        "audit_tolerance",
        "rotation_reconstruction_tolerance",
        "perturbative_denominator_floor",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    diag.require_quantum_stack()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_phase7a2_inputs(args.phase7a2_dir)
    bundle = p7a2.load_inputs(args.phase7a_dir, args.phase7a1_dir)
    print("Reconstructing the accepted rotated orbitals...", flush=True)
    frozen, reconstruction_rows = p7a2.reconstruct_frozen_geometries(
        bundle,
        args.rotation_reconstruction_tolerance,
    )
    accepted_reconstruction_rows = reconcile_with_accepted_physical_orbitals(
        frozen,
        inputs.phase7a2_frozen_support,
        args.rotation_reconstruction_tolerance,
    )
    accepted_by_key = {
        str(row["geometry_key"]): row for row in accepted_reconstruction_rows
    }
    reconstruction_rows = [
        {**row, **accepted_by_key[str(row["geometry_key"])]}
        for row in reconstruction_rows
    ]
    write_csv(
        args.output_dir / "phase7b_rotation_reconstruction.csv",
        reconstruction_rows,
    )

    print("Rebuilding audited Hamiltonians without diagonalization...", flush=True)
    systems: dict[str, BuiltSystem] = {}
    construction_rows: list[dict[str, Any]] = []
    for key in sorted(frozen, key=float):
        accepted = inputs.audits_by_geometry[key]
        built = build_optimization_system_without_diagonalization(
            frozen[key],
            accepted,
            inputs.phase7a2_frozen_support["geometry_rotations"][key],
            args.audit_tolerance,
        )
        systems[key] = built
        system = built.optimization_system
        construction_rows.append(
            {
                "bond_length_angstrom": built.bond_length,
                "geometry_key": key,
                "configuration_fingerprint": system.fingerprint,
                "current_gauge_configuration_fingerprint": (
                    built.current_gauge_configuration_fingerprint
                ),
                "raw_mo_gauge_fingerprint_equality_required": False,
                "accepted_exact_total_energy_ha": system.exact_total_ha,
                "exact_reference_source": "accepted_phase7a2_audit",
                "exact_diagonalization_performed_in_phase7b": False,
                "hf_total_energy_ha": system.hf_total_driver_ha,
                "total_constant_offset_ha": system.total_offset_ha,
                "gauge_orthogonality_residual": (
                    built.gauge_orthogonality_residual
                ),
                "physical_rotated_orbital_residual": (
                    built.physical_rotated_orbital_residual
                ),
                "hamiltonian_hermiticity_residual": built.hermiticity_residual,
            }
        )
    write_csv(
        args.output_dir / "phase7b_hamiltonian_reconstruction.csv",
        construction_rows,
    )

    equilibrium = systems[
        geometry_key(REFERENCE_BOND_LENGTH)
    ].optimization_system
    support_path = args.output_dir / "phase7b_frozen_supports.json"
    supports, operator_rows, variants = load_or_create_frozen_support(
        support_path,
        inputs,
        equilibrium,
        args,
    )
    write_csv(args.output_dir / "phase7b_operator_scores.csv", operator_rows)

    protocol = protocol_payload(supports, args)
    protocol_path = args.output_dir / "phase7b_preregistered_protocol.json"
    write_or_validate_signed_payload(
        protocol_path,
        protocol,
        "phase7b_protocol_fingerprint",
    )
    args.support_fingerprint = supports["phase7b_support_fingerprint"]
    print(
        "All Phase-7B supports frozen before VQE: "
        f"{args.support_fingerprint}",
        flush=True,
    )
    print(
        "Phase-7B protocol fingerprint: "
        f"{protocol['phase7b_protocol_fingerprint']}",
        flush=True,
    )
    if args.freeze_only:
        print("Freeze-only gate complete; no VQE performed.")
        return 0

    variants_to_run = select_variants_for_group(variants, args.run_group)
    runs = run_experiment(
        systems,
        variants,
        variants_to_run,
        args.output_dir,
        protocol["phase7b_protocol_fingerprint"],
        args,
    )
    summary = summarize_runs(runs, systems, variants, args)
    summary.to_csv(
        args.output_dir / "phase7b_variant_geometry_summary.csv", index=False
    )
    random_rows = random_support_summary(summary, variants)
    write_csv(args.output_dir / "phase7b_random_support_summary.csv", random_rows)
    conclusions = make_conclusions(protocol, inputs, summary, random_rows)
    write_json(
        args.output_dir / "phase7b_transfer_conclusions.json", conclusions
    )
    write_report(args.output_dir / "PHASE7B_REPORT.md", conclusions, summary)

    deterministic = summary[
        summary["variant"].isin(
            ["full24", "transferred_compact10", "perturbative10"]
        )
    ]
    print("\nPhase 7B deterministic summary:")
    print(
        deterministic[
            [
                "bond_length_angstrom",
                "variant",
                "analysis_stable",
                "best_error_exact_ha",
                "absolute_best_energy_difference_vs_full24_ha",
                "robust_chemical_accuracy",
                "transfer_classification",
            ]
        ].to_string(index=False)
    )
    print("\nPhase 7B conclusions:")
    print(
        json.dumps(
            {
                "analysis_status": conclusions["analysis_status"],
                "primary_transfer_classification": conclusions[
                    "primary_transfer_classification"
                ],
                "execution_complete": conclusions["execution_complete"],
                "all_optimizers_resolved": conclusions[
                    "all_optimizers_resolved"
                ],
                "protocol_fingerprint": conclusions["protocol_fingerprint"],
                "support_fingerprint": conclusions["support_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if conclusions["analysis_status"] != "optimizer_unresolved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
