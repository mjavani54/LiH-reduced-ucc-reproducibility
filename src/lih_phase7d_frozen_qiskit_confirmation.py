#!/usr/bin/env python3
"""Phase 7D: independently confirm the frozen Phase-7C LiH support.

Phase 7C discovered a 47-operator, spin-complement-complete support in the
frozen-core LiH/6-31G 2e,10o problem.  This program consumes that signed
discovery package, reconstructs the accepted physical orbitals, independently
builds the expanded Hamiltonian through Qiskit Nature, freezes all confirmation
variants, and only then evaluates exact and variational quantities.

The VQE simulator operates in the exact Nalpha=1,Nbeta=1 sector, but every
Hamiltonian and UCC generator is produced by Qiskit Nature and projected from
its Jordan--Wigner Pauli representation.  A full 20-qubit Qiskit Statevector
calculation at the equilibrium candidate optimum spot-checks the projection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import lih_phase7a_cross_basis_transfer as p7
    import lih_phase7a2_rotated_hamiltonian_audit as p7a2
    import lih_phase7b_cross_basis_support_transfer as p7b
    import lih_phase7c_expanded_space_augmentation as p7c
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place this script beside the Phase-7A through Phase-7C scripts and "
        "the Phase-3, Phase-2B, and reference-diagnostic modules."
    ) from exc


SCHEMA_VERSION = 1
ANALYSIS_PHASE = "Phase 7D frozen Qiskit confirmation"
EXPECTED_PHASE7C_STATUS = "complete_candidate_frozen_for_phase7d_confirmation"
EXPECTED_PHASE7C_PROTOCOL_FINGERPRINT = (
    "23e67a5e7699394b0da8b00100edf65822400169421eee30b017be31dbf4d33f"
)
EXPECTED_PHASE7C_SUPPORT_FINGERPRINT = (
    "86bec8f309b3945e5bbb0e7bb4776a3411ed692f52cc49798be4de74576e0026"
)
EXPECTED_BOND_LENGTHS = (1.595, 2.5, 3.0)
REFERENCE_BOND_LENGTH = 1.595

CHEMICAL_ACCURACY_HA = 0.0016
GUARD_BAND_HA = 0.0008
STRICT_EQUIVALENCE_HA = 1e-6
SPIN_SQUARE_TOLERANCE = 1e-7
STATEVECTOR_BASIS_GATES = ("u", "cx")
STATEVECTOR_ALLOWED_OPERATIONS = frozenset((*STATEVECTOR_BASIS_GATES, "barrier"))

EXPANDED_ORBITALS = 10
EXPANDED_QUBITS = 20
SECTOR_DIMENSION = 100
FULL_POOL_SIZE = 99
COMPACT_SIZE = 10
BOUNDARY_SIZE = 46
CANDIDATE_SIZE = 47
BOUNDARY_EXTERNAL_K = 36
CANDIDATE_EXTERNAL_K = 37
DECISIVE_ALPHA_ID = 8
DECISIVE_BETA_ID = 17


Excitation = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Variant:
    name: str
    selected_ids: tuple[int, ...]
    scientific_role: str


@dataclass
class Phase7DInputs:
    support: dict[str, Any]
    protocol: dict[str, Any]
    conclusions: dict[str, Any]
    deterministic: pd.DataFrame
    exact_audit: pd.DataFrame
    full99: pd.DataFrame
    random_summary: pd.DataFrame


@dataclass
class QiskitExpandedSystem:
    geometry_key: str
    bond_length: float
    problem: Any
    mapper: Any
    qubit_op: Any
    spin_square_qubit_op: Any
    total_offset_ha: float
    hf_total_ha: float
    qiskit_sector_total: np.ndarray
    spin_square_sector: np.ndarray
    pyscf_sector_total: np.ndarray
    accepted_exact_total_ha: float
    accepted_internal_exact_total_ha: float
    gauge_orthogonality_residual: float
    physical_orbital_residual: float
    qiskit_hermiticity_residual: float
    pyscf_hermiticity_residual: float
    qiskit_pyscf_gauge_aligned_residual: float


@dataclass(frozen=True)
class SectorGenerator:
    global_id: int
    pairs: tuple[tuple[int, int, complex, float], ...]


@dataclass
class QiskitAnsatzData:
    variant: Variant
    ansatz: Any
    generators: list[SectorGenerator]


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
    if isinstance(value, (np.complexfloating, complex)):
        item = complex(value)
        return {"real": float(item.real), "imag": float(item.imag)}
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
) -> dict[str, Any]:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_signed_payload(existing, fingerprint_field)
        if existing.get(fingerprint_field) != payload.get(fingerprint_field):
            raise RuntimeError(
                f"Existing preregistration {path} has a different fingerprint. "
                "Use a new output directory; do not overwrite it."
            )
        return existing
    write_json(path, payload)
    return payload


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_inputs(phase7c_dir: Path) -> Phase7DInputs:
    paths = {
        "support": phase7c_dir / "phase7c_frozen_discovery_support.json",
        "protocol": phase7c_dir / "phase7c_preregistered_protocol.json",
        "conclusions": phase7c_dir / "phase7c_augmentation_conclusions.json",
        "deterministic": phase7c_dir / "phase7c_deterministic_path_summary.csv",
        "exact": phase7c_dir / "phase7c_exact_hamiltonian_audit.csv",
        "full99": phase7c_dir / "phase7c_full99_control_summary.csv",
        "random": phase7c_dir / "phase7c_random_support_summary.csv",
    }
    for path in paths.values():
        require_file(path)
    support = json.loads(paths["support"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    conclusions = json.loads(paths["conclusions"].read_text(encoding="utf-8"))
    validate_signed_payload(support, "phase7c_support_fingerprint")
    validate_signed_payload(protocol, "phase7c_protocol_fingerprint")

    if support["phase7c_support_fingerprint"] != EXPECTED_PHASE7C_SUPPORT_FINGERPRINT:
        raise RuntimeError("The supplied Phase-7C support is not the accepted package.")
    if protocol["phase7c_protocol_fingerprint"] != EXPECTED_PHASE7C_PROTOCOL_FINGERPRINT:
        raise RuntimeError("The supplied Phase-7C protocol is not the accepted package.")
    if protocol.get("phase7c_support_fingerprint") != support.get(
        "phase7c_support_fingerprint"
    ):
        raise RuntimeError("Phase-7C support and protocol fingerprints disagree.")
    if conclusions.get("analysis_status") != EXPECTED_PHASE7C_STATUS:
        raise RuntimeError("Phase 7C did not authorize frozen confirmation.")
    if not bool(conclusions.get("execution_complete", False)):
        raise RuntimeError("Phase-7C execution_complete is not true.")
    if not bool(conclusions.get("exact_hamiltonian_audit_passed", False)):
        raise RuntimeError("Phase-7C exact Hamiltonian audit did not pass.")
    if conclusions.get("phase7c_protocol_fingerprint") != (
        protocol["phase7c_protocol_fingerprint"]
    ):
        raise RuntimeError("Phase-7C conclusions and protocol disagree.")
    if conclusions.get("phase7c_support_fingerprint") != (
        support["phase7c_support_fingerprint"]
    ):
        raise RuntimeError("Phase-7C conclusions and support disagree.")
    if int(conclusions.get("primary_minimal_chemical_external_k", -1)) != (
        BOUNDARY_EXTERNAL_K
    ):
        raise RuntimeError("The accepted Phase-7C chemical boundary changed.")
    if int(conclusions.get("phase7d_confirmation_external_k", -1)) != (
        CANDIDATE_EXTERNAL_K
    ):
        raise RuntimeError("The accepted Phase-7C confirmation k changed.")
    if int(conclusions.get("phase7d_confirmation_total_operator_count", -1)) != (
        CANDIDATE_SIZE
    ):
        raise RuntimeError("The accepted Phase-7C candidate cardinality changed.")

    deterministic = pd.read_csv(paths["deterministic"], dtype={"geometry_key": str})
    exact = pd.read_csv(paths["exact"], dtype={"geometry_key": str})
    full99 = pd.read_csv(paths["full99"], dtype={"geometry_key": str})
    random = pd.read_csv(paths["random"])
    expected_geometries = {geometry_key(value) for value in EXPECTED_BOND_LENGTHS}
    for frame, name in ((deterministic, "deterministic"), (exact, "exact"), (full99, "full99")):
        if set(frame["geometry_key"].astype(str)) != expected_geometries:
            raise RuntimeError(f"Phase-7C {name} geometries are incomplete.")
    if not all(exact["exact_hamiltonian_audit_passed"].map(as_bool)):
        raise RuntimeError("At least one Phase-7C exact audit row failed.")
    if not all(full99["analysis_stable"].map(as_bool)):
        raise RuntimeError("At least one Phase-7C full99 row is unstable.")
    if len(random) != 20 or not all(random["all_geometries_stable"].map(as_bool)):
        raise RuntimeError("The Phase-7C random-control cohort is incomplete.")
    if any(random["all_geometries_chemical"].map(as_bool)):
        raise RuntimeError("A Phase-7C random control unexpectedly became chemical.")
    for external_k in (BOUNDARY_EXTERNAL_K, CANDIDATE_EXTERNAL_K):
        cohort = deterministic[deterministic["external_k"] == external_k]
        if len(cohort) != len(EXPECTED_BOND_LENGTHS) or not all(
            cohort["analysis_stable"].map(as_bool)
        ):
            raise RuntimeError(f"Phase-7C k={external_k} is incomplete or unstable.")
    return Phase7DInputs(
        support, protocol, conclusions, deterministic, exact, full99, random
    )


def excitation_from_json(value: Sequence[Sequence[int]]) -> Excitation:
    if len(value) != 2:
        raise ValueError("An excitation JSON record must contain two lists.")
    return tuple(int(item) for item in value[0]), tuple(
        int(item) for item in value[1]
    )


def frozen_pool(inputs: Phase7DInputs) -> list[Excitation]:
    records = inputs.support.get("expanded_pool", [])
    records = sorted(records, key=lambda item: int(item["global_uccsd_index"]))
    if [int(item["global_uccsd_index"]) for item in records] != list(
        range(FULL_POOL_SIZE)
    ):
        raise RuntimeError("The frozen Phase-7C expanded pool is incomplete.")
    result = [excitation_from_json(item["excitation"]) for item in records]
    if result != p7c.expanded_pool():
        raise RuntimeError("The frozen Phase-7C expanded excitation pool changed.")
    return result


def spin_partner_id(pool: Sequence[Excitation], global_id: int) -> int:
    index = {excitation: position for position, excitation in enumerate(pool)}
    occupied, virtual = pool[global_id]
    if len(occupied) == 1:
        partner_occupied = (
            occupied[0] + EXPANDED_ORBITALS
            if occupied[0] < EXPANDED_ORBITALS
            else occupied[0] - EXPANDED_ORBITALS,
        )
        partner_virtual = (
            virtual[0] + EXPANDED_ORBITALS
            if virtual[0] < EXPANDED_ORBITALS
            else virtual[0] - EXPANDED_ORBITALS,
        )
    else:
        partner_occupied = occupied
        partner_virtual = (
            virtual[1] - EXPANDED_ORBITALS,
            virtual[0] + EXPANDED_ORBITALS,
        )
    try:
        return index[(partner_occupied, partner_virtual)]
    except KeyError as exc:
        raise RuntimeError(f"No spin partner exists for operator E{global_id}.") from exc


def spin_missing_partners(
    pool: Sequence[Excitation], selected_ids: Sequence[int]
) -> list[tuple[int, int]]:
    selected = set(int(value) for value in selected_ids)
    missing = []
    for global_id in sorted(selected):
        partner = spin_partner_id(pool, global_id)
        if partner not in selected:
            missing.append((global_id, partner))
    return missing


def build_variants(
    inputs: Phase7DInputs,
    pool: Sequence[Excitation],
) -> list[Variant]:
    compact = tuple(
        sorted(int(value) for value in inputs.support["transferred_compact10_global_indices"])
    )
    ranking = tuple(int(value) for value in inputs.support["external_ranking_global_indices"])
    boundary = tuple(sorted((*compact, *ranking[:BOUNDARY_EXTERNAL_K])))
    candidate = tuple(sorted((*compact, *ranking[:CANDIDATE_EXTERNAL_K])))
    accepted_candidate = tuple(
        int(value)
        for value in inputs.conclusions["phase7d_confirmation_global_uccsd_indices"]
    )
    if len(compact) != COMPACT_SIZE or len(boundary) != BOUNDARY_SIZE:
        raise RuntimeError("A frozen Phase-7D control has the wrong cardinality.")
    if len(candidate) != CANDIDATE_SIZE or candidate != accepted_candidate:
        raise RuntimeError("The reconstructed Phase-7D candidate changed.")
    if ranking[BOUNDARY_EXTERNAL_K - 1] != DECISIVE_ALPHA_ID:
        raise RuntimeError("The Phase-7C boundary no longer ends with E8.")
    if ranking[CANDIDATE_EXTERNAL_K - 1] != DECISIVE_BETA_ID:
        raise RuntimeError("The Phase-7C confirmation candidate no longer adds E17.")
    if spin_partner_id(pool, DECISIVE_ALPHA_ID) != DECISIVE_BETA_ID:
        raise RuntimeError("The decisive alpha/beta operator pairing changed.")
    if spin_missing_partners(pool, candidate):
        raise RuntimeError("The 47-operator confirmation candidate is not spin complete.")
    if spin_missing_partners(pool, compact):
        raise RuntimeError("The transferred compact10 support is not spin complete.")
    if spin_missing_partners(pool, tuple(range(FULL_POOL_SIZE))):
        raise RuntimeError("The full99 pool is not spin complete.")
    if spin_missing_partners(pool, boundary) != [(DECISIVE_ALPHA_ID, DECISIVE_BETA_ID)]:
        raise RuntimeError("The 46-operator boundary has an unexpected spin defect.")
    return [
        Variant(
            "compact10",
            compact,
            "Transferred internal control; expected to retain the 2e,5o truncation error.",
        ),
        Variant(
            "chemical_boundary46",
            boundary,
            "Phase-7C minimal chemical boundary, missing beta partner E17 of alpha E8.",
        ),
        Variant(
            "spin_complete_candidate47",
            candidate,
            "Frozen primary confirmation candidate; E17 completes the E8/E17 spin pair.",
        ),
        Variant(
            "full99",
            tuple(range(FULL_POOL_SIZE)),
            "Full spin-preserving UCCSD variational control in the expanded space.",
        ),
    ]


def determinant_bit_indices() -> tuple[int, ...]:
    return tuple(
        (1 << alpha) | (1 << (EXPANDED_ORBITALS + beta))
        for alpha in range(EXPANDED_ORBITALS)
        for beta in range(EXPANDED_ORBITALS)
    )


def pauli_masks(label: str) -> tuple[int, int, int]:
    if len(label) != EXPANDED_QUBITS:
        raise RuntimeError("A projected Pauli label has the wrong register length.")
    x_mask = 0
    z_mask = 0
    y_count = 0
    for qubit, symbol in enumerate(reversed(label)):
        if symbol in {"X", "Y"}:
            x_mask |= 1 << qubit
        if symbol in {"Z", "Y"}:
            z_mask |= 1 << qubit
        if symbol == "Y":
            y_count += 1
        if symbol not in {"I", "X", "Y", "Z"}:
            raise RuntimeError(f"Unsupported Pauli symbol: {symbol}")
    return x_mask, z_mask, y_count


def project_pauli_operator(operator: Any) -> np.ndarray:
    primitive = getattr(operator, "primitive", operator)
    if not hasattr(primitive, "paulis") or not hasattr(primitive, "coeffs"):
        raise TypeError("Expected a SparsePauliOp-compatible Qiskit operator.")
    bits = determinant_bit_indices()
    address = {bit: index for index, bit in enumerate(bits)}
    result = np.zeros((SECTOR_DIMENSION, SECTOR_DIMENSION), dtype=complex)
    labels = primitive.paulis.to_labels()
    for label, coefficient in zip(labels, primitive.coeffs):
        x_mask, z_mask, y_count = pauli_masks(str(label))
        y_phase = (1j) ** y_count
        for column, bitstring in enumerate(bits):
            target = bitstring ^ x_mask
            row = address.get(target)
            if row is None:
                continue
            parity = (bitstring & z_mask).bit_count() & 1
            phase = -y_phase if parity else y_phase
            result[row, column] += complex(coefficient) * phase
    return result


def diagonal_gauge_aligned_residual(
    reference: np.ndarray,
    candidate: np.ndarray,
    edge_tolerance: float = 1e-10,
) -> tuple[float, np.ndarray]:
    if reference.shape != candidate.shape:
        raise ValueError("Gauge-alignment matrices have different shapes.")
    size = reference.shape[0]
    phases = np.full(size, np.nan + 0.0j, dtype=complex)
    for root in range(size):
        if np.isfinite(phases[root].real):
            continue
        phases[root] = 1.0
        stack = [root]
        while stack:
            left = stack.pop()
            for right in range(size):
                if left == right:
                    continue
                ref = reference[left, right]
                cur = candidate[left, right]
                if max(abs(ref), abs(cur)) <= edge_tolerance:
                    continue
                if min(abs(ref), abs(cur)) <= edge_tolerance:
                    continue
                required = ref * phases[left] / cur
                required /= abs(required)
                if not np.isfinite(phases[right].real):
                    phases[right] = required
                    stack.append(right)
    phases[~np.isfinite(phases.real)] = 1.0
    aligned = phases[:, None].conj() * candidate * phases[None, :]
    return float(np.max(np.abs(reference - aligned))), phases


def build_qiskit_expanded_system(
    frozen: p7a2.FrozenGeometry,
    accepted_audit: dict[str, Any],
    audit_tolerance: float,
) -> QiskitExpandedSystem:
    from pyscf import mcscf
    from pyscf.fci import direct_spin1
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import ElectronicIntegrals
    from qiskit_nature.second_q.problems import ElectronicBasis
    from qiskit_nature.second_q.properties import AngularMomentum
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, BasisTransformer
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
    raw_problem = driver.to_problem(basis=ElectronicBasis.MO, include_dipole=False)
    raw_problem = p7a2.unfold_problem_two_body_integrals(raw_problem)
    driver_coefficients = p7a2.extract_driver_coefficients(schema)
    direct_coefficients = np.asarray(frozen.target.mean_field.mo_coeff, dtype=float)
    ao_overlap = np.asarray(frozen.target.mean_field.get_ovlp(), dtype=float)
    gauge = direct_coefficients.T @ ao_overlap @ driver_coefficients
    gauge_residual = float(
        np.max(np.abs(gauge.T @ gauge - np.eye(p7a2.FULL_TARGET_ORBITALS)))
    )
    if gauge_residual > audit_tolerance:
        raise RuntimeError(f"Direct/Qiskit MO gauge failed at R={bond}.")
    driver_rotation = gauge.T @ frozen.full_rotation_direct_mo
    physical_direct = direct_coefficients @ frozen.full_rotation_direct_mo
    physical_driver = driver_coefficients @ driver_rotation
    difference = physical_direct - physical_driver
    orbital_residual = float(
        np.max(
            np.sqrt(
                np.maximum(
                    0.0,
                    np.diag(difference.T @ ao_overlap @ difference),
                )
            )
        )
    )
    if orbital_residual > audit_tolerance:
        raise RuntimeError(f"Physical-orbital reconstruction failed at R={bond}.")

    coefficients = ElectronicIntegrals.from_raw_integrals(
        driver_rotation, h1_b=driver_rotation
    )
    rotated_full = BasisTransformer(
        ElectronicBasis.MO, ElectronicBasis.MO, coefficients
    ).transform(copy.deepcopy(raw_problem))
    problem = ActiveSpaceTransformer(
        num_electrons=(1, 1),
        num_spatial_orbitals=EXPANDED_ORBITALS,
        active_orbitals=list(range(1, 11)),
    ).transform(rotated_full)
    mapper = JordanWignerMapper()
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    constants = {
        str(key): float(np.real(value))
        for key, value in problem.hamiltonian.constants.items()
    }
    total_offset = float(sum(constants.values()))
    qiskit_sector = project_pauli_operator(qubit_op)
    qiskit_sector += total_offset * np.eye(SECTOR_DIMENSION)
    qiskit_hermiticity = float(
        np.max(np.abs(qiskit_sector - qiskit_sector.conj().T))
    )
    if qiskit_hermiticity > audit_tolerance:
        raise RuntimeError(f"Projected Qiskit Hamiltonian is not Hermitian at R={bond}.")

    spin_ops = AngularMomentum(EXPANDED_ORBITALS).second_q_ops()
    if "AngularMomentum" not in spin_ops:
        raise RuntimeError("Qiskit Nature did not construct the AngularMomentum operator.")
    spin_qubit = mapper.map(spin_ops["AngularMomentum"])
    spin_sector = project_pauli_operator(spin_qubit)
    spin_hermiticity = float(np.max(np.abs(spin_sector - spin_sector.conj().T)))
    if spin_hermiticity > audit_tolerance:
        raise RuntimeError("The projected S^2 operator is not Hermitian.")

    physical = direct_coefficients @ np.asarray(
        frozen.full_rotation_direct_mo, dtype=float
    )
    casci = mcscf.CASCI(frozen.target.mean_field, EXPANDED_ORBITALS, (1, 1))
    casci.verbose = 0
    casci.mo_coeff = physical
    h1eff, core_energy = casci.get_h1eff()
    h2eff = casci.get_h2eff()
    addresses, pyscf_electronic = direct_spin1.pspace(
        np.asarray(h1eff, dtype=float),
        np.asarray(h2eff, dtype=float),
        EXPANDED_ORBITALS,
        (1, 1),
        np=SECTOR_DIMENSION,
    )
    if not np.array_equal(addresses, np.arange(SECTOR_DIMENSION)):
        raise RuntimeError("PySCF returned an unexpected determinant address order.")
    pyscf_sector = np.asarray(pyscf_electronic, dtype=complex)
    pyscf_sector += float(core_energy) * np.eye(SECTOR_DIMENSION)
    pyscf_hermiticity = float(
        np.max(np.abs(pyscf_sector - pyscf_sector.conj().T))
    )
    aligned_residual, _ = diagonal_gauge_aligned_residual(
        qiskit_sector, pyscf_sector
    )
    if max(pyscf_hermiticity, aligned_residual) > audit_tolerance:
        raise RuntimeError(
            f"Independent Qiskit/PySCF determinant Hamiltonians disagree at "
            f"R={bond}: {aligned_residual:.3e}."
        )
    if problem.reference_energy is None:
        raise RuntimeError("The expanded Qiskit problem lacks an HF reference energy.")
    energies = accepted_audit["energies_ha"]
    return QiskitExpandedSystem(
        geometry_key=geometry_key(bond),
        bond_length=bond,
        problem=problem,
        mapper=mapper,
        qubit_op=qubit_op,
        spin_square_qubit_op=spin_qubit,
        total_offset_ha=total_offset,
        hf_total_ha=float(np.real(problem.reference_energy)),
        qiskit_sector_total=qiskit_sector,
        spin_square_sector=spin_sector,
        pyscf_sector_total=pyscf_sector,
        accepted_exact_total_ha=float(
            energies["pyscf_full_noncore_2e10o_casci_total_ha"]
        ),
        accepted_internal_exact_total_ha=float(
            energies["pyscf_rotated_2e5o_casci_total_ha"]
        ),
        gauge_orthogonality_residual=gauge_residual,
        physical_orbital_residual=orbital_residual,
        qiskit_hermiticity_residual=qiskit_hermiticity,
        pyscf_hermiticity_residual=pyscf_hermiticity,
        qiskit_pyscf_gauge_aligned_residual=aligned_residual,
    )


def support_payload(
    inputs: Phase7DInputs,
    pool: Sequence[Excitation],
    variants: Sequence[Variant],
) -> dict[str, Any]:
    candidate = next(item for item in variants if item.name == "spin_complete_candidate47")
    boundary = next(item for item in variants if item.name == "chemical_boundary46")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "phase7c_protocol_fingerprint": inputs.protocol[
            "phase7c_protocol_fingerprint"
        ],
        "phase7c_support_fingerprint": inputs.support[
            "phase7c_support_fingerprint"
        ],
        "confirmation_support_frozen_before_phase7d_exact_and_vqe_evaluation": True,
        "support_reselection_in_phase7d": False,
        "geometries_angstrom": list(EXPECTED_BOND_LENGTHS),
        "expanded_pool": [
            {
                "global_uccsd_index": index,
                "operator_label": p7c.excitation_label(index, excitation),
                "excitation": json_ready(excitation),
            }
            for index, excitation in enumerate(pool)
        ],
        "variants": [
            {
                "name": variant.name,
                "selected_global_uccsd_indices": list(variant.selected_ids),
                "operator_count": len(variant.selected_ids),
                "spin_complement_complete": not bool(
                    spin_missing_partners(pool, variant.selected_ids)
                ),
                "missing_spin_partners": [
                    {"present_global_id": left, "missing_partner_global_id": right}
                    for left, right in spin_missing_partners(pool, variant.selected_ids)
                ],
                "scientific_role": variant.scientific_role,
            }
            for variant in variants
        ],
        "decisive_pair": {
            "alpha_global_id": DECISIVE_ALPHA_ID,
            "alpha_label": p7c.excitation_label(
                DECISIVE_ALPHA_ID, pool[DECISIVE_ALPHA_ID]
            ),
            "beta_global_id": DECISIVE_BETA_ID,
            "beta_label": p7c.excitation_label(
                DECISIVE_BETA_ID, pool[DECISIVE_BETA_ID]
            ),
            "boundary46_contains_alpha_only": bool(
                DECISIVE_ALPHA_ID in boundary.selected_ids
                and DECISIVE_BETA_ID not in boundary.selected_ids
            ),
            "candidate47_completes_pair": bool(
                DECISIVE_ALPHA_ID in candidate.selected_ids
                and DECISIVE_BETA_ID in candidate.selected_ids
            ),
            "interpretation": (
                "E17 is not a large direct HF-coupling channel. It is the "
                "spin complement of E8, and its addition restores structural "
                "spin-complement completeness."
            ),
        },
        "claim_boundary": (
            "The frozen variants test noiseless LiH/6-31G 2e,10o confirmation "
            "at the three accepted geometries only."
        ),
    }
    payload["phase7d_support_fingerprint"] = p7.fingerprint(payload)
    return payload


def protocol_payload(
    support: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "phase7d_support_fingerprint": support["phase7d_support_fingerprint"],
        "phase7c_protocol_fingerprint": support["phase7c_protocol_fingerprint"],
        "phase7c_support_fingerprint": support["phase7c_support_fingerprint"],
        "hamiltonian_construction": (
            "Qiskit Nature PySCFDriver in MO basis, accepted physical orbital "
            "rotation through BasisTransformer, explicit noncore indices 1..10 "
            "through ActiveSpaceTransformer, JordanWignerMapper"
        ),
        "independent_hamiltonian_crosscheck": (
            "Qiskit Pauli Hamiltonian projected to the 100-determinant sector "
            "versus independent PySCF CASCI effective-integral pspace matrix"
        ),
        "variational_simulator": (
            "Qiskit-generated UCC excitation operators projected from their "
            "Jordan-Wigner SparsePauliOp representations into Nalpha=1,Nbeta=1"
        ),
        "ansatz_order": "ascending expanded global UCCSD index",
        "ansatz_repetitions": 1,
        "optimizer": "L-BFGS-B_with_analytic_reverse_mode_gradient",
        "maxiter": args.maxiter,
        "optimizer_tolerance": args.optimizer_tolerance,
        "initial_scale": args.initial_scale,
        "parameter_seed": args.parameter_seed,
        "base_restarts": 2,
        "rescue_restarts": args.rescue_restarts,
        "restart_stability_tolerance_ha": args.disagreement_tolerance,
        "primary_confirmation_gate": {
            "candidate47_stable_all_geometries": True,
            "candidate47_strict_error_at_most_ha": STRICT_EQUIVALENCE_HA,
            "candidate47_spin_square_at_most": SPIN_SQUARE_TOLERANCE,
            "full99_stable_and_strict_all_geometries": True,
            "compact10_reproduces_phase7c_energy_within_ha": args.audit_tolerance,
            "boundary46_reproduces_phase7c_energy_within_ha": args.audit_tolerance,
            "candidate47_reproduces_phase7c_energy_within_ha": args.audit_tolerance,
            "qiskit_pyscf_exact_audit_within_ha": args.audit_tolerance,
            "equilibrium_full_statevector_spotcheck_required": True,
        },
        "classification_thresholds_ha": {
            "strict": STRICT_EQUIVALENCE_HA,
            "guarded": GUARD_BAND_HA,
            "chemical": CHEMICAL_ACCURACY_HA,
        },
        "spin_observable": "Qiskit Nature AngularMomentum S^2",
        "full_statevector_spotcheck": {
            "variant": "spin_complete_candidate47",
            "geometry_angstrom": REFERENCE_BOND_LENGTH,
            "statevector_qubits": EXPANDED_QUBITS,
            "minimum_sector_fidelity": 1.0 - args.statevector_tolerance,
            "maximum_particle_sector_leakage": args.statevector_tolerance,
            "maximum_energy_residual_ha": args.statevector_tolerance,
        },
        "execution_controls_excluded_from_protocol_identity": [
            "resume",
            "freeze_only",
            "skip_full_statevector_spotcheck",
            "output_dir",
        ],
        "noiseless_only": True,
    }
    payload["phase7d_protocol_fingerprint"] = p7.fingerprint(payload)
    return payload


def extract_sector_generator(
    global_id: int,
    matrix: np.ndarray,
    tolerance: float,
) -> SectorGenerator:
    hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
    diagonal = float(np.max(np.abs(np.diag(matrix))))
    if max(hermiticity, diagonal) > tolerance:
        raise RuntimeError(
            f"Projected Qiskit UCC generator E{global_id} failed Hermiticity/diagonal audit."
        )
    pairs: list[tuple[int, int, complex, float]] = []
    used: set[int] = set()
    for left in range(SECTOR_DIMENSION):
        targets = [
            right
            for right in range(left + 1, SECTOR_DIMENSION)
            if abs(matrix[left, right]) > tolerance
        ]
        if len(targets) > 1:
            raise RuntimeError(f"Qiskit UCC generator E{global_id} is not pair-disjoint.")
        if not targets:
            continue
        right = targets[0]
        if left in used or right in used:
            raise RuntimeError(f"Qiskit UCC generator E{global_id} reuses a sector state.")
        if np.count_nonzero(np.abs(matrix[right, :]) > tolerance) != 1:
            raise RuntimeError(f"Qiskit UCC generator E{global_id} has a non-pair row.")
        coupling = complex(matrix[left, right])
        magnitude = abs(coupling)
        if magnitude <= tolerance:
            continue
        pairs.append((left, right, coupling, magnitude))
        used.update((left, right))
    if not pairs:
        raise RuntimeError(f"Projected Qiskit UCC generator E{global_id} is zero.")
    return SectorGenerator(global_id, tuple(pairs))


def build_qiskit_ansatz_data(
    mapper: Any,
    pool: Sequence[Excitation],
    variant: Variant,
    tolerance: float,
) -> QiskitAnsatzData:
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

    selected = [pool[index] for index in variant.selected_ids]

    def generator(
        num_spatial_orbitals: int,
        num_particles: tuple[int, int],
    ) -> list[Excitation]:
        del num_spatial_orbitals, num_particles
        return list(selected)

    initial = HartreeFock(EXPANDED_ORBITALS, (1, 1), mapper)
    ansatz = UCC(
        num_spatial_orbitals=EXPANDED_ORBITALS,
        num_particles=(1, 1),
        excitations=generator,
        qubit_mapper=mapper,
        preserve_spin=True,
        reps=1,
        initial_state=initial,
    )
    _ = ansatz.num_parameters
    if ansatz.num_parameters != len(variant.selected_ids):
        raise RuntimeError(f"Qiskit UCC parameter count changed for {variant.name}.")
    operators = list(ansatz.operators)
    if len(operators) != len(variant.selected_ids):
        raise RuntimeError(f"Qiskit UCC operator count changed for {variant.name}.")
    generators = [
        extract_sector_generator(
            global_id,
            project_pauli_operator(operator),
            tolerance,
        )
        for global_id, operator in zip(variant.selected_ids, operators)
    ]
    return QiskitAnsatzData(variant, ansatz, generators)


def apply_generator(
    state: np.ndarray,
    generator: SectorGenerator,
    theta: float,
) -> np.ndarray:
    result = np.array(state, dtype=complex, copy=True)
    for left, right, coupling, magnitude in generator.pairs:
        cosine = math.cos(magnitude * float(theta))
        sine = math.sin(magnitude * float(theta))
        old_left = state[left]
        old_right = state[right]
        result[left] = cosine * old_left - 1j * sine * coupling / magnitude * old_right
        result[right] = (
            cosine * old_right
            - 1j * sine * coupling.conjugate() / magnitude * old_left
        )
    return result


def apply_generator_derivative(
    state: np.ndarray,
    generator: SectorGenerator,
    theta: float,
) -> np.ndarray:
    result = np.zeros_like(state, dtype=complex)
    for left, right, coupling, magnitude in generator.pairs:
        cosine = math.cos(magnitude * float(theta))
        sine = math.sin(magnitude * float(theta))
        old_left = state[left]
        old_right = state[right]
        result[left] = (
            -magnitude * sine * old_left - 1j * cosine * coupling * old_right
        )
        result[right] = (
            -magnitude * sine * old_right
            - 1j * cosine * coupling.conjugate() * old_left
        )
    return result


def state_from_parameters(
    parameters: np.ndarray,
    generators: Sequence[SectorGenerator],
) -> np.ndarray:
    state = np.zeros(SECTOR_DIMENSION, dtype=complex)
    state[0] = 1.0
    for theta, generator in zip(parameters, generators):
        state = apply_generator(state, generator, float(theta))
    return state


def energy_and_gradient(
    parameters: np.ndarray,
    hamiltonian: np.ndarray,
    generators: Sequence[SectorGenerator],
) -> tuple[float, np.ndarray]:
    if len(parameters) != len(generators):
        raise ValueError("Parameter and Qiskit generator counts differ.")
    initial = np.zeros(SECTOR_DIMENSION, dtype=complex)
    initial[0] = 1.0
    forward = [initial]
    for theta, generator in zip(parameters, generators):
        forward.append(apply_generator(forward[-1], generator, float(theta)))
    final = forward[-1]
    h_state = hamiltonian @ final
    energy = float(np.real(np.vdot(final, h_state)))
    gradient = np.zeros(len(parameters), dtype=float)
    costate = h_state
    for index in range(len(parameters) - 1, -1, -1):
        derivative = apply_generator_derivative(
            forward[index], generators[index], float(parameters[index])
        )
        gradient[index] = 2.0 * float(np.real(np.vdot(costate, derivative)))
        costate = apply_generator(costate, generators[index], -float(parameters[index]))
    return energy, gradient


def verify_gradient(
    ansatz_data: QiskitAnsatzData,
    hamiltonian: np.ndarray,
) -> float:
    rng = np.random.default_rng(719)
    parameters = rng.normal(scale=0.1, size=len(ansatz_data.generators))
    _, analytic = energy_and_gradient(parameters, hamiltonian, ansatz_data.generators)
    numerical = np.zeros_like(parameters)
    step = 1e-6
    indices = sorted({0, len(parameters) // 2, len(parameters) - 1})
    for index in indices:
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            energy_and_gradient(plus, hamiltonian, ansatz_data.generators)[0]
            - energy_and_gradient(minus, hamiltonian, ansatz_data.generators)[0]
        ) / (2.0 * step)
    residual = max(abs(analytic[index] - numerical[index]) for index in indices)
    if residual > 1e-6:
        raise RuntimeError(
            f"The Qiskit-sector analytic gradient failed: {residual:.3e}."
        )
    return float(residual)


def load_checkpoint(path: Path, protocol_fingerprint: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(
        path,
        dtype={
            "run_id": str,
            "phase7d_protocol_fingerprint": str,
            "phase7d_support_fingerprint": str,
            "variant": str,
            "geometry_key": str,
            "restart_name": str,
            "parameters_json": str,
        },
    )
    if frame.empty:
        return []
    if set(frame["phase7d_protocol_fingerprint"].astype(str)) != {
        protocol_fingerprint
    }:
        raise RuntimeError("The existing checkpoint belongs to another protocol.")
    return frame.to_dict(orient="records")


def checkpoint_row(rows: Sequence[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("run_id")) == run_id]
    if len(matches) > 1:
        raise RuntimeError(f"Checkpoint contains duplicate run_id {run_id}.")
    return matches[0] if matches else None


def optimize_once(
    system: QiskitExpandedSystem,
    ansatz_data: QiskitAnsatzData,
    initial: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return energy_and_gradient(
            values, system.qiskit_sector_total, ansatz_data.generators
        )

    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": args.maxiter,
            "ftol": args.optimizer_tolerance,
            "gtol": args.optimizer_tolerance,
            "maxls": 50,
        },
    )
    energy, gradient = objective(np.asarray(result.x, dtype=float))
    state = state_from_parameters(np.asarray(result.x), ansatz_data.generators)
    spin_square = float(np.real(np.vdot(state, system.spin_square_sector @ state)))
    norm_residual = abs(float(np.real(np.vdot(state, state))) - 1.0)
    return {
        "success": bool(result.success) and np.isfinite(energy),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "energy_total_ha": energy,
        "spin_square": spin_square,
        "state_norm_residual": norm_residual,
        "gradient_infinity_norm": float(np.max(np.abs(gradient))),
        "parameters_json": json.dumps([float(value) for value in result.x]),
    }


def successful_rows(
    rows: Sequence[dict[str, Any]], variant: str, key: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("variant")) == variant
        and str(row.get("geometry_key")) == key
        and as_bool(row.get("success"))
        and np.isfinite(float(row.get("energy_total_ha", np.nan)))
    ]


def run_variant_geometry(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: dict[str, Any],
    system: QiskitExpandedSystem,
    ansatz_data: QiskitAnsatzData,
    args: argparse.Namespace,
) -> None:
    variant = ansatz_data.variant.name
    seed_base = (
        args.parameter_seed
        + int(round(system.bond_length * 1000)) * 100000
        + sum(ord(value) for value in variant)
    )
    restart_specs = [
        ("zero", seed_base, np.zeros(len(ansatz_data.generators))),
        (
            "seeded_random_0",
            seed_base + 1,
            np.random.default_rng(seed_base + 1).normal(
                scale=args.initial_scale, size=len(ansatz_data.generators)
            ),
        ),
    ]
    for name, seed, initial in restart_specs:
        run_id = f"{variant}|{system.geometry_key}|{name}"
        prior = checkpoint_row(rows, run_id)
        if prior is not None:
            if args.resume:
                continue
            raise RuntimeError(f"Run {run_id} already exists; use --resume.")
        result = optimize_once(system, ansatz_data, initial, args)
        rows.append(
            {
                "run_id": run_id,
                "phase7d_protocol_fingerprint": protocol[
                    "phase7d_protocol_fingerprint"
                ],
                "phase7d_support_fingerprint": protocol[
                    "phase7d_support_fingerprint"
                ],
                "variant": variant,
                "geometry_key": system.geometry_key,
                "bond_length_angstrom": system.bond_length,
                "operator_count": len(ansatz_data.variant.selected_ids),
                "restart_name": name,
                "parameter_seed": seed,
                **result,
            }
        )
        write_csv(checkpoint_path, rows)
    current = successful_rows(rows, variant, system.geometry_key)
    spread = (
        max(float(row["energy_total_ha"]) for row in current)
        - min(float(row["energy_total_ha"]) for row in current)
        if len(current) >= 2
        else float("inf")
    )
    if len(current) >= 2 and spread <= args.disagreement_tolerance:
        return
    for rescue in range(args.rescue_restarts):
        name = f"rescue_{rescue}"
        run_id = f"{variant}|{system.geometry_key}|{name}"
        prior = checkpoint_row(rows, run_id)
        if prior is not None:
            if args.resume:
                continue
            raise RuntimeError(f"Run {run_id} already exists; use --resume.")
        seed = seed_base + 100 + rescue
        initial = np.random.default_rng(seed).normal(
            scale=args.initial_scale, size=len(ansatz_data.generators)
        )
        result = optimize_once(system, ansatz_data, initial, args)
        rows.append(
            {
                "run_id": run_id,
                "phase7d_protocol_fingerprint": protocol[
                    "phase7d_protocol_fingerprint"
                ],
                "phase7d_support_fingerprint": protocol[
                    "phase7d_support_fingerprint"
                ],
                "variant": variant,
                "geometry_key": system.geometry_key,
                "bond_length_angstrom": system.bond_length,
                "operator_count": len(ansatz_data.variant.selected_ids),
                "restart_name": name,
                "parameter_seed": seed,
                **result,
            }
        )
        write_csv(checkpoint_path, rows)


def summarize_runs(
    rows: Sequence[dict[str, Any]],
    systems: dict[str, QiskitExpandedSystem],
    variants: Sequence[Variant],
    phase7c: Phase7DInputs,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in sorted(systems, key=float):
        system = systems[key]
        for variant in variants:
            successful = successful_rows(rows, variant.name, key)
            energies = [float(row["energy_total_ha"]) for row in successful]
            if energies:
                best_row = min(successful, key=lambda row: float(row["energy_total_ha"]))
                best = float(best_row["energy_total_ha"])
                spin_square = float(best_row["spin_square"])
                parameters_json = str(best_row["parameters_json"])
            else:
                best = float("nan")
                spin_square = float("nan")
                parameters_json = "[]"
            spread = max(energies) - min(energies) if len(energies) >= 2 else float("nan")
            stable = bool(
                len(energies) >= 2
                and np.isfinite(spread)
                and spread <= args.disagreement_tolerance
            )
            exact_error = best - system.accepted_exact_total_ha
            phase7c_match = phase7c.deterministic[
                (phase7c.deterministic["geometry_key"].astype(str) == key)
                & (
                    phase7c.deterministic["external_k"]
                    == {
                        "compact10": 0,
                        "chemical_boundary46": BOUNDARY_EXTERNAL_K,
                        "spin_complete_candidate47": CANDIDATE_EXTERNAL_K,
                    }.get(variant.name, -1)
                )
            ]
            phase7c_energy = (
                float(phase7c_match.iloc[0]["best_variational_total_ha"])
                if len(phase7c_match) == 1
                else float(
                    phase7c.full99[
                        phase7c.full99["geometry_key"].astype(str) == key
                    ].iloc[0]["best_variational_total_ha"]
                )
            )
            result.append(
                {
                    "geometry_key": key,
                    "bond_length_angstrom": system.bond_length,
                    "variant": variant.name,
                    "operator_count": len(variant.selected_ids),
                    "successful_restarts": len(energies),
                    "analysis_stable": stable,
                    "best_variational_total_ha": best,
                    "restart_energy_spread_ha": spread,
                    "accepted_full_2e10o_exact_total_ha": system.accepted_exact_total_ha,
                    "best_error_vs_full_exact_ha": exact_error,
                    "phase7c_discovery_best_total_ha": phase7c_energy,
                    "phase7d_vs_phase7c_best_energy_residual_ha": best - phase7c_energy,
                    "best_spin_square": spin_square,
                    "spin_singlet_passed": bool(
                        np.isfinite(spin_square)
                        and abs(spin_square) <= SPIN_SQUARE_TOLERANCE
                    ),
                    "strict_accuracy": bool(
                        stable and abs(exact_error) <= STRICT_EQUIVALENCE_HA
                    ),
                    "guarded_accuracy": bool(
                        stable and abs(exact_error) <= GUARD_BAND_HA
                    ),
                    "chemical_accuracy": bool(
                        stable and abs(exact_error) <= CHEMICAL_ACCURACY_HA
                    ),
                    "variational_bound_respected": bool(
                        np.isfinite(exact_error) and exact_error >= -STRICT_EQUIVALENCE_HA
                    ),
                    "best_parameters_json": parameters_json,
                }
            )
    return result


def post_freeze_exact_audit(
    systems: dict[str, QiskitExpandedSystem],
    tolerance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(systems, key=float):
        system = systems[key]
        qiskit_values, qiskit_vectors = np.linalg.eigh(system.qiskit_sector_total)
        qiskit_exact = float(qiskit_values[0])
        qiskit_ground = qiskit_vectors[:, 0]
        qiskit_ground_spin_square = float(
            np.real(
                np.vdot(
                    qiskit_ground,
                    system.spin_square_sector @ qiskit_ground,
                )
            )
        )
        pyscf_exact = float(np.linalg.eigvalsh(system.pyscf_sector_total)[0])
        qiskit_residual = qiskit_exact - system.accepted_exact_total_ha
        pyscf_residual = pyscf_exact - system.accepted_exact_total_ha
        passed = bool(
            abs(qiskit_residual) <= tolerance
            and abs(pyscf_residual) <= tolerance
            and abs(qiskit_exact - pyscf_exact) <= tolerance
            and abs(qiskit_ground_spin_square) <= SPIN_SQUARE_TOLERANCE
        )
        rows.append(
            {
                "geometry_key": key,
                "bond_length_angstrom": system.bond_length,
                "qiskit_sector_exact_total_ha": qiskit_exact,
                "pyscf_sector_exact_total_ha": pyscf_exact,
                "accepted_phase7c_exact_total_ha": system.accepted_exact_total_ha,
                "qiskit_vs_accepted_residual_ha": qiskit_residual,
                "pyscf_vs_accepted_residual_ha": pyscf_residual,
                "qiskit_vs_pyscf_exact_residual_ha": qiskit_exact - pyscf_exact,
                "qiskit_exact_ground_spin_square": qiskit_ground_spin_square,
                "qiskit_exact_ground_singlet_passed": bool(
                    abs(qiskit_ground_spin_square) <= SPIN_SQUARE_TOLERANCE
                ),
                "exact_audit_passed": passed,
            }
        )
    if not all(row["exact_audit_passed"] for row in rows):
        raise RuntimeError("The post-freeze Phase-7D exact Hamiltonian audit failed.")
    return rows


def full_statevector_spotcheck(
    ansatz_data: QiskitAnsatzData,
    system: QiskitExpandedSystem,
    summary_row: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    parameters = np.asarray(json.loads(summary_row["best_parameters_json"]), dtype=float)
    sector_state = state_from_parameters(parameters, ansatz_data.generators)
    ansatz = ansatz_data.ansatz
    try:
        ordered = list(ansatz.ordered_parameters)
    except AttributeError:
        ordered = list(ansatz.parameters)
    if len(ordered) != len(parameters):
        raise RuntimeError("Qiskit ansatz parameter binding order is incomplete.")
    bound = ansatz.assign_parameters(
        {parameter: value for parameter, value in zip(ordered, parameters)},
        inplace=False,
    )

    # Statevector.from_instruction() checks whether an instruction supplies a
    # matrix before recursively using its definition.  A bound 20-qubit
    # PauliEvolutionGate therefore calls PauliEvolutionGate.to_matrix(), which
    # attempts to materialize a 2**20 by 2**20 complex array (16 TiB).  Compile
    # the already-defined Qiskit UCC circuit to one- and two-qubit gates first.
    # This changes only the circuit representation; it does not change the
    # frozen support, parameters, operator order, or protocol identity.
    simulation_circuit = transpile(
        bound,
        basis_gates=list(STATEVECTOR_BASIS_GATES),
        optimization_level=0,
    )
    operation_counts = {
        str(name): int(count)
        for name, count in simulation_circuit.count_ops().items()
    }
    unsupported = sorted(
        set(operation_counts).difference(STATEVECTOR_ALLOWED_OPERATIONS)
    )
    if unsupported:
        raise RuntimeError(
            "The statevector circuit still contains unsynthesized operations: "
            + ", ".join(unsupported)
        )
    maximum_instruction_qubits = max(
        (len(instruction.qubits) for instruction in simulation_circuit.data),
        default=0,
    )
    if maximum_instruction_qubits > 2:
        raise RuntimeError(
            "The statevector circuit contains an instruction wider than two qubits."
        )
    if simulation_circuit.num_qubits != EXPANDED_QUBITS:
        raise RuntimeError("The compiled statevector circuit has the wrong width.")
    if simulation_circuit.num_clbits:
        raise RuntimeError("The statevector spot-check circuit contains classical bits.")

    print(
        "Statevector circuit synthesized to "
        f"{sum(operation_counts.values())} one-/two-qubit operations "
        f"(depth {simulation_circuit.depth()}); evolving 2^{EXPANDED_QUBITS} amplitudes...",
        flush=True,
    )
    full_state = np.asarray(
        Statevector.from_instruction(simulation_circuit).data,
        dtype=complex,
    )
    bits = np.asarray(determinant_bit_indices(), dtype=int)
    projected = full_state[bits]
    projected_norm = float(np.real(np.vdot(projected, projected)))
    leakage = max(0.0, 1.0 - projected_norm)
    overlap = complex(np.vdot(sector_state, projected))
    fidelity = abs(overlap) ** 2 / max(projected_norm, 1e-300)
    phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0 + 0.0j
    amplitude_residual = float(np.linalg.norm(projected - phase * sector_state))
    statevector_energy = float(
        np.real(np.vdot(projected, system.qiskit_sector_total @ projected))
    )
    sector_energy = float(summary_row["best_variational_total_ha"])
    energy_residual = statevector_energy - sector_energy
    statevector_spin = float(
        np.real(np.vdot(projected, system.spin_square_sector @ projected))
    )
    passed = bool(
        leakage <= tolerance
        and 1.0 - fidelity <= tolerance
        and abs(energy_residual) <= tolerance
        and amplitude_residual <= math.sqrt(tolerance)
    )
    return {
        "variant": ansatz_data.variant.name,
        "geometry_key": system.geometry_key,
        "bond_length_angstrom": system.bond_length,
        "full_statevector_dimension": int(len(full_state)),
        "statevector_simulation_method": (
            "Qiskit Statevector after explicit Pauli-evolution circuit synthesis"
        ),
        "transpiled_basis_gates": list(STATEVECTOR_BASIS_GATES),
        "transpiled_operation_counts": operation_counts,
        "transpiled_total_operation_count": int(sum(operation_counts.values())),
        "transpiled_circuit_depth": int(simulation_circuit.depth()),
        "maximum_instruction_qubits": int(maximum_instruction_qubits),
        "statevector_storage_bytes_complex128": int(full_state.nbytes),
        "avoided_dense_operator_bytes_complex128": int(
            (1 << EXPANDED_QUBITS) ** 2 * np.dtype(np.complex128).itemsize
        ),
        "sector_probability": projected_norm,
        "particle_sector_leakage": leakage,
        "sector_state_fidelity": fidelity,
        "global_phase_aligned_amplitude_residual": amplitude_residual,
        "full_statevector_projected_energy_total_ha": statevector_energy,
        "sector_simulator_energy_total_ha": sector_energy,
        "energy_residual_ha": energy_residual,
        "full_statevector_projected_spin_square": statevector_spin,
        "spotcheck_passed": passed,
    }


def cohort(summary: Sequence[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    return [row for row in summary if row["variant"] == variant]


def make_conclusions(
    support: dict[str, Any],
    protocol: dict[str, Any],
    exact_rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    spotcheck: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate = cohort(summary, "spin_complete_candidate47")
    boundary = cohort(summary, "chemical_boundary46")
    compact = cohort(summary, "compact10")
    full = cohort(summary, "full99")
    complete = all(len(group) == len(EXPECTED_BOND_LENGTHS) for group in (
        candidate, boundary, compact, full
    ))

    def all_true(group: Sequence[dict[str, Any]], field: str) -> bool:
        return bool(group) and all(bool(row[field]) for row in group)

    reproduction = {
        name: max(
            abs(float(row["phase7d_vs_phase7c_best_energy_residual_ha"]))
            for row in group
        )
        for name, group in (
            ("compact10", compact),
            ("chemical_boundary46", boundary),
            ("spin_complete_candidate47", candidate),
            ("full99", full),
        )
        if group
    }
    candidate_gate = bool(
        complete
        and all_true(candidate, "analysis_stable")
        and all_true(candidate, "strict_accuracy")
        and all_true(candidate, "spin_singlet_passed")
        and all_true(candidate, "variational_bound_respected")
        and all_true(full, "analysis_stable")
        and all_true(full, "strict_accuracy")
        and all_true(full, "spin_singlet_passed")
        and all_true(full, "variational_bound_respected")
        and all(row["exact_audit_passed"] for row in exact_rows)
        and all(value <= args.audit_tolerance for value in reproduction.values())
        and spotcheck is not None
        and bool(spotcheck["spotcheck_passed"])
    )
    boundary_worst = (
        max(float(row["best_error_vs_full_exact_ha"]) for row in boundary)
        if boundary
        else None
    )
    candidate_worst = (
        max(float(row["best_error_vs_full_exact_ha"]) for row in candidate)
        if candidate
        else None
    )
    boundary_spin_max = (
        max(abs(float(row["best_spin_square"])) for row in boundary)
        if boundary
        else None
    )
    candidate_spin_max = (
        max(abs(float(row["best_spin_square"])) for row in candidate)
        if candidate
        else None
    )
    if candidate_gate:
        status = "complete_strict_47_operator_spin_complete_confirmation"
        classification = "strict_spin_complete_expanded_space_confirmation"
    elif complete:
        status = "confirmation_failed_or_inconclusive"
        classification = "not_confirmed"
    else:
        status = "partial_protocol_execution"
        classification = "incomplete"
    return {
        "analysis_status": status,
        "execution_complete": candidate_gate,
        "primary_confirmation_classification": classification,
        "phase7d_protocol_fingerprint": protocol["phase7d_protocol_fingerprint"],
        "phase7d_support_fingerprint": support["phase7d_support_fingerprint"],
        "phase7c_protocol_fingerprint": support["phase7c_protocol_fingerprint"],
        "phase7c_support_fingerprint": support["phase7c_support_fingerprint"],
        "confirmed_candidate": {
            "variant": "spin_complete_candidate47",
            "operator_count": CANDIDATE_SIZE,
            "full99_operator_reduction_count": FULL_POOL_SIZE - CANDIDATE_SIZE,
            "full99_operator_reduction_fraction": (
                FULL_POOL_SIZE - CANDIDATE_SIZE
            ) / FULL_POOL_SIZE,
            "worst_geometry_error_ha": candidate_worst,
            "maximum_spin_square": candidate_spin_max,
            "strict_all_geometries": all_true(candidate, "strict_accuracy"),
            "spin_singlet_all_geometries": all_true(candidate, "spin_singlet_passed"),
        },
        "chemical_boundary_control": {
            "variant": "chemical_boundary46",
            "operator_count": BOUNDARY_SIZE,
            "structurally_spin_incomplete": True,
            "missing_beta_partner_global_id": DECISIVE_BETA_ID,
            "worst_geometry_error_ha": boundary_worst,
            "maximum_spin_square": boundary_spin_max,
            "chemical_all_geometries": all_true(boundary, "chemical_accuracy"),
            "guarded_all_geometries": all_true(boundary, "guarded_accuracy"),
        },
        "mechanistic_interpretation": (
            "The Phase-7C k=36 to k=37 discontinuity is a spin-complement "
            "completion effect. E8 is the alpha excitation 0->9; E17 is its "
            "beta counterpart 10->19. The frozen 47-operator support is fully "
            "closed under alpha/beta spin complementation, whereas boundary46 "
            "contains E8 without E17."
        ),
        "phase7c_energy_reproduction_maximum_residuals_ha": reproduction,
        "exact_hamiltonian_audit_passed": all(
            row["exact_audit_passed"] for row in exact_rows
        ),
        "full_statevector_spotcheck": spotcheck,
        "claim_boundary": (
            "The result confirms a noiseless 47-operator spin-complete UCC "
            "support for the three frozen LiH/6-31G 2e,10o Hamiltonians. It "
            "does not establish transfer to other molecules, bases, active "
            "spaces, geometries outside the tested set, or noisy hardware."
        ),
    }


def write_report(
    path: Path,
    conclusions: dict[str, Any],
    summary: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# LiH Phase 7D Frozen Qiskit Confirmation",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        f"Primary classification: **{conclusions['primary_confirmation_classification']}**",
        "",
        "## Variational confirmation",
        "",
        "| R (Å) | Variant | Operators | Error vs exact (Ha) | S² | Stable | Strict | Chemical |",
        "|---:|---|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in sorted(summary, key=lambda item: (item["bond_length_angstrom"], item["variant"])):
        lines.append(
            f"| {float(row['bond_length_angstrom']):.3f} | {row['variant']} | "
            f"{row['operator_count']} | {float(row['best_error_vs_full_exact_ha']):.10e} | "
            f"{float(row['best_spin_square']):.3e} | {row['analysis_stable']} | "
            f"{row['strict_accuracy']} | {row['chemical_accuracy']} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            conclusions["mechanistic_interpretation"],
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
            "Confirm the frozen Phase-7C 47-operator LiH/6-31G 2e,10o support "
            "with independently constructed Qiskit Hamiltonians and UCC generators."
        )
    )
    parser.add_argument("--phase7a-dir", type=Path, default=Path("lih_phase7a_audit"))
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
        "--phase7c-dir", type=Path, default=Path("lih_phase7c_expanded_augmentation")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("lih_phase7d_qiskit_confirmation")
    )
    parser.add_argument("--maxiter", type=int, default=1500)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-10)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--parameter-seed", type=int, default=20260814)
    parser.add_argument("--rescue-restarts", type=int, default=2)
    parser.add_argument("--disagreement-tolerance", type=float, default=1e-7)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--rotation-reconstruction-tolerance", type=float, default=1e-10
    )
    parser.add_argument("--statevector-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Freeze confirmation variants and protocol before exact/VQE evaluation.",
    )
    parser.add_argument(
        "--skip-full-statevector-spotcheck",
        action="store_true",
        help=(
            "Diagnostic escape hatch only. A skipped spot-check cannot produce "
            "a completed Phase-7D confirmation."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume matching restart checkpoints (default: true).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.maxiter < 1 or args.rescue_restarts < 0:
        raise ValueError("Iteration and restart counts are invalid.")
    for name in (
        "optimizer_tolerance",
        "initial_scale",
        "disagreement_tolerance",
        "audit_tolerance",
        "rotation_reconstruction_tolerance",
        "statevector_tolerance",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    diag.require_quantum_stack()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase7c = load_inputs(args.phase7c_dir)
    pool = frozen_pool(phase7c)
    variants = build_variants(phase7c, pool)
    phase7a2_inputs = p7b.load_phase7a2_inputs(args.phase7a2_dir)
    if phase7a2_inputs.phase7a2_audit["protocol_fingerprint"] != (
        phase7c.support["phase7a2_protocol_fingerprint"]
    ):
        raise RuntimeError("Phase-7A.2 and Phase-7C protocol identities disagree.")
    if phase7a2_inputs.phase7a2_audit["rotated_support_fingerprint"] != (
        phase7c.support["phase7a2_rotated_support_fingerprint"]
    ):
        raise RuntimeError("Phase-7A.2 and Phase-7C support identities disagree.")

    bundle = p7a2.load_inputs(args.phase7a_dir, args.phase7a1_dir)
    print("Reconstructing the accepted Phase-7A.2 physical orbitals...", flush=True)
    frozen, reconstruction_rows = p7a2.reconstruct_frozen_geometries(
        bundle, args.rotation_reconstruction_tolerance
    )
    accepted_rows = p7b.reconcile_with_accepted_physical_orbitals(
        frozen,
        phase7a2_inputs.phase7a2_frozen_support,
        args.rotation_reconstruction_tolerance,
    )
    accepted_by_key = {str(row["geometry_key"]): row for row in accepted_rows}
    write_csv(
        args.output_dir / "phase7d_rotation_reconstruction.csv",
        [
            {**row, **accepted_by_key[str(row["geometry_key"])]}
            for row in reconstruction_rows
        ],
    )

    print("Building independent Qiskit 2e,10o Hamiltonians without diagonalization...", flush=True)
    systems: dict[str, QiskitExpandedSystem] = {}
    construction_rows: list[dict[str, Any]] = []
    for key in sorted(frozen, key=float):
        system = build_qiskit_expanded_system(
            frozen[key], phase7a2_inputs.audits_by_geometry[key], args.audit_tolerance
        )
        systems[key] = system
        construction_rows.append(
            {
                "geometry_key": key,
                "bond_length_angstrom": system.bond_length,
                "qubits": EXPANDED_QUBITS,
                "sector_dimension": SECTOR_DIMENSION,
                "gauge_orthogonality_residual": system.gauge_orthogonality_residual,
                "physical_orbital_residual": system.physical_orbital_residual,
                "qiskit_hermiticity_residual": system.qiskit_hermiticity_residual,
                "pyscf_hermiticity_residual": system.pyscf_hermiticity_residual,
                "qiskit_pyscf_gauge_aligned_matrix_residual": (
                    system.qiskit_pyscf_gauge_aligned_residual
                ),
                "exact_diagonalization_performed_during_freeze": False,
            }
        )
    write_csv(
        args.output_dir / "phase7d_hamiltonian_construction_audit.csv",
        construction_rows,
    )

    support = support_payload(phase7c, pool, variants)
    support = write_or_validate_signed_payload(
        args.output_dir / "phase7d_frozen_confirmation_support.json",
        support,
        "phase7d_support_fingerprint",
    )
    protocol = protocol_payload(support, args)
    protocol = write_or_validate_signed_payload(
        args.output_dir / "phase7d_preregistered_protocol.json",
        protocol,
        "phase7d_protocol_fingerprint",
    )
    print(
        "Phase-7D confirmation support frozen before exact/VQE evaluation: "
        f"{support['phase7d_support_fingerprint']}",
        flush=True,
    )
    print(
        f"Phase-7D protocol fingerprint: {protocol['phase7d_protocol_fingerprint']}",
        flush=True,
    )
    if args.freeze_only:
        print("Freeze-only gate complete; no exact diagonalization or VQE performed.")
        return 0

    print("Running post-freeze exact Hamiltonian audits...", flush=True)
    exact_rows = post_freeze_exact_audit(systems, args.audit_tolerance)
    write_csv(args.output_dir / "phase7d_exact_hamiltonian_audit.csv", exact_rows)

    reference_system = systems[geometry_key(REFERENCE_BOND_LENGTH)]
    print("Constructing Qiskit UCC operators and projected sector generators...", flush=True)
    ansatz_data = {
        variant.name: build_qiskit_ansatz_data(
            reference_system.mapper, pool, variant, args.audit_tolerance
        )
        for variant in variants
    }
    gradient_rows = [
        {
            "variant": variant.name,
            "operator_count": len(variant.selected_ids),
            "analytic_vs_finite_difference_gradient_residual": verify_gradient(
                ansatz_data[variant.name], reference_system.qiskit_sector_total
            ),
        }
        for variant in variants
    ]
    write_csv(args.output_dir / "phase7d_generator_gradient_audit.csv", gradient_rows)

    checkpoint_path = args.output_dir / "phase7d_restart_runs.csv"
    runs = load_checkpoint(
        checkpoint_path, protocol["phase7d_protocol_fingerprint"]
    )
    for variant in variants:
        print(f"Running frozen Phase-7D variant {variant.name}...", flush=True)
        for key in sorted(systems, key=float):
            run_variant_geometry(
                runs,
                checkpoint_path,
                protocol,
                systems[key],
                ansatz_data[variant.name],
                args,
            )
    summary = summarize_runs(runs, systems, variants, phase7c, args)
    write_csv(args.output_dir / "phase7d_variant_geometry_summary.csv", summary)

    candidate_reference = next(
        row
        for row in summary
        if row["variant"] == "spin_complete_candidate47"
        and row["geometry_key"] == geometry_key(REFERENCE_BOND_LENGTH)
    )
    if args.skip_full_statevector_spotcheck:
        print(
            "WARNING: full 20-qubit Statevector spot-check skipped; Phase 7D "
            "cannot be classified complete.",
            file=sys.stderr,
        )
        spotcheck = None
    else:
        print("Running the full 20-qubit Qiskit Statevector spot-check...", flush=True)
        spotcheck = full_statevector_spotcheck(
            ansatz_data["spin_complete_candidate47"],
            reference_system,
            candidate_reference,
            args.statevector_tolerance,
        )
        write_json(args.output_dir / "phase7d_full_statevector_spotcheck.json", spotcheck)

    conclusions = make_conclusions(
        support, protocol, exact_rows, summary, spotcheck, args
    )
    write_json(args.output_dir / "phase7d_confirmation_conclusions.json", conclusions)
    write_report(args.output_dir / "PHASE7D_REPORT.md", conclusions, summary)

    print("\nPhase 7D deterministic summary:")
    print(
        pd.DataFrame(summary)[
            [
                "bond_length_angstrom",
                "variant",
                "operator_count",
                "analysis_stable",
                "best_error_vs_full_exact_ha",
                "best_spin_square",
                "strict_accuracy",
                "chemical_accuracy",
            ]
        ].to_string(index=False)
    )
    print("\nPhase 7D conclusions:")
    print(
        json.dumps(
            {
                "analysis_status": conclusions["analysis_status"],
                "execution_complete": conclusions["execution_complete"],
                "primary_confirmation_classification": conclusions[
                    "primary_confirmation_classification"
                ],
                "confirmed_operator_count": conclusions["confirmed_candidate"][
                    "operator_count"
                ],
                "confirmed_worst_geometry_error_ha": conclusions[
                    "confirmed_candidate"
                ]["worst_geometry_error_ha"],
                "confirmed_maximum_spin_square": conclusions[
                    "confirmed_candidate"
                ]["maximum_spin_square"],
                "phase7d_protocol_fingerprint": conclusions[
                    "phase7d_protocol_fingerprint"
                ],
                "phase7d_support_fingerprint": conclusions[
                    "phase7d_support_fingerprint"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if conclusions["execution_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
