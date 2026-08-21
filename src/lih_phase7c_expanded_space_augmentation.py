#!/usr/bin/env python3
"""Phase 7C: discover a compact external-correlation augmentation in LiH/6-31G.

The accepted Phase-7B compact10 support is embedded in the frozen-core 2e,10o
space.  The 75 newly available spin-preserving UCCSD excitations are ranked at
R=1.595 Angstrom with an Epstein--Nesbet-like HF-only score.  The complete
nested path, a bank of composition-matched random paths, and the decision rule
are signed before any exact diagonalization or variational optimization.

After that freeze gate, a 100-determinant simulator evaluates one ordered
product layer of fermionic excitation rotations.  This is a discovery and
screening phase.  A separately frozen Phase 7D must confirm the chosen support
with the production Qiskit construction before a final scientific claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import lih_phase7a_cross_basis_transfer as p7
    import lih_phase7a2_rotated_hamiltonian_audit as p7a2
    import lih_phase7b_cross_basis_support_transfer as p7b
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place this script beside the Phase-7A, Phase-7A.1, Phase-7A.2, "
        "Phase-7B, Phase-3, Phase-2B, and reference-diagnostic modules."
    ) from exc


SCHEMA_VERSION = 1
ANALYSIS_PHASE = "Phase 7C expanded-space augmentation discovery"
EXPECTED_PHASE7B_STATUS = "complete_strict_support_transfer"
EXPECTED_PHASE7B_CLASSIFICATION = "strict_support_transfer"
EXPECTED_PHASE7B_PROTOCOL_FINGERPRINT = (
    "3bdc33540a2ca90a17aeef9292938f3b88cc7d82e2bed59a162cf6aed9034e68"
)
EXPECTED_PHASE7B_SUPPORT_FINGERPRINT = (
    "babf3bae7e76f9283449b1d39f60bd33ce987e3d0771a1b7916c9103c4193c04"
)
EXPECTED_BOND_LENGTHS = (1.595, 2.5, 3.0)
REFERENCE_BOND_LENGTH = 1.595

ACTIVE_ELECTRONS = 2
EXPANDED_SPATIAL_ORBITALS = 10
EXPANDED_QUBITS = 20
EXPANDED_DETERMINANTS = 100
EXPANDED_FULL_POOL = 99
INTERNAL_FULL_POOL = 24
COMPACT_POOL = 10
EXTERNAL_POOL = 75

CHEMICAL_ACCURACY_HA = 0.0016
GUARD_BAND_HA = 0.0008
STRICT_EQUIVALENCE_HA = 1e-6


Excitation = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass
class Phase7CInputs:
    phase7a2: p7b.Phase7BInputs
    phase7b_support: dict[str, Any]
    phase7b_protocol: dict[str, Any]
    phase7b_conclusions: dict[str, Any]


@dataclass
class ExpandedSystem:
    geometry_key: str
    bond_length: float
    hamiltonian_total: np.ndarray
    core_energy_ha: float
    accepted_full_exact_total_ha: float
    accepted_internal_exact_total_ha: float
    hermiticity_residual: float
    determinant_addresses_ordered: bool


@dataclass(frozen=True)
class Generator:
    global_id: int
    excitation: Excitation
    pairs: tuple[tuple[int, int], ...]


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


def load_inputs(phase7a2_dir: Path, phase7b_dir: Path) -> Phase7CInputs:
    phase7a2 = p7b.load_phase7a2_inputs(phase7a2_dir)
    support_path = phase7b_dir / "phase7b_frozen_supports.json"
    protocol_path = phase7b_dir / "phase7b_preregistered_protocol.json"
    conclusions_path = phase7b_dir / "phase7b_transfer_conclusions.json"
    for path in (support_path, protocol_path, conclusions_path):
        require_file(path)
    support = json.loads(support_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    conclusions = json.loads(conclusions_path.read_text(encoding="utf-8"))
    validate_signed_payload(support, "phase7b_support_fingerprint")
    validate_signed_payload(protocol, "phase7b_protocol_fingerprint")

    if conclusions.get("analysis_status") != EXPECTED_PHASE7B_STATUS:
        raise RuntimeError("Phase 7B is not a completed accepted experiment.")
    if conclusions.get("primary_transfer_classification") != (
        EXPECTED_PHASE7B_CLASSIFICATION
    ):
        raise RuntimeError("Phase 7B did not establish strict support transfer.")
    if not bool(conclusions.get("execution_complete", False)):
        raise RuntimeError("Phase 7B execution_complete is not true.")
    if not bool(conclusions.get("all_optimizers_resolved", False)):
        raise RuntimeError("Phase 7B has unresolved optimizers.")
    if conclusions.get("protocol_fingerprint") != protocol.get(
        "phase7b_protocol_fingerprint"
    ):
        raise RuntimeError("Phase-7B protocol and conclusions disagree.")
    if conclusions.get("support_fingerprint") != support.get(
        "phase7b_support_fingerprint"
    ):
        raise RuntimeError("Phase-7B support and conclusions disagree.")
    if protocol.get("phase7b_support_fingerprint") != support.get(
        "phase7b_support_fingerprint"
    ):
        raise RuntimeError("Phase-7B support and protocol disagree.")
    if protocol.get("phase7a2_protocol_fingerprint") != (
        phase7a2.phase7a2_audit.get("protocol_fingerprint")
    ):
        raise RuntimeError("Phase-7B and Phase-7A.2 protocol identities disagree.")
    if protocol.get("phase7a2_rotated_support_fingerprint") != (
        phase7a2.phase7a2_audit.get("rotated_support_fingerprint")
    ):
        raise RuntimeError("Phase-7B and Phase-7A.2 supports disagree.")
    if conclusions.get("protocol_fingerprint") != (
        EXPECTED_PHASE7B_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError(
            "The Phase-7B protocol fingerprint is not the accepted completed run."
        )
    if conclusions.get("support_fingerprint") != (
        EXPECTED_PHASE7B_SUPPORT_FINGERPRINT
    ):
        raise RuntimeError(
            "The Phase-7B support fingerprint is not the accepted completed run."
        )
    return Phase7CInputs(phase7a2, support, protocol, conclusions)


def expanded_pool() -> list[Excitation]:
    norb = EXPANDED_SPATIAL_ORBITALS
    result: list[Excitation] = []
    result.extend([((0,), (orbital,)) for orbital in range(1, norb)])
    result.extend(
        [((norb,), (norb + orbital,)) for orbital in range(1, norb)]
    )
    result.extend(
        [
            ((0, norb), (alpha, norb + beta))
            for alpha in range(1, norb)
            for beta in range(1, norb)
        ]
    )
    if len(result) != EXPANDED_FULL_POOL or len(set(result)) != len(result):
        raise RuntimeError("The expanded spin-preserving UCCSD pool is invalid.")
    return result


def map_small_excitation(excitation: Excitation) -> Excitation:
    def map_index(index: int) -> int:
        if index < p7a2.ACTIVE_ORBITALS:
            return int(index)
        return int(index - p7a2.ACTIVE_ORBITALS + EXPANDED_SPATIAL_ORBITALS)

    occupied, virtual = excitation
    return (
        tuple(map_index(value) for value in occupied),
        tuple(map_index(value) for value in virtual),
    )


def pool_identities() -> tuple[
    list[Excitation], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    pool = expanded_pool()
    index_by_excitation = {item: index for index, item in enumerate(pool)}
    small_full, small_compact, regenerated = p7a2.compact_excitations()
    if small_full != regenerated or len(small_full) != INTERNAL_FULL_POOL:
        raise RuntimeError("The accepted internal full24 pool changed.")
    mapped_full = tuple(
        sorted(index_by_excitation[map_small_excitation(item)] for item in small_full)
    )
    mapped_compact = tuple(
        sorted(
            index_by_excitation[map_small_excitation(item)]
            for item in small_compact
        )
    )
    external = tuple(
        index for index in range(len(pool)) if index not in set(mapped_full)
    )
    if len(mapped_full) != INTERNAL_FULL_POOL:
        raise RuntimeError("The embedded internal full24 support is incomplete.")
    if len(mapped_compact) != COMPACT_POOL:
        raise RuntimeError("The embedded compact10 support is incomplete.")
    if len(external) != EXTERNAL_POOL:
        raise RuntimeError("The expanded external pool does not contain 75 items.")

    return pool, mapped_full, mapped_compact, external


def excitation_type(excitation: Excitation) -> str:
    return "single" if len(excitation[0]) == 1 else "double"


def excitation_label(global_id: int, excitation: Excitation) -> str:
    occupied, virtual = excitation
    left = ",".join(str(value) for value in occupied)
    right = ",".join(str(value) for value in virtual)
    return f"E{global_id}:({left})->({right})"


def validate_phase7b_compact_identity(
    inputs: Phase7CInputs,
    mapped_compact: Sequence[int],
) -> None:
    variant = next(
        (
            item
            for item in inputs.phase7b_support.get("variants", [])
            if item.get("name") == "transferred_compact10"
        ),
        None,
    )
    if variant is None:
        raise RuntimeError("The frozen Phase-7B compact10 variant is missing.")
    small_ids = tuple(int(value) for value in variant["selected_global_uccsd_indices"])
    if small_ids != tuple(int(value) for value in p7.compact_reference_indices()):
        raise RuntimeError("The accepted Phase-7B compact10 indices changed.")
    pool = expanded_pool()
    small_full, _, _ = p7a2.compact_excitations()
    expected = tuple(
        sorted(pool.index(map_small_excitation(small_full[index])) for index in small_ids)
    )
    if expected != tuple(mapped_compact):
        raise RuntimeError("The Phase-7B compact10 embedding changed.")


def build_expanded_system(
    frozen: p7a2.FrozenGeometry,
    accepted_audit: dict[str, Any],
    audit_tolerance: float,
) -> ExpandedSystem:
    from pyscf import mcscf
    from pyscf.fci import direct_spin1

    direct = np.asarray(frozen.target.mean_field.mo_coeff, dtype=float)
    physical = direct @ np.asarray(frozen.full_rotation_direct_mo, dtype=float)
    casci = mcscf.CASCI(
        frozen.target.mean_field,
        EXPANDED_SPATIAL_ORBITALS,
        (1, 1),
    )
    casci.verbose = 0
    casci.mo_coeff = physical
    h1eff, core_energy = casci.get_h1eff()
    h2eff = casci.get_h2eff()
    addresses, electronic = direct_spin1.pspace(
        np.asarray(h1eff, dtype=float),
        np.asarray(h2eff, dtype=float),
        EXPANDED_SPATIAL_ORBITALS,
        (1, 1),
        np=EXPANDED_DETERMINANTS,
    )
    ordered = bool(np.array_equal(addresses, np.arange(EXPANDED_DETERMINANTS)))
    if not ordered:
        raise RuntimeError("PySCF returned an unexpected determinant address order.")
    electronic = np.asarray(electronic, dtype=float)
    total = electronic + float(core_energy) * np.eye(EXPANDED_DETERMINANTS)
    hermiticity = float(np.max(np.abs(total - total.T)))
    if hermiticity > audit_tolerance:
        raise RuntimeError("The expanded determinant Hamiltonian is not Hermitian.")
    energies = accepted_audit["energies_ha"]
    return ExpandedSystem(
        geometry_key=geometry_key(frozen.bond_length),
        bond_length=float(frozen.bond_length),
        hamiltonian_total=total,
        core_energy_ha=float(core_energy),
        accepted_full_exact_total_ha=float(
            energies["pyscf_full_noncore_2e10o_casci_total_ha"]
        ),
        accepted_internal_exact_total_ha=float(
            energies["pyscf_rotated_2e5o_casci_total_ha"]
        ),
        hermiticity_residual=hermiticity,
        determinant_addresses_ordered=ordered,
    )


def determinant_index(alpha_orbital: int, beta_orbital: int) -> int:
    return int(alpha_orbital) * EXPANDED_SPATIAL_ORBITALS + int(beta_orbital)


def internal_determinant_indices() -> tuple[int, ...]:
    return tuple(
        determinant_index(alpha, beta)
        for alpha in range(p7a2.ACTIVE_ORBITALS)
        for beta in range(p7a2.ACTIVE_ORBITALS)
    )


def generator_for_excitation(global_id: int, excitation: Excitation) -> Generator:
    occupied, virtual = excitation
    norb = EXPANDED_SPATIAL_ORBITALS
    if len(occupied) == 1:
        source, target = occupied[0], virtual[0]
        if source < norb and target < norb:
            pairs = tuple(
                (
                    determinant_index(source, beta),
                    determinant_index(target, beta),
                )
                for beta in range(norb)
            )
        elif source >= norb and target >= norb:
            pairs = tuple(
                (
                    determinant_index(alpha, source - norb),
                    determinant_index(alpha, target - norb),
                )
                for alpha in range(norb)
            )
        else:
            raise RuntimeError("A single excitation changes spin.")
    elif len(occupied) == 2:
        if occupied != (0, norb):
            raise RuntimeError("An expanded double has unexpected occupied indices.")
        pairs = (
            (
                determinant_index(0, 0),
                determinant_index(virtual[0], virtual[1] - norb),
            ),
        )
    else:
        raise RuntimeError("Only single and double excitations are supported.")
    return Generator(global_id, excitation, pairs)


def apply_generator_rotation(
    state: np.ndarray,
    generator: Generator,
    theta: float,
) -> np.ndarray:
    result = np.array(state, dtype=float, copy=True)
    cosine = math.cos(float(theta))
    sine = math.sin(float(theta))
    for source, target in generator.pairs:
        old_source = state[source]
        old_target = state[target]
        result[source] = cosine * old_source - sine * old_target
        result[target] = sine * old_source + cosine * old_target
    return result


def apply_generator_derivative(
    state: np.ndarray,
    generator: Generator,
    theta: float,
) -> np.ndarray:
    result = np.zeros_like(state, dtype=float)
    cosine = math.cos(float(theta))
    sine = math.sin(float(theta))
    for source, target in generator.pairs:
        old_source = state[source]
        old_target = state[target]
        result[source] = -sine * old_source - cosine * old_target
        result[target] = cosine * old_source - sine * old_target
    return result


def energy_and_gradient(
    parameters: np.ndarray,
    hamiltonian: np.ndarray,
    generators: Sequence[Generator],
) -> tuple[float, np.ndarray]:
    if len(parameters) != len(generators):
        raise ValueError("Parameter and generator counts differ.")
    initial = np.zeros(EXPANDED_DETERMINANTS, dtype=float)
    initial[determinant_index(0, 0)] = 1.0
    forward = [initial]
    for theta, generator in zip(parameters, generators):
        forward.append(apply_generator_rotation(forward[-1], generator, theta))
    final = forward[-1]
    h_state = hamiltonian @ final
    energy = float(final @ h_state)
    gradient = np.zeros(len(parameters), dtype=float)
    costate = h_state
    for index in range(len(parameters) - 1, -1, -1):
        derivative = apply_generator_derivative(
            forward[index], generators[index], parameters[index]
        )
        gradient[index] = 2.0 * float(costate @ derivative)
        costate = apply_generator_rotation(
            costate, generators[index], -parameters[index]
        )
    return energy, gradient


def verify_simulator_gradient(pool: Sequence[Excitation]) -> None:
    rng = np.random.default_rng(713)
    matrix = rng.normal(size=(EXPANDED_DETERMINANTS, EXPANDED_DETERMINANTS))
    matrix = 0.5 * (matrix + matrix.T)
    ids = (0, 9, 18, 28)
    generators = [generator_for_excitation(index, pool[index]) for index in ids]
    parameters = rng.normal(scale=0.2, size=len(ids))
    _, analytic = energy_and_gradient(parameters, matrix, generators)
    numerical = np.zeros_like(parameters)
    step = 1e-6
    for index in range(len(parameters)):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            energy_and_gradient(plus, matrix, generators)[0]
            - energy_and_gradient(minus, matrix, generators)[0]
        ) / (2.0 * step)
    residual = float(np.max(np.abs(analytic - numerical)))
    if residual > 1e-6:
        raise RuntimeError(f"The determinant-simulator gradient failed: {residual:.3e}.")


def rank_external_operators(
    equilibrium: ExpandedSystem,
    pool: Sequence[Excitation],
    internal_ids: set[int],
    compact_ids: set[int],
    denominator_floor_ha: float,
) -> list[dict[str, Any]]:
    if denominator_floor_ha <= 0:
        raise ValueError("The perturbative denominator floor must be positive.")
    hf = determinant_index(0, 0)
    hf_diagonal = float(equilibrium.hamiltonian_total[hf, hf])
    rows: list[dict[str, Any]] = []
    for global_id, excitation in enumerate(pool):
        generator = generator_for_excitation(global_id, excitation)
        target = next(
            target
            for source, target in generator.pairs
            if source == hf
        )
        coupling = float(equilibrium.hamiltonian_total[target, hf])
        target_diagonal = float(equilibrium.hamiltonian_total[target, target])
        signed_gap = target_diagonal - hf_diagonal
        denominator = max(abs(signed_gap), denominator_floor_ha)
        rows.append(
            {
                "global_uccsd_index": global_id,
                "operator_label": excitation_label(global_id, excitation),
                "operator_type": excitation_type(excitation),
                "excitation": json_ready(excitation),
                "internal_2e5o_operator": global_id in internal_ids,
                "transferred_compact10_operator": global_id in compact_ids,
                "new_external_operator": global_id not in internal_ids,
                "hf_determinant_index": hf,
                "directly_excited_determinant_index": target,
                "absolute_hamiltonian_coupling_ha": abs(coupling),
                "signed_diagonal_gap_ha": signed_gap,
                "absolute_denominator_ha": denominator,
                "en_amplitude_score": abs(coupling) / denominator,
                "en_second_order_lowering_score_ha": coupling * coupling / denominator,
            }
        )
    external_rows = [row for row in rows if row["new_external_operator"]]
    ranked = sorted(
        external_rows,
        key=lambda row: (
            -float(row["en_amplitude_score"]),
            int(row["global_uccsd_index"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["external_en_amplitude_rank"] = rank
    rank_by_id = {
        int(row["global_uccsd_index"]): int(row["external_en_amplitude_rank"])
        for row in ranked
    }
    for row in rows:
        row["external_en_amplitude_rank"] = rank_by_id.get(
            int(row["global_uccsd_index"])
        )
    return rows


def external_ranking(operator_rows: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    ranked = sorted(
        (row for row in operator_rows if row["new_external_operator"]),
        key=lambda row: int(row["external_en_amplitude_rank"]),
    )
    result = tuple(int(row["global_uccsd_index"]) for row in ranked)
    if len(result) != EXTERNAL_POOL or len(set(result)) != EXTERNAL_POOL:
        raise RuntimeError("The external ranking is incomplete.")
    return result


def random_path_bank(
    external_ids: Sequence[int],
    pool: Sequence[Excitation],
    seed: int,
    bank_size: int,
) -> list[dict[str, Any]]:
    singles = np.asarray(
        [index for index in external_ids if excitation_type(pool[index]) == "single"],
        dtype=int,
    )
    doubles = np.asarray(
        [index for index in external_ids if excitation_type(pool[index]) == "double"],
        dtype=int,
    )
    if len(singles) != 10 or len(doubles) != 65:
        raise RuntimeError("The external single/double composition changed.")
    rng = np.random.default_rng(seed)
    paths: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    attempts = 0
    while len(paths) < bank_size:
        attempts += 1
        if attempts > bank_size * 1000:
            raise RuntimeError("Could not generate a unique random-path bank.")
        single_order = tuple(int(value) for value in rng.permutation(singles))
        double_order = tuple(int(value) for value in rng.permutation(doubles))
        mixed_order = tuple(int(value) for value in rng.permutation(external_ids))
        signature = (single_order, double_order)
        if signature in seen:
            continue
        seen.add(signature)
        paths.append(
            {
                "path_id": f"random_path_{len(paths):03d}",
                "external_single_order": list(single_order),
                "external_double_order": list(double_order),
                "external_mixed_order": list(mixed_order),
            }
        )
    return paths


def build_support_payload(
    inputs: Phase7CInputs,
    pool: Sequence[Excitation],
    internal_ids: Sequence[int],
    compact_ids: Sequence[int],
    operator_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ranking = external_ranking(operator_rows)
    paths = random_path_bank(
        ranking,
        pool,
        args.random_support_seed,
        args.random_path_bank_size,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "phase7b_protocol_fingerprint": inputs.phase7b_conclusions[
            "protocol_fingerprint"
        ],
        "phase7b_support_fingerprint": inputs.phase7b_conclusions[
            "support_fingerprint"
        ],
        "phase7a2_protocol_fingerprint": inputs.phase7a2.phase7a2_audit[
            "protocol_fingerprint"
        ],
        "phase7a2_rotated_support_fingerprint": inputs.phase7a2.phase7a2_audit[
            "rotated_support_fingerprint"
        ],
        "support_frozen_before_exact_diagonalization_and_variational_evaluation": True,
        "exact_energy_or_eigenvector_used_for_ranking": False,
        "reference_geometry_angstrom": REFERENCE_BOND_LENGTH,
        "geometries_angstrom": list(EXPECTED_BOND_LENGTHS),
        "expanded_space": {
            "electrons": ACTIVE_ELECTRONS,
            "spatial_orbitals": EXPANDED_SPATIAL_ORBITALS,
            "qubits": EXPANDED_QUBITS,
            "determinants_in_nalpha1_nbeta1_sector": EXPANDED_DETERMINANTS,
        },
        "expanded_full_pool_size": EXPANDED_FULL_POOL,
        "internal_full24_global_indices": list(internal_ids),
        "transferred_compact10_global_indices": list(compact_ids),
        "external_pool_size": EXTERNAL_POOL,
        "external_ranking_global_indices": list(ranking),
        "nested_external_k_schedule": list(range(EXTERNAL_POOL + 1)),
        "ranking": {
            "method": (
                "abs(<D_mu|H|HF>)/max(abs(<D_mu|H|D_mu>-<HF|H|HF>),floor)"
            ),
            "denominator_floor_ha": args.perturbative_denominator_floor,
            "uses_only_hf_diagonal_excited_diagonal_and_direct_coupling": True,
            "ordinary_mp2_label_rejected_for_noncanonical_rotated_orbitals": True,
        },
        "expanded_pool": [
            {
                "global_uccsd_index": index,
                "operator_label": excitation_label(index, excitation),
                "operator_type": excitation_type(excitation),
                "excitation": json_ready(excitation),
            }
            for index, excitation in enumerate(pool)
        ],
        "operator_score_rows": list(operator_rows),
        "random_control": {
            "seed": args.random_support_seed,
            "path_bank_size": args.random_path_bank_size,
            "number_of_paths_evaluated_at_selected_k": args.random_controls,
            "composition_matching": (
                "At selected k, take the same counts of external singles and "
                "doubles as the deterministic top-k path."
            ),
            "fallback_if_fewer_than_requested_composition_matches": (
                "Fill from frozen mixed external permutations at the same k."
            ),
            "full_external_k75_random_control_not_applicable": True,
            "first_unique_non_deterministic_paths_from_frozen_bank": True,
            "paths": paths,
        },
        "claim_boundary": (
            "This frozen support defines a LiH/6-31G 2e,10o discovery path. "
            "Phase 7C cannot by itself establish production-Qiskit VQE "
            "performance or transfer beyond these three geometries."
        ),
    }
    payload["phase7c_support_fingerprint"] = p7.fingerprint(payload)
    return payload


def load_or_create_support(
    path: Path,
    inputs: Phase7CInputs,
    pool: Sequence[Excitation],
    internal_ids: Sequence[int],
    compact_ids: Sequence[int],
    operator_rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if path.is_file():
        support = json.loads(path.read_text(encoding="utf-8"))
        validate_signed_payload(support, "phase7c_support_fingerprint")
        if support.get("phase7b_protocol_fingerprint") != (
            inputs.phase7b_conclusions.get("protocol_fingerprint")
        ):
            raise RuntimeError("The frozen Phase-7C input protocol changed.")
        if support.get("phase7b_support_fingerprint") != (
            inputs.phase7b_conclusions.get("support_fingerprint")
        ):
            raise RuntimeError("The frozen Phase-7C input support changed.")
        ranking = support.get("ranking", {})
        if not math.isclose(
            float(ranking.get("denominator_floor_ha", np.nan)),
            args.perturbative_denominator_floor,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("The requested ranking floor differs from the freeze.")
        random = support.get("random_control", {})
        if int(random.get("seed", -1)) != args.random_support_seed:
            raise RuntimeError("The requested random seed differs from the freeze.")
        if int(random.get("path_bank_size", -1)) != args.random_path_bank_size:
            raise RuntimeError("The requested random bank size differs from the freeze.")
        if int(random.get("number_of_paths_evaluated_at_selected_k", -1)) != (
            args.random_controls
        ):
            raise RuntimeError("The requested random-control count differs from the freeze.")
        return support
    support = build_support_payload(
        inputs,
        pool,
        internal_ids,
        compact_ids,
        operator_rows,
        args,
    )
    write_json(path, support)
    return support


def validate_current_operator_scores(
    support: dict[str, Any],
    current_rows: Sequence[dict[str, Any]],
    tolerance: float,
) -> None:
    frozen_by_id = {
        int(row["global_uccsd_index"]): row
        for row in support.get("operator_score_rows", [])
    }
    current_by_id = {
        int(row["global_uccsd_index"]): row for row in current_rows
    }
    if set(frozen_by_id) != set(current_by_id) or len(frozen_by_id) != EXPANDED_FULL_POOL:
        raise RuntimeError("The frozen and current expanded operator pools differ.")
    numeric_fields = (
        "absolute_hamiltonian_coupling_ha",
        "signed_diagonal_gap_ha",
        "absolute_denominator_ha",
        "en_amplitude_score",
        "en_second_order_lowering_score_ha",
    )
    maximum = 0.0
    for global_id in sorted(frozen_by_id):
        frozen = frozen_by_id[global_id]
        current = current_by_id[global_id]
        for field in (
            "operator_type",
            "internal_2e5o_operator",
            "transferred_compact10_operator",
            "new_external_operator",
            "directly_excited_determinant_index",
        ):
            if frozen.get(field) != current.get(field):
                raise RuntimeError(
                    f"Current operator metadata changed for E{global_id}: {field}."
                )
        for field in numeric_fields:
            residual = abs(float(frozen[field]) - float(current[field]))
            maximum = max(maximum, residual)
    if maximum > tolerance:
        raise RuntimeError(
            "The current expanded Hamiltonian does not reproduce the frozen "
            f"operator scores: {maximum:.3e}."
        )


def protocol_payload(support: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_phase": ANALYSIS_PHASE,
        "phase7c_support_fingerprint": support["phase7c_support_fingerprint"],
        "simulator": (
            "100-determinant Nalpha=1,Nbeta=1 ordered product of exact "
            "fermionic excitation rotations"
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
        "deterministic_path": (
            "transferred compact10 plus every prefix k=0..75 of the frozen "
            "external ranking, stopping after the first stable all-geometry "
            "guard-band candidate or after k=75"
        ),
        "primary_candidate_rule": (
            "smallest k whose best variational error is <=0.0016 Ha and whose "
            "restart analysis is stable at all three fixed geometries"
        ),
        "guarded_confirmation_candidate_rule": (
            "smallest k whose best variational error is <=0.0008 Ha and whose "
            "restart analysis is stable at all three fixed geometries; fall "
            "back to the primary candidate if no guarded candidate exists"
        ),
        "random_control_rule": (
            "Evaluate the first 20 unique non-deterministic composition-matched "
            "supports from the frozen path bank at the confirmation-candidate k; "
            "fill from frozen mixed permutations if the type-matched support "
            "space is too small; random comparison is inapplicable at k=75"
        ),
        "thresholds_ha": {
            "chemical_accuracy": CHEMICAL_ACCURACY_HA,
            "guard_band": GUARD_BAND_HA,
            "strict_equivalence": STRICT_EQUIVALENCE_HA,
        },
        "post_freeze_audits": [
            "full 100x100 eigenspectrum minimum matches accepted Phase-7A.2 2e10o CASCI",
            "internal 25x25 block minimum matches accepted rotated 2e5o CASCI",
            "compact10 k=0 variational minimum matches the internal exact energy",
            "full99 product-UCC control is evaluated",
        ],
        "execution_controls_excluded_from_protocol_identity": [
            "run_group",
            "resume",
            "freeze_only",
            "output_dir",
        ],
        "phase7d_required": True,
    }
    payload["phase7c_protocol_fingerprint"] = p7.fingerprint(payload)
    return payload


def selected_ids_for_k(
    support: dict[str, Any],
    external_k: int,
) -> tuple[int, ...]:
    compact = tuple(int(value) for value in support["transferred_compact10_global_indices"])
    ranking = tuple(int(value) for value in support["external_ranking_global_indices"])
    if external_k < 0 or external_k > len(ranking):
        raise ValueError("external_k lies outside the frozen schedule.")
    return tuple(sorted((*compact, *ranking[:external_k])))


def frozen_random_supports_at_k(
    support: dict[str, Any],
    pool: Sequence[Excitation],
    external_k: int,
) -> list[tuple[str, tuple[int, ...]]]:
    deterministic_external = tuple(
        int(value) for value in support["external_ranking_global_indices"][:external_k]
    )
    single_count = sum(
        excitation_type(pool[index]) == "single" for index in deterministic_external
    )
    double_count = external_k - single_count
    compact = tuple(int(value) for value in support["transferred_compact10_global_indices"])
    deterministic = tuple(sorted((*compact, *deterministic_external)))
    requested = int(
        support["random_control"]["number_of_paths_evaluated_at_selected_k"]
    )
    if external_k == EXTERNAL_POOL:
        return []
    selected: list[tuple[str, tuple[int, ...]]] = []
    seen = {deterministic}
    for path in support["random_control"]["paths"]:
        external = (
            tuple(int(value) for value in path["external_single_order"][:single_count])
            + tuple(int(value) for value in path["external_double_order"][:double_count])
        )
        ids = tuple(sorted((*compact, *external)))
        if ids in seen:
            continue
        seen.add(ids)
        selected.append((f"{path['path_id']}_composition", ids))
        if len(selected) == requested:
            return selected
    for path in support["random_control"]["paths"]:
        external = tuple(
            int(value) for value in path["external_mixed_order"][:external_k]
        )
        ids = tuple(sorted((*compact, *external)))
        if ids in seen:
            continue
        seen.add(ids)
        selected.append((f"{path['path_id']}_mixed_fallback", ids))
        if len(selected) == requested:
            return selected
    if len(selected) != requested:
        raise RuntimeError(
            "The frozen random-path bank cannot supply enough unique controls "
            f"at k={external_k}. Use a new output directory with a larger bank."
        )
    return selected


def load_checkpoint(path: Path, protocol_fingerprint: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(
        path,
        dtype={
            "run_id": str,
            "phase7c_protocol_fingerprint": str,
            "phase7c_support_fingerprint": str,
            "family": str,
            "support_name": str,
            "geometry_key": str,
            "selected_global_uccsd_indices_json": str,
            "restart_name": str,
            "parameters_json": str,
        },
    )
    if frame.empty:
        return []
    if set(frame["phase7c_protocol_fingerprint"].astype(str)) != {
        protocol_fingerprint
    }:
        raise RuntimeError("The existing checkpoint belongs to another protocol.")
    return frame.to_dict(orient="records")


def parameter_map(ids: Sequence[int], parameters: Sequence[float]) -> dict[int, float]:
    return {int(index): float(value) for index, value in zip(ids, parameters)}


def vector_from_map(ids: Sequence[int], values: dict[int, float]) -> np.ndarray:
    return np.asarray([float(values.get(int(index), 0.0)) for index in ids])


def optimize_once(
    system: ExpandedSystem,
    pool: Sequence[Excitation],
    selected_ids: Sequence[int],
    initial: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    generators = [generator_for_excitation(index, pool[index]) for index in selected_ids]

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return energy_and_gradient(values, system.hamiltonian_total, generators)

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
    return {
        "success": bool(result.success) and np.isfinite(energy),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "energy_total_ha": float(energy),
        "gradient_infinity_norm": float(np.max(np.abs(gradient))) if len(gradient) else 0.0,
        "parameters_json": json.dumps([float(value) for value in result.x]),
    }


def existing_run(
    rows: Sequence[dict[str, Any]],
    run_id: str,
) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("run_id")) == run_id]
    if len(matches) > 1:
        raise RuntimeError(f"Checkpoint contains duplicate run_id {run_id}.")
    return matches[0] if matches else None


def run_restart(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: dict[str, Any],
    system: ExpandedSystem,
    pool: Sequence[Excitation],
    selected_ids: Sequence[int],
    family: str,
    support_name: str,
    external_k: int,
    restart_name: str,
    seed: int,
    initial: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_id = (
        f"{family}|{support_name}|{system.geometry_key}|k{external_k}|{restart_name}"
    )
    prior = existing_run(rows, run_id)
    if prior is not None:
        if args.resume:
            return prior
        raise RuntimeError(f"Run {run_id} already exists; use --resume.")
    result = optimize_once(system, pool, selected_ids, initial, args)
    row = {
        "run_id": run_id,
        "phase7c_protocol_fingerprint": protocol["phase7c_protocol_fingerprint"],
        "phase7c_support_fingerprint": protocol["phase7c_support_fingerprint"],
        "family": family,
        "support_name": support_name,
        "geometry_key": system.geometry_key,
        "bond_length_angstrom": system.bond_length,
        "external_k": external_k,
        "total_operator_count": len(selected_ids),
        "selected_global_uccsd_indices_json": json.dumps(list(selected_ids)),
        "restart_name": restart_name,
        "parameter_seed": seed,
        **result,
    }
    rows.append(row)
    write_csv(checkpoint_path, rows)
    return row


def successful_rows(
    rows: Sequence[dict[str, Any]],
    family: str,
    support_name: str,
    geometry_key_value: str,
    external_k: int,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if (
            str(row.get("family")) == family
            and str(row.get("support_name")) == support_name
            and str(row.get("geometry_key")) == geometry_key_value
            and int(row.get("external_k")) == external_k
            and str(row.get("success")).strip().lower() in {"true", "1"}
            and np.isfinite(float(row.get("energy_total_ha", np.nan)))
        ):
            result.append(row)
    return result


def run_support_at_geometry(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: dict[str, Any],
    system: ExpandedSystem,
    pool: Sequence[Excitation],
    selected_ids: Sequence[int],
    family: str,
    support_name: str,
    external_k: int,
    continuation: dict[int, float] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    seed_base = (
        args.parameter_seed
        + int(round(system.bond_length * 1000)) * 100000
        + external_k * 1000
        + sum(ord(value) for value in support_name)
    )
    if continuation is None:
        first = np.zeros(len(selected_ids), dtype=float)
        first_name = "zero"
    else:
        first = vector_from_map(selected_ids, continuation)
        first_name = "continuation"
    run_restart(
        rows,
        checkpoint_path,
        protocol,
        system,
        pool,
        selected_ids,
        family,
        support_name,
        external_k,
        first_name,
        seed_base,
        first,
        args,
    )
    rng = np.random.default_rng(seed_base + 1)
    run_restart(
        rows,
        checkpoint_path,
        protocol,
        system,
        pool,
        selected_ids,
        family,
        support_name,
        external_k,
        "seeded_random_0",
        seed_base + 1,
        rng.normal(scale=args.initial_scale, size=len(selected_ids)),
        args,
    )
    current = successful_rows(
        rows, family, support_name, system.geometry_key, external_k
    )
    spread = (
        max(float(row["energy_total_ha"]) for row in current)
        - min(float(row["energy_total_ha"]) for row in current)
        if len(current) >= 2
        else float("inf")
    )
    needs_rescue = len(current) < 2 or spread > args.disagreement_tolerance
    if needs_rescue:
        for rescue in range(args.rescue_restarts):
            seed = seed_base + 100 + rescue
            rescue_rng = np.random.default_rng(seed)
            run_restart(
                rows,
                checkpoint_path,
                protocol,
                system,
                pool,
                selected_ids,
                family,
                support_name,
                external_k,
                f"rescue_{rescue}",
                seed,
                rescue_rng.normal(scale=args.initial_scale, size=len(selected_ids)),
                args,
            )
    return successful_rows(rows, family, support_name, system.geometry_key, external_k)


def summarize_support_rows(
    rows: Sequence[dict[str, Any]],
    system: ExpandedSystem,
    family: str,
    support_name: str,
    external_k: int,
    disagreement_tolerance: float,
) -> dict[str, Any]:
    successful = successful_rows(
        rows, family, support_name, system.geometry_key, external_k
    )
    energies = [float(row["energy_total_ha"]) for row in successful]
    best = min(energies) if energies else float("nan")
    spread = max(energies) - min(energies) if len(energies) >= 2 else float("nan")
    stable = bool(
        len(energies) >= 2
        and np.isfinite(spread)
        and spread <= disagreement_tolerance
    )
    error = best - system.accepted_full_exact_total_ha if np.isfinite(best) else float("nan")
    return {
        "family": family,
        "support_name": support_name,
        "geometry_key": system.geometry_key,
        "bond_length_angstrom": system.bond_length,
        "external_k": external_k,
        "total_operator_count": COMPACT_POOL + external_k if family != "full99_control" else EXPANDED_FULL_POOL,
        "successful_restarts": len(energies),
        "analysis_stable": stable,
        "best_variational_total_ha": best,
        "restart_energy_spread_ha": spread,
        "accepted_full_2e10o_exact_total_ha": system.accepted_full_exact_total_ha,
        "best_error_vs_full_2e10o_exact_ha": error,
        "chemical_accuracy": bool(stable and error <= CHEMICAL_ACCURACY_HA),
        "guard_band_accuracy": bool(stable and error <= GUARD_BAND_HA),
        "variational_bound_respected": bool(
            np.isfinite(error) and error >= -STRICT_EQUIVALENCE_HA
        ),
    }


def run_deterministic_path(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: dict[str, Any],
    support: dict[str, Any],
    systems: dict[str, ExpandedSystem],
    pool: Sequence[Excitation],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    continuations: dict[str, dict[int, float] | None] = {
        key: None for key in systems
    }
    for external_k in range(EXTERNAL_POOL + 1):
        current_k: list[dict[str, Any]] = []
        for key in sorted(systems, key=float):
            system = systems[key]
            ids = selected_ids_for_k(support, external_k)
            completed = run_support_at_geometry(
                rows,
                checkpoint_path,
                protocol,
                system,
                pool,
                ids,
                "deterministic_path",
                f"compact10_plus_top_{external_k:02d}",
                external_k,
                continuations[key],
                args,
            )
            if completed:
                best_row = min(completed, key=lambda row: float(row["energy_total_ha"]))
                parameters = json.loads(str(best_row["parameters_json"]))
                continuations[key] = parameter_map(ids, parameters)
            summary = summarize_support_rows(
                rows,
                system,
                "deterministic_path",
                f"compact10_plus_top_{external_k:02d}",
                external_k,
                args.disagreement_tolerance,
            )
            summaries.append(summary)
            current_k.append(summary)
        worst = max(
            float(row["best_error_vs_full_2e10o_exact_ha"])
            for row in current_k
        )
        all_stable = all(bool(row["analysis_stable"]) for row in current_k)
        print(
            f"Completed nested Phase-7C k={external_k}: "
            f"worst error={worst:.6e} Ha, stable={all_stable}.",
            flush=True,
        )
        if all_stable and worst <= GUARD_BAND_HA:
            print(
                "Preregistered guarded-candidate stopping rule reached.",
                flush=True,
            )
            break
    return summaries


def decision_from_deterministic(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_k: list[dict[str, Any]] = []
    for external_k in range(EXTERNAL_POOL + 1):
        cohort = [row for row in summaries if int(row["external_k"]) == external_k]
        if len(cohort) != len(EXPECTED_BOND_LENGTHS):
            continue
        stable = all(bool(row["analysis_stable"]) for row in cohort)
        errors = [float(row["best_error_vs_full_2e10o_exact_ha"]) for row in cohort]
        worst = max(errors) if all(np.isfinite(errors)) else float("nan")
        by_k.append(
            {
                "external_k": external_k,
                "total_operator_count": COMPACT_POOL + external_k,
                "all_geometries_stable": stable,
                "worst_geometry_error_ha": worst,
                "all_geometries_chemical": bool(
                    stable and np.isfinite(worst) and worst <= CHEMICAL_ACCURACY_HA
                ),
                "all_geometries_guarded": bool(
                    stable and np.isfinite(worst) and worst <= GUARD_BAND_HA
                ),
            }
        )
    chemical = next((row for row in by_k if row["all_geometries_chemical"]), None)
    guarded = next((row for row in by_k if row["all_geometries_guarded"]), None)
    confirmation = guarded or chemical
    return {
        "path_summary": by_k,
        "primary_minimal_chemical_external_k": (
            None if chemical is None else int(chemical["external_k"])
        ),
        "guarded_external_k": None if guarded is None else int(guarded["external_k"]),
        "phase7d_confirmation_external_k": (
            None if confirmation is None else int(confirmation["external_k"])
        ),
        "guarded_candidate_available": guarded is not None,
    }


def run_random_controls(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: dict[str, Any],
    support: dict[str, Any],
    systems: dict[str, ExpandedSystem],
    pool: Sequence[Excitation],
    external_k: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    variants = frozen_random_supports_at_k(support, pool, external_k)
    summaries: list[dict[str, Any]] = []
    for path_id, ids in variants:
        for key in sorted(systems, key=float):
            system = systems[key]
            run_support_at_geometry(
                rows,
                checkpoint_path,
                protocol,
                system,
                pool,
                ids,
                "random_control",
                path_id,
                external_k,
                None,
                args,
            )
            summaries.append(
                summarize_support_rows(
                    rows,
                    system,
                    "random_control",
                    path_id,
                    external_k,
                    args.disagreement_tolerance,
                )
            )
        print(f"Completed Phase-7C random control {path_id}.", flush=True)
    return summaries


def run_full99_control(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    protocol: dict[str, Any],
    systems: dict[str, ExpandedSystem],
    pool: Sequence[Excitation],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    ids = tuple(range(EXPANDED_FULL_POOL))
    result: list[dict[str, Any]] = []
    for key in sorted(systems, key=float):
        system = systems[key]
        run_support_at_geometry(
            rows,
            checkpoint_path,
            protocol,
            system,
            pool,
            ids,
            "full99_control",
            "full99",
            EXTERNAL_POOL,
            None,
            args,
        )
        result.append(
            summarize_support_rows(
                rows,
                system,
                "full99_control",
                "full99",
                EXTERNAL_POOL,
                args.disagreement_tolerance,
            )
        )
    return result


def exact_audit_rows(
    systems: dict[str, ExpandedSystem],
    audit_tolerance: float,
) -> list[dict[str, Any]]:
    internal = internal_determinant_indices()
    rows: list[dict[str, Any]] = []
    for key in sorted(systems, key=float):
        system = systems[key]
        full_exact = float(np.linalg.eigvalsh(system.hamiltonian_total)[0])
        internal_exact = float(
            np.linalg.eigvalsh(
                system.hamiltonian_total[np.ix_(internal, internal)]
            )[0]
        )
        full_residual = full_exact - system.accepted_full_exact_total_ha
        internal_residual = internal_exact - system.accepted_internal_exact_total_ha
        passed = bool(
            abs(full_residual) <= audit_tolerance
            and abs(internal_residual) <= audit_tolerance
        )
        rows.append(
            {
                "geometry_key": key,
                "bond_length_angstrom": system.bond_length,
                "computed_full_2e10o_exact_total_ha": full_exact,
                "accepted_full_2e10o_exact_total_ha": system.accepted_full_exact_total_ha,
                "full_exact_residual_ha": full_residual,
                "computed_internal_rotated_2e5o_exact_total_ha": internal_exact,
                "accepted_internal_rotated_2e5o_exact_total_ha": system.accepted_internal_exact_total_ha,
                "internal_exact_residual_ha": internal_residual,
                "exact_hamiltonian_audit_passed": passed,
            }
        )
    if not all(row["exact_hamiltonian_audit_passed"] for row in rows):
        raise RuntimeError("The post-freeze expanded Hamiltonian exact audit failed.")
    return rows


def random_path_summary(
    random_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = sorted({str(row["support_name"]) for row in random_rows})
    result: list[dict[str, Any]] = []
    for name in names:
        cohort = [row for row in random_rows if str(row["support_name"]) == name]
        errors = [float(row["best_error_vs_full_2e10o_exact_ha"]) for row in cohort]
        stable = len(cohort) == len(EXPECTED_BOND_LENGTHS) and all(
            bool(row["analysis_stable"]) for row in cohort
        )
        worst = max(errors) if all(np.isfinite(errors)) else float("nan")
        result.append(
            {
                "support_name": name,
                "external_k": int(cohort[0]["external_k"]) if cohort else None,
                "all_geometries_stable": stable,
                "worst_geometry_error_ha": worst,
                "all_geometries_chemical": bool(
                    stable and np.isfinite(worst) and worst <= CHEMICAL_ACCURACY_HA
                ),
                "all_geometries_guarded": bool(
                    stable and np.isfinite(worst) and worst <= GUARD_BAND_HA
                ),
            }
        )
    return result


def make_conclusions(
    protocol: dict[str, Any],
    support: dict[str, Any],
    exact_rows: Sequence[dict[str, Any]],
    deterministic_rows: Sequence[dict[str, Any]],
    decision: dict[str, Any],
    random_rows: Sequence[dict[str, Any]],
    full99_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    confirmation_k = decision["phase7d_confirmation_external_k"]
    evaluated_k = sorted({int(row["external_k"]) for row in deterministic_rows})
    complete_cohorts = bool(
        evaluated_k
        and all(
            sum(int(row["external_k"]) == external_k for row in deterministic_rows)
            == len(EXPECTED_BOND_LENGTHS)
            for external_k in evaluated_k
        )
    )
    deterministic_complete = bool(
        complete_cohorts
        and (
            decision["guarded_external_k"] is not None
            or max(evaluated_k) == EXTERNAL_POOL
        )
    )
    random_summary = random_path_summary(random_rows) if random_rows else []
    random_applicable = confirmation_k != EXTERNAL_POOL
    expected_random = (
        int(support["random_control"]["number_of_paths_evaluated_at_selected_k"])
        if random_applicable
        else 0
    )
    random_complete = len(random_summary) == expected_random
    full99_complete = len(full99_rows) == len(EXPECTED_BOND_LENGTHS)
    compact_rows = [row for row in deterministic_rows if int(row["external_k"]) == 0]
    compact_internal_residuals = []
    for row in compact_rows:
        exact = next(
            item
            for item in exact_rows
            if str(item["geometry_key"]) == str(row["geometry_key"])
        )
        compact_internal_residuals.append(
            float(row["best_variational_total_ha"])
            - float(exact["computed_internal_rotated_2e5o_exact_total_ha"])
        )
    compact_simulator_audit = bool(
        len(compact_internal_residuals) == len(EXPECTED_BOND_LENGTHS)
        and all(abs(value) <= STRICT_EQUIVALENCE_HA for value in compact_internal_residuals)
    )
    confirmation_rows = (
        [row for row in deterministic_rows if int(row["external_k"]) == confirmation_k]
        if confirmation_k is not None
        else []
    )
    candidate_worst = (
        max(float(row["best_error_vs_full_2e10o_exact_ha"]) for row in confirmation_rows)
        if confirmation_rows
        else None
    )
    random_stable = [row for row in random_summary if row["all_geometries_stable"]]
    candidate_beats_random = (
        sum(
            candidate_worst < float(row["worst_geometry_error_ha"])
            for row in random_stable
        )
        if candidate_worst is not None
        else 0
    )
    execution_complete = bool(
        deterministic_complete
        and confirmation_k is not None
        and random_complete
        and full99_complete
        and compact_simulator_audit
        and all(row["exact_hamiltonian_audit_passed"] for row in exact_rows)
    )
    if execution_complete:
        status = "complete_candidate_frozen_for_phase7d_confirmation"
    elif deterministic_complete and confirmation_k is None:
        status = "complete_no_chemical_augmentation_found"
    else:
        status = "partial_protocol_execution"
    confirmation_ids = (
        list(selected_ids_for_k(support, int(confirmation_k)))
        if confirmation_k is not None
        else []
    )
    return {
        "analysis_status": status,
        "execution_complete": execution_complete,
        "phase7d_required": True,
        "phase7c_protocol_fingerprint": protocol["phase7c_protocol_fingerprint"],
        "phase7c_support_fingerprint": support["phase7c_support_fingerprint"],
        "phase7b_protocol_fingerprint": support["phase7b_protocol_fingerprint"],
        "phase7b_support_fingerprint": support["phase7b_support_fingerprint"],
        "primary_minimal_chemical_external_k": decision[
            "primary_minimal_chemical_external_k"
        ],
        "guarded_external_k": decision["guarded_external_k"],
        "phase7d_confirmation_external_k": confirmation_k,
        "phase7d_confirmation_total_operator_count": (
            None if confirmation_k is None else COMPACT_POOL + int(confirmation_k)
        ),
        "phase7d_confirmation_global_uccsd_indices": confirmation_ids,
        "phase7d_confirmation_operator_labels": [
            item["operator_label"]
            for item in support["expanded_pool"]
            if int(item["global_uccsd_index"]) in set(confirmation_ids)
        ],
        "confirmation_candidate_worst_geometry_error_ha": candidate_worst,
        "exact_hamiltonian_audit_passed": all(
            row["exact_hamiltonian_audit_passed"] for row in exact_rows
        ),
        "compact10_simulator_reproduces_internal_exact": compact_simulator_audit,
        "compact10_internal_exact_residuals_ha": compact_internal_residuals,
        "full99_control_all_stable": bool(
            full99_complete and all(row["analysis_stable"] for row in full99_rows)
        ),
        "full99_control_worst_exact_error_ha": (
            max(float(row["best_error_vs_full_2e10o_exact_ha"]) for row in full99_rows)
            if full99_rows
            else None
        ),
        "random_control": {
            "applicable": random_applicable,
            "inapplicability_reason": (
                None
                if random_applicable
                else "At k=75 every compact-preserving external support is identical."
            ),
            "execution_complete": random_complete,
            "number_stable_all_geometries": len(random_stable),
            "number_chemical_all_geometries": sum(
                bool(row["all_geometries_chemical"]) for row in random_summary
            ),
            "number_guarded_all_geometries": sum(
                bool(row["all_geometries_guarded"]) for row in random_summary
            ),
            "candidate_beats_random_count_on_worst_geometry_error": candidate_beats_random,
            "candidate_empirical_percentile_among_stable_random_supports": (
                candidate_beats_random / len(random_stable) if random_stable else None
            ),
        },
        "claim_boundary": (
            "Phase 7C is a determinant-simulator discovery result for frozen "
            "LiH/6-31G 2e,10o Hamiltonians. The selected support is a hypothesis, "
            "not a confirmed production-Qiskit VQE result. Phase 7D must be "
            "preregistered and frozen before confirmation."
        ),
    }


def write_report(
    path: Path,
    conclusions: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# LiH Phase 7C Expanded-Space Augmentation Discovery",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        f"Minimal chemical external k: "
        f"**{conclusions['primary_minimal_chemical_external_k']}**",
        "",
        f"Guarded external k: **{conclusions['guarded_external_k']}**",
        "",
        f"Phase-7D confirmation external k: "
        f"**{conclusions['phase7d_confirmation_external_k']}**",
        "",
        "## Frozen decision rule",
        "",
        "The primary candidate is the first stable nested support below 0.0016 Ha "
        "at all three geometries. The confirmation candidate is the first stable "
        "support below 0.0008 Ha, with fallback to the primary candidate.",
        "",
        "## Deterministic path",
        "",
        "| External k | Total operators | Stable at all geometries | Worst error (Ha) | Chemical | Guarded |",
        "|---:|---:|:---:|---:|:---:|:---:|",
    ]
    for row in decision["path_summary"]:
        lines.append(
            f"| {row['external_k']} | {row['total_operator_count']} | "
            f"{row['all_geometries_stable']} | "
            f"{float(row['worst_geometry_error_ha']):.10f} | "
            f"{row['all_geometries_chemical']} | {row['all_geometries_guarded']} |"
        )
    lines.extend(
        [
            "",
            "## Audits and controls",
            "",
            f"Exact Hamiltonian audit passed: "
            f"**{conclusions['exact_hamiltonian_audit_passed']}**.",
            "",
            f"compact10 reproduces the internal exact result: "
            f"**{conclusions['compact10_simulator_reproduces_internal_exact']}**.",
            "",
            f"Full99 worst exact error: "
            f"**{conclusions['full99_control_worst_exact_error_ha']} Ha**.",
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
            "Discover a frozen compact10 external-correlation augmentation in "
            "the accepted LiH/6-31G 2e,10o Hamiltonian."
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
        "--phase7b-dir", type=Path, default=Path("lih_phase7b_support_transfer_ec65")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("lih_phase7c_expanded_augmentation")
    )
    parser.add_argument(
        "--run-group",
        choices=["deterministic", "random", "all"],
        default="all",
        help="Execution staging only; all discovery paths are frozen regardless.",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-10)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--parameter-seed", type=int, default=20260813)
    parser.add_argument("--rescue-restarts", type=int, default=2)
    parser.add_argument("--disagreement-tolerance", type=float, default=1e-7)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--rotation-reconstruction-tolerance", type=float, default=1e-10
    )
    parser.add_argument("--perturbative-denominator-floor", type=float, default=1e-8)
    parser.add_argument("--random-support-seed", type=int, default=314159)
    parser.add_argument("--random-path-bank-size", type=int, default=100)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Freeze support and protocol before any exact or variational evaluation.",
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
    if args.random_controls < 1 or args.random_path_bank_size < args.random_controls:
        raise ValueError("The random-path bank must cover all requested controls.")
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

    inputs = load_inputs(args.phase7a2_dir, args.phase7b_dir)
    bundle = p7a2.load_inputs(args.phase7a_dir, args.phase7a1_dir)
    print("Reconstructing the accepted Phase-7A.2 physical orbitals...", flush=True)
    frozen, reconstruction_rows = p7a2.reconstruct_frozen_geometries(
        bundle, args.rotation_reconstruction_tolerance
    )
    accepted_rows = p7b.reconcile_with_accepted_physical_orbitals(
        frozen,
        inputs.phase7a2.phase7a2_frozen_support,
        args.rotation_reconstruction_tolerance,
    )
    accepted_by_key = {str(row["geometry_key"]): row for row in accepted_rows}
    write_csv(
        args.output_dir / "phase7c_rotation_reconstruction.csv",
        [
            {**row, **accepted_by_key[str(row["geometry_key"])]}
            for row in reconstruction_rows
        ],
    )

    pool, internal_ids, compact_ids, _ = pool_identities()
    validate_phase7b_compact_identity(inputs, compact_ids)
    verify_simulator_gradient(pool)

    print("Building expanded 2e,10o determinant Hamiltonians without diagonalization...", flush=True)
    systems: dict[str, ExpandedSystem] = {}
    construction_rows: list[dict[str, Any]] = []
    for key in sorted(frozen, key=float):
        system = build_expanded_system(
            frozen[key],
            inputs.phase7a2.audits_by_geometry[key],
            args.audit_tolerance,
        )
        systems[key] = system
        construction_rows.append(
            {
                "geometry_key": key,
                "bond_length_angstrom": system.bond_length,
                "determinant_dimension": EXPANDED_DETERMINANTS,
                "determinant_addresses_ordered": system.determinant_addresses_ordered,
                "hamiltonian_hermiticity_residual": system.hermiticity_residual,
                "accepted_full_exact_reference_input_ha": (
                    system.accepted_full_exact_total_ha
                ),
                "exact_diagonalization_performed_during_freeze": False,
            }
        )
    write_csv(
        args.output_dir / "phase7c_hamiltonian_construction.csv",
        construction_rows,
    )

    equilibrium = systems[geometry_key(REFERENCE_BOND_LENGTH)]
    operator_rows = rank_external_operators(
        equilibrium,
        pool,
        set(internal_ids),
        set(compact_ids),
        args.perturbative_denominator_floor,
    )
    support_path = args.output_dir / "phase7c_frozen_discovery_support.json"
    support = load_or_create_support(
        support_path,
        inputs,
        pool,
        internal_ids,
        compact_ids,
        operator_rows,
        args,
    )
    validate_current_operator_scores(support, operator_rows, args.audit_tolerance)
    write_csv(args.output_dir / "phase7c_operator_scores.csv", support["operator_score_rows"])
    protocol = protocol_payload(support, args)
    protocol = write_or_validate_signed_payload(
        args.output_dir / "phase7c_preregistered_protocol.json",
        protocol,
        "phase7c_protocol_fingerprint",
    )
    print(
        "Phase-7C discovery support frozen before exact evaluation: "
        f"{support['phase7c_support_fingerprint']}",
        flush=True,
    )
    print(
        "Phase-7C protocol fingerprint: "
        f"{protocol['phase7c_protocol_fingerprint']}",
        flush=True,
    )
    if args.freeze_only:
        print("Freeze-only gate complete; no diagonalization or optimization performed.")
        return 0

    print("Running post-freeze exact Hamiltonian audits...", flush=True)
    exact_rows = exact_audit_rows(systems, args.audit_tolerance)
    write_csv(args.output_dir / "phase7c_exact_hamiltonian_audit.csv", exact_rows)
    checkpoint_path = args.output_dir / "phase7c_restart_runs.csv"
    runs = load_checkpoint(
        checkpoint_path, protocol["phase7c_protocol_fingerprint"]
    )

    deterministic_path = args.output_dir / "phase7c_deterministic_path_summary.csv"
    deterministic_rows: list[dict[str, Any]] = []
    if args.run_group in {"deterministic", "all"}:
        print("Evaluating the frozen nested deterministic path...", flush=True)
        deterministic_rows = run_deterministic_path(
            runs,
            checkpoint_path,
            protocol,
            support,
            systems,
            pool,
            args,
        )
        write_csv(deterministic_path, deterministic_rows)
    elif deterministic_path.is_file():
        deterministic_rows = pd.read_csv(
            deterministic_path, dtype={"geometry_key": str}
        ).to_dict(orient="records")
    else:
        raise RuntimeError(
            "Random-only execution requires a completed deterministic path summary."
        )
    decision = decision_from_deterministic(deterministic_rows)
    write_csv(args.output_dir / "phase7c_cardinality_decision_path.csv", decision["path_summary"])
    confirmation_k = decision["phase7d_confirmation_external_k"]

    full99_rows = run_full99_control(
        runs,
        checkpoint_path,
        protocol,
        systems,
        pool,
        args,
    )
    write_csv(args.output_dir / "phase7c_full99_control_summary.csv", full99_rows)

    random_rows: list[dict[str, Any]] = []
    random_path = args.output_dir / "phase7c_random_geometry_summary.csv"
    if confirmation_k is not None and args.run_group in {"random", "all"}:
        print(
            f"Evaluating frozen random controls at external k={confirmation_k}...",
            flush=True,
        )
        random_rows = run_random_controls(
            runs,
            checkpoint_path,
            protocol,
            support,
            systems,
            pool,
            int(confirmation_k),
            args,
        )
        write_csv(random_path, random_rows)
    elif random_path.is_file():
        random_rows = pd.read_csv(random_path).to_dict(orient="records")
    random_summary = random_path_summary(random_rows) if random_rows else []
    if random_summary:
        write_csv(args.output_dir / "phase7c_random_support_summary.csv", random_summary)

    conclusions = make_conclusions(
        protocol,
        support,
        exact_rows,
        deterministic_rows,
        decision,
        random_rows,
        full99_rows,
    )
    write_json(args.output_dir / "phase7c_augmentation_conclusions.json", conclusions)
    write_report(args.output_dir / "PHASE7C_REPORT.md", conclusions, decision)

    print("\nPhase 7C cardinality decision:")
    print(
        pd.DataFrame(decision["path_summary"])[
            [
                "external_k",
                "total_operator_count",
                "all_geometries_stable",
                "worst_geometry_error_ha",
                "all_geometries_chemical",
                "all_geometries_guarded",
            ]
        ].to_string(index=False)
    )
    print("\nPhase 7C conclusions:")
    print(
        json.dumps(
            {
                "analysis_status": conclusions["analysis_status"],
                "execution_complete": conclusions["execution_complete"],
                "primary_minimal_chemical_external_k": conclusions[
                    "primary_minimal_chemical_external_k"
                ],
                "guarded_external_k": conclusions["guarded_external_k"],
                "phase7d_confirmation_external_k": conclusions[
                    "phase7d_confirmation_external_k"
                ],
                "phase7d_confirmation_total_operator_count": conclusions[
                    "phase7d_confirmation_total_operator_count"
                ],
                "phase7c_protocol_fingerprint": conclusions[
                    "phase7c_protocol_fingerprint"
                ],
                "phase7c_support_fingerprint": conclusions[
                    "phase7c_support_fingerprint"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if conclusions["analysis_status"] == "complete_no_chemical_augmentation_found":
        return 2
    if conclusions["execution_complete"] or args.run_group == "deterministic":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
