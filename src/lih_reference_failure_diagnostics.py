#!/usr/bin/env python3
"""Reference validation and failure diagnosis for selected-excitation LiH VQE.

This program intentionally separates two questions:

1. Are Hartree--Fock, exact, and VQE energies being compared for the same
   Hamiltonian and with the same constant-energy convention?
2. When a selected ansatz misses chemical accuracy, is the selected operator
   subset inadequate, or did continuous parameter optimization fail?

The default system is LiH/STO-3G at 1.595 Angstrom with the Li 1s core frozen,
Jordan--Wigner mapping, and spin-preserving double excitations. All reported
energies are total energies in Hartree unless a field explicitly says
``active_electronic``.

The script targets Qiskit Nature 0.8 and Qiskit Algorithms 0.4. It uses direct
statevector expectation values for the restart study, avoiding version-sensitive
Estimator result formats.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


CHEMICAL_ACCURACY_HA = 0.0016
GOOD_ACCURACY_HA = 0.005
ACCEPTABLE_ACCURACY_HA = 0.010


Excitation = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class MolecularConfig:
    atom: str = "Li 0 0 0; H 0 0 1.595"
    basis: str = "sto3g"
    charge: int = 0
    spin: int = 0  # PySCF convention: 2S = N_alpha - N_beta
    unit: str = "ANGSTROM"
    freeze_core: bool = True
    mapper: str = "JordanWignerMapper"
    excitation_rank: int = 2
    preserve_spin: bool = True


@dataclass
class SystemData:
    config: MolecularConfig
    problem: Any
    mapper: Any
    fermionic_op: Any
    qubit_op: Any
    constant_offsets: dict[str, float]
    total_offset_ha: float
    exact_active_electronic_ha: float
    exact_total_ha: float
    hf_active_electronic_ha: float
    hf_total_from_qubit_ha: float
    hf_total_driver_ha: float
    candidate_excitations: list[Excitation]
    fingerprint: str


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "qiskit": package_version("qiskit"),
        "qiskit-algorithms": package_version("qiskit-algorithms"),
        "qiskit-nature": package_version("qiskit-nature"),
        "pyscf": package_version("pyscf"),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "pandas": package_version("pandas"),
    }


def require_quantum_stack() -> None:
    required = ["qiskit", "qiskit-algorithms", "qiskit-nature", "pyscf"]
    missing = [name for name in required if package_version(name) == "not installed"]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required packages: {joined}. Install the packages from "
            "requirements-lih-diagnostics.txt in the same Python environment "
            "that runs this script."
        )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        if abs(value.imag) > 1e-10:
            return {"real": float(value.real), "imag": float(value.imag)}
        return float(value.real)
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n")


def canonical_excitation(excitation: Any) -> Excitation:
    if not isinstance(excitation, (list, tuple)) or len(excitation) != 2:
        raise ValueError(f"Invalid excitation {excitation!r}; expected [occupied, virtual].")
    occupied, virtual = excitation
    if not isinstance(occupied, (list, tuple)) or not isinstance(virtual, (list, tuple)):
        raise ValueError(f"Invalid excitation {excitation!r}; indices must be sequences.")
    result = (tuple(int(i) for i in occupied), tuple(int(i) for i in virtual))
    if len(result[0]) != len(result[1]) or not result[0]:
        raise ValueError(f"Invalid excitation {excitation!r}; ranks do not match.")
    return result


def excitation_to_json(excitation: Excitation) -> list[list[int]]:
    return [list(excitation[0]), list(excitation[1])]


def make_fingerprint(config: MolecularConfig, candidates: Sequence[Excitation]) -> str:
    payload = {
        "config": asdict(config),
        "candidate_excitations": [excitation_to_json(e) for e in candidates],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def candidate_double_excitations(problem: Any, preserve_spin: bool) -> list[Excitation]:
    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )

    generated = generate_fermionic_excitations(
        num_excitations=2,
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        preserve_spin=preserve_spin,
    )
    return [canonical_excitation(item) for item in generated]


def total_energy_from_result(result: Any) -> float:
    energies = getattr(result, "total_energies", None)
    if energies is None or len(energies) == 0:
        raise RuntimeError("Qiskit Nature did not return a total energy.")
    return float(np.real(energies[0]))


def raw_eigenvalue_from_result(result: Any) -> float:
    raw_result = getattr(result, "raw_result", None)
    eigenvalue = getattr(raw_result, "eigenvalue", None)
    if eigenvalue is None:
        raise RuntimeError("Qiskit Nature did not expose the raw mapped eigenvalue.")
    return float(np.real(eigenvalue))


def build_system(config: MolecularConfig) -> SystemData:
    """Build one Hamiltonian and derive every reference from that object."""
    require_quantum_stack()

    from qiskit.quantum_info import Statevector
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.transformers import FreezeCoreTransformer
    from qiskit_nature.units import DistanceUnit

    if config.mapper != "JordanWignerMapper":
        raise ValueError("This audited version supports only JordanWignerMapper.")
    try:
        unit = getattr(DistanceUnit, config.unit.upper())
    except AttributeError as exc:
        raise ValueError(f"Unsupported distance unit: {config.unit}") from exc

    raw_problem = PySCFDriver(
        atom=config.atom,
        basis=config.basis,
        charge=config.charge,
        spin=config.spin,
        unit=unit,
    ).run()
    problem = FreezeCoreTransformer().transform(raw_problem) if config.freeze_core else raw_problem

    mapper = JordanWignerMapper()
    fermionic_op = problem.hamiltonian.second_q_op()
    qubit_op = mapper.map(fermionic_op)

    constants = {
        str(key): float(np.real(value))
        for key, value in problem.hamiltonian.constants.items()
    }
    total_offset = float(sum(constants.values()))

    # The particle/spin filter is important: an unconstrained minimum over the
    # full Fock space need not be the requested electronic sector.
    exact_solver = NumPyMinimumEigensolver(
        filter_criterion=problem.get_default_filter_criterion()
    )
    exact_result = GroundStateEigensolver(mapper, exact_solver).solve(problem)
    exact_total = total_energy_from_result(exact_result)
    exact_active = raw_eigenvalue_from_result(exact_result)

    hf_circuit = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    )
    hf_state = Statevector.from_instruction(hf_circuit)
    hf_active = float(np.real(hf_state.expectation_value(qubit_op)))
    hf_total_qubit = hf_active + total_offset
    if problem.reference_energy is None:
        raise RuntimeError("The transformed problem has no Hartree--Fock reference energy.")
    hf_total_driver = float(np.real(problem.reference_energy))

    candidates = candidate_double_excitations(problem, config.preserve_spin)
    fingerprint = make_fingerprint(config, candidates)

    return SystemData(
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
        candidate_excitations=candidates,
        fingerprint=fingerprint,
    )


def validate_reference(
    system: SystemData,
    reported_exact_energies: Sequence[float],
    tolerance_ha: float,
) -> dict[str, Any]:
    problem = system.problem
    offset_reconstruction_error = abs(
        (system.exact_active_electronic_ha + system.total_offset_ha)
        - system.exact_total_ha
    )
    hf_reconstruction_error = abs(system.hf_total_from_qubit_ha - system.hf_total_driver_ha)
    variational_gap = system.hf_total_driver_ha - system.exact_total_ha

    checks = {
        "exact_total_equals_active_plus_offsets": offset_reconstruction_error <= tolerance_ha,
        "hf_qubit_equals_driver_reference": hf_reconstruction_error <= tolerance_ha,
        "exact_not_above_hf": variational_gap >= -tolerance_ha,
        "nonempty_candidate_pool": len(system.candidate_excitations) > 0,
        "qubit_count_matches_spin_orbitals": (
            int(system.qubit_op.num_qubits) == int(2 * problem.num_spatial_orbitals)
        ),
    }

    comparisons = []
    for value in reported_exact_energies:
        delta = abs(float(value) - system.exact_total_ha)
        comparisons.append(
            {
                "reported_exact_total_ha": float(value),
                "computed_exact_total_ha": system.exact_total_ha,
                "absolute_difference_ha": delta,
                "matches_within_tolerance": delta <= tolerance_ha,
            }
        )

    matching = [item for item in comparisons if item["matches_within_tolerance"]]
    if not comparisons:
        historical_resolution = "no_reported_references_supplied"
    elif len(matching) == 1:
        historical_resolution = "one_reported_reference_matches"
    elif len(matching) > 1:
        historical_resolution = "multiple_reported_references_match"
    else:
        historical_resolution = "no_reported_reference_matches"

    audit = {
        "audit_passed": bool(all(checks.values())),
        "internal_consistency_checks": checks,
        "historical_reference_resolution": historical_resolution,
        "reported_reference_comparisons": comparisons,
        "configuration": asdict(system.config),
        "configuration_fingerprint": system.fingerprint,
        "software_versions": dependency_versions(),
        "system_dimensions": {
            "num_particles_alpha_beta": list(problem.num_particles),
            "num_active_electrons": int(sum(problem.num_particles)),
            "num_spatial_orbitals": int(problem.num_spatial_orbitals),
            "num_spin_orbitals": int(2 * problem.num_spatial_orbitals),
            "num_qubits": int(system.qubit_op.num_qubits),
            "num_pauli_terms": int(len(system.qubit_op)),
            "num_candidate_doubles": len(system.candidate_excitations),
        },
        "energies_ha": {
            "exact_active_electronic": system.exact_active_electronic_ha,
            "constant_offsets": system.constant_offsets,
            "total_constant_offset": system.total_offset_ha,
            "exact_total": system.exact_total_ha,
            "hf_active_electronic": system.hf_active_electronic_ha,
            "hf_total_from_qubit_expectation": system.hf_total_from_qubit_ha,
            "hf_total_from_driver": system.hf_total_driver_ha,
            "hf_error_vs_exact": variational_gap,
        },
        "numerical_residuals_ha": {
            "exact_offset_reconstruction": offset_reconstruction_error,
            "hf_qubit_vs_driver": hf_reconstruction_error,
        },
        "tolerance_ha": tolerance_ha,
    }
    return audit


def write_candidate_pool(system: SystemData, path: Path) -> None:
    rows = []
    for candidate_id, excitation in enumerate(system.candidate_excitations):
        rows.append(
            {
                "candidate_id": candidate_id,
                "occupied_spin_orbitals": json.dumps(list(excitation[0])),
                "virtual_spin_orbitals": json.dumps(list(excitation[1])),
                "excitation_json": json.dumps(excitation_to_json(excitation)),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def make_selection_template(system: SystemData, num_seeds: int, path: Path) -> None:
    data = {
        "schema_version": 1,
        "configuration_fingerprint": system.fingerprint,
        "instructions": (
            "For each Phase-1 seed, replace selected_ids with candidate_id values "
            "from candidate_excitation_pool.csv and supply phase1_error_ha when available."
        ),
        "records": [
            {
                "selection_seed": seed,
                "selected_ids": [],
                "phase1_error_ha": None,
            }
            for seed in range(num_seeds)
        ],
    }
    write_json(path, data)


def read_selection_records(path: Path, system: SystemData) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            records = payload
            fingerprint = None
        elif isinstance(payload, dict):
            records = payload.get("records")
            fingerprint = payload.get("configuration_fingerprint")
        else:
            raise ValueError("Selection JSON must be a list or an object containing records.")
        if fingerprint is not None and fingerprint != system.fingerprint:
            raise ValueError(
                "Selection file fingerprint does not match the audited Hamiltonian. "
                "Regenerate the template for this configuration; do not reuse candidate IDs."
            )
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        records = frame.to_dict(orient="records")
    else:
        raise ValueError("Selection input must be JSON or CSV.")

    if not isinstance(records, list) or not records:
        raise ValueError("No selection records were found.")

    normalized: list[dict[str, Any]] = []
    seen_seeds: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {row_number} is not an object.")
        seed = record.get(
            "selection_seed",
            record.get("seed", record.get("Seed")),
        )
        if seed is None:
            raise ValueError(f"Record {row_number} is missing selection_seed.")
        seed_key = str(seed)
        if seed_key in seen_seeds:
            raise ValueError(f"Duplicate selection_seed: {seed}")
        seen_seeds.add(seed_key)

        selected_ids = record.get("selected_ids", record.get("Selected Indices"))
        explicit_excitations = record.get(
            "selected_excitations",
            record.get("Selected Excitations"),
        )
        if isinstance(selected_ids, str):
            selected_ids = ast.literal_eval(selected_ids)
        if isinstance(explicit_excitations, str):
            explicit_excitations = ast.literal_eval(explicit_excitations)

        if selected_ids is None and explicit_excitations is None:
            raise ValueError(
                f"Record {row_number} must provide selected_ids or "
                "selected_excitations."
            )

        if selected_ids is not None:
            if not isinstance(selected_ids, (list, tuple)):
                raise ValueError(f"Record {row_number} selected_ids is not a sequence.")
            ids = [int(value) for value in selected_ids]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Record {row_number} has duplicate selected_ids.")
            if any(value < 0 or value >= len(system.candidate_excitations) for value in ids):
                raise ValueError(f"Record {row_number} contains an out-of-range candidate ID.")
            excitations = [system.candidate_excitations[value] for value in ids]
        else:
            if not isinstance(explicit_excitations, (list, tuple)):
                raise ValueError(
                    f"Record {row_number} selected_excitations is not a sequence."
                )
            excitations = [canonical_excitation(value) for value in explicit_excitations]
            try:
                ids = [system.candidate_excitations.index(value) for value in excitations]
            except ValueError as exc:
                raise ValueError(
                    f"Record {row_number} contains an excitation outside the frozen pool."
                ) from exc

        phase1_error = record.get(
            "phase1_error_ha",
            record.get("error_ha", record.get("Error (Ha)")),
        )
        normalized.append(
            {
                "selection_seed": seed,
                "selected_ids": ids,
                "selected_excitations": excitations,
                "phase1_error_ha": (
                    None
                    if phase1_error is None or pd.isna(phase1_error)
                    else float(phase1_error)
                ),
            }
        )
    return normalized


def custom_excitation_generator(
    excitations: Sequence[Excitation],
) -> Callable[[int, tuple[int, int]], list[Excitation]]:
    """Freeze an explicitly ordered excitation list for Qiskit Nature UCC."""
    frozen = tuple(canonical_excitation(value) for value in excitations)

    def generate(
        num_spatial_orbitals: int,
        num_particles: tuple[int, int],
    ) -> list[Excitation]:
        del num_spatial_orbitals, num_particles
        return list(frozen)

    return generate


def build_selected_ansatz(
    system: SystemData,
    excitations: Sequence[Excitation],
) -> Any:
    """Build the frozen one-repetition, ordered UCC product ansatz."""
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

    selected = [canonical_excitation(value) for value in excitations]
    initial_state = HartreeFock(
        system.problem.num_spatial_orbitals,
        system.problem.num_particles,
        system.mapper,
    )
    ansatz = UCC(
        num_spatial_orbitals=system.problem.num_spatial_orbitals,
        num_particles=system.problem.num_particles,
        excitations=custom_excitation_generator(selected),
        qubit_mapper=system.mapper,
        preserve_spin=True,
        reps=1,
        initial_state=initial_state,
    )
    _ = ansatz.num_parameters
    if int(ansatz.num_parameters) != len(selected):
        raise RuntimeError(
            f"Ansatz has {ansatz.num_parameters} parameters; expected {len(selected)}."
        )
    return ansatz


def initial_point_for_restart(
    restart: int,
    num_parameters: int,
    rng: np.random.Generator,
    initial_scale: float,
) -> np.ndarray:
    """Use the zero vector first and deterministic uniform restarts afterward."""
    if restart < 0 or num_parameters < 0 or initial_scale <= 0:
        raise ValueError("Invalid restart initialization request.")
    if restart == 0:
        return np.zeros(num_parameters, dtype=float)
    return rng.uniform(-initial_scale, initial_scale, size=num_parameters)


def circuit_resource_metrics(circuit: Any, seed_transpiler: int) -> dict[str, int]:
    """Return logical and deterministic ``u``/``cx`` transpiled resources."""
    from qiskit import transpile

    compiled = transpile(
        circuit,
        basis_gates=["u", "cx"],
        optimization_level=3,
        seed_transpiler=int(seed_transpiler),
    )
    operations = compiled.count_ops()
    return {
        "logical_depth": int(circuit.depth()),
        "logical_size": int(circuit.size()),
        "compiled_depth": int(compiled.depth()),
        "compiled_size": int(compiled.size()),
        "compiled_cx": int(operations.get("cx", 0)),
    }


def run_one_optimization(
    system: SystemData,
    ansatz: Any,
    initial_point: Sequence[float],
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
) -> dict[str, Any]:
    """Optimize one noiseless statevector restart against the audited Hamiltonian."""
    from qiskit.quantum_info import Statevector

    parameters = list(ansatz.parameters)
    initial = np.asarray(initial_point, dtype=float)
    if len(parameters) != len(initial):
        raise ValueError("Initial-point length does not match the ansatz parameter count.")
    if optimizer not in {"SLSQP", "COBYLA", "L-BFGS-B"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    evaluations = 0

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        bound = ansatz.assign_parameters(
            dict(zip(parameters, np.asarray(values, dtype=float))),
            inplace=False,
        )
        state = Statevector.from_instruction(bound)
        active = float(np.real(state.expectation_value(system.qubit_op)))
        return active + float(system.total_offset_ha)

    if optimizer == "SLSQP":
        options = {"maxiter": int(maxiter), "ftol": float(optimizer_tolerance)}
    elif optimizer == "COBYLA":
        options = {
            "maxiter": int(maxiter),
            "tol": float(optimizer_tolerance),
            "catol": float(optimizer_tolerance),
        }
    else:
        options = {
            "maxiter": int(maxiter),
            "ftol": float(optimizer_tolerance),
            "gtol": float(optimizer_tolerance),
        }

    start = time.perf_counter()
    result = minimize(
        objective,
        initial,
        method=optimizer,
        bounds=[(-math.pi, math.pi)] * len(initial),
        options=options,
        tol=float(optimizer_tolerance),
    )
    elapsed = time.perf_counter() - start
    energy = float(result.fun)
    signed_error = energy - float(system.exact_total_ha)
    absolute_error = abs(signed_error)
    return {
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "num_objective_evaluations": int(getattr(result, "nfev", evaluations)),
        "num_iterations": int(getattr(result, "nit", -1)),
        "elapsed_seconds": float(elapsed),
        "vqe_total_energy_ha": energy,
        "signed_error_ha": signed_error,
        "absolute_error_ha": absolute_error,
        "variational_violation": bool(signed_error < -1e-8),
        "chemical_accuracy": bool(absolute_error <= CHEMICAL_ACCURACY_HA),
        "good_accuracy": bool(absolute_error <= GOOD_ACCURACY_HA),
        "acceptable_accuracy": bool(absolute_error <= ACCEPTABLE_ACCURACY_HA),
        "optimal_parameters_json": json.dumps(
            np.asarray(result.x, dtype=float).tolist()
        ),
    }


# The archived helper supplied for repository assembly ended mid-statement at
# 16 KiB.  The functions above complete the API used by Phases 2B--7B using
# the same Qiskit/SciPy conventions visible in the surviving source.  See
# docs/SOURCE_PROVENANCE.md before treating this helper as a byte-identical
# copy of the original run source.
