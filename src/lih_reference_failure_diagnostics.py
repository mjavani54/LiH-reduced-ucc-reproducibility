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
            try:
                selected_ids = json.loads(selected_ids)
            except json.JSONDecodeError:
                selected_ids = ast.literal_eval(selected_ids)
        if isinstance(explicit_excitations, str):
            try:
                explicit_excitations = json.loads(explicit_excitations)
            except json.JSONDecodeError:
                explicit_excitations = ast.literal_eval(explicit_excitations)

        if selected_ids is not None and len(selected_ids) > 0:
            ids = sorted({int(item) for item in selected_ids})
            invalid = [i for i in ids if i < 0 or i >= len(system.candidate_excitations)]
            if invalid:
                raise ValueError(f"Record {row_number} has invalid candidate IDs: {invalid}")
            excitations = [system.candidate_excitations[i] for i in ids]
            if explicit_excitations is not None and len(explicit_excitations) > 0:
                supplied = [canonical_excitation(item) for item in explicit_excitations]
                if supplied != excitations:
                    raise ValueError(
                        f"Record {row_number} has Selected Indices that do not match "
                        "Selected Excitations for this audited candidate pool."
                    )
        elif explicit_excitations is not None and len(explicit_excitations) > 0:
            excitations = [canonical_excitation(item) for item in explicit_excitations]
            lookup = {excitation: i for i, excitation in enumerate(system.candidate_excitations)}
            missing = [item for item in excitations if item not in lookup]
            if missing:
                raise ValueError(
                    f"Record {row_number} contains excitations outside the audited candidate pool: {missing}"
                )
            ids = sorted({lookup[item] for item in excitations})
            excitations = [system.candidate_excitations[i] for i in ids]
        else:
            raise ValueError(
                f"Record {row_number} has no selected excitations. Fill the template before diagnosis."
            )

        phase1_error = record.get(
            "phase1_error_ha",
            record.get("Error vs Exact (Ha)"),
        )
        if phase1_error is not None and not pd.isna(phase1_error):
            phase1_error = float(phase1_error)
        else:
            phase1_error = None

        normalized.append(
            {
                "selection_seed": seed,
                "selected_ids": ids,
                "excitations": excitations,
                "phase1_error_ha": phase1_error,
            }
        )
    return normalized


def custom_excitation_generator(excitations: Sequence[Excitation]) -> Callable[..., list[Excitation]]:
    frozen = list(excitations)

    def generator(
        num_spatial_orbitals: int | None = None,
        num_particles: tuple[int, int] | None = None,
        **_: Any,
    ) -> list[Excitation]:
        del num_spatial_orbitals, num_particles
        return list(frozen)

    return generator


def build_selected_ansatz(system: SystemData, excitations: Sequence[Excitation]) -> Any:
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCC

    if not excitations:
        raise ValueError("A diagnostic ansatz must contain at least one excitation.")
    initial_state = HartreeFock(
        system.problem.num_spatial_orbitals,
        system.problem.num_particles,
        system.mapper,
    )
    ansatz = UCC(
        num_spatial_orbitals=system.problem.num_spatial_orbitals,
        num_particles=system.problem.num_particles,
        excitations=custom_excitation_generator(excitations),
        qubit_mapper=system.mapper,
        preserve_spin=system.config.preserve_spin,
        reps=1,
        initial_state=initial_state,
    )
    # Trigger lazy construction now so malformed excitation lists fail before a long run.
    _ = ansatz.num_parameters
    if ansatz.num_parameters <= 0:
        raise RuntimeError("The selected UCC ansatz has no free parameters.")
    return ansatz


def circuit_resource_metrics(ansatz: Any, seed_transpiler: int) -> dict[str, int | None]:
    from qiskit import transpile

    metrics: dict[str, int | None] = {
        "logical_depth": int(ansatz.decompose(reps=2).depth()),
        "logical_size": int(ansatz.decompose(reps=2).size()),
        "compiled_depth": None,
        "compiled_size": None,
        "compiled_cx": None,
    }
    try:
        compiled = transpile(
            ansatz,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=3,
            seed_transpiler=seed_transpiler,
        )
        counts = compiled.count_ops()
        metrics.update(
            {
                "compiled_depth": int(compiled.depth()),
                "compiled_size": int(compiled.size()),
                "compiled_cx": int(counts.get("cx", 0)),
            }
        )
    except Exception as exc:  # Energy diagnosis remains useful if compilation fails.
        print(f"WARNING: resource transpilation failed: {exc}", file=sys.stderr)
    return metrics


def initial_point_for_restart(
    restart: int,
    num_parameters: int,
    rng: np.random.Generator,
    initial_scale: float,
) -> np.ndarray:
    if restart == 0:
        return np.zeros(num_parameters, dtype=float)
    return rng.uniform(-initial_scale, initial_scale, size=num_parameters)


def run_one_optimization(
    system: SystemData,
    ansatz: Any,
    initial_point: np.ndarray,
    optimizer: str,
    maxiter: int,
    tolerance: float,
) -> dict[str, Any]:
    from qiskit.quantum_info import Statevector

    parameters = list(ansatz.parameters)
    evaluations = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        bound = ansatz.assign_parameters(dict(zip(parameters, theta)), inplace=False)
        state = Statevector.from_instruction(bound)
        active_energy = float(np.real(state.expectation_value(system.qubit_op)))
        return active_energy + system.total_offset_ha

    method = optimizer.upper()
    options: dict[str, Any] = {"maxiter": maxiter}
    bounds = None
    if method in {"SLSQP", "L-BFGS-B"}:
        bounds = [(-math.pi, math.pi)] * len(initial_point)
        options["ftol"] = tolerance
    elif method == "COBYLA":
        options["tol"] = tolerance
        options["catol"] = tolerance
    else:
        raise ValueError("optimizer must be SLSQP, COBYLA, or L-BFGS-B")

    started = time.perf_counter()
    result = minimize(
        objective,
        np.asarray(initial_point, dtype=float),
        method=method,
        bounds=bounds,
        options=options,
        tol=tolerance,
    )
    elapsed = time.perf_counter() - started
    total_energy = float(result.fun)
    error = total_energy - system.exact_total_ha

    return {
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "num_objective_evaluations": int(getattr(result, "nfev", evaluations)),
        "num_iterations": int(getattr(result, "nit", -1)),
        "elapsed_seconds": elapsed,
        "vqe_total_energy_ha": total_energy,
        "signed_error_ha": error,
        "absolute_error_ha": abs(error),
        "variational_violation": bool(error < -1e-8),
        "chemical_accuracy": bool(abs(error) <= CHEMICAL_ACCURACY_HA),
        "good_accuracy": bool(abs(error) <= GOOD_ACCURACY_HA),
        "acceptable_accuracy": bool(abs(error) <= ACCEPTABLE_ACCURACY_HA),
        "optimal_parameters_json": json.dumps(np.asarray(result.x, dtype=float).tolist()),
    }


def checkpoint_runs(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def seed_key(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def diagnose_records(
    system: SystemData,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    restarts: int,
    optimizer: str,
    maxiter: int,
    tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    resume: bool,
) -> dict[str, Any]:
    if restarts < 2:
        raise ValueError("Use at least two restarts; ten is recommended for failure diagnosis.")

    runs_path = output_dir / "diagnostic_restart_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, int]] = set()
    if resume and runs_path.exists():
        previous = pd.read_csv(runs_path)
        rows = previous.to_dict(orient="records")
        matching_previous = previous[
            previous["configuration_fingerprint"].astype(str) == system.fingerprint
        ]
        completed = {
            (seed_key(row["selection_seed"]), int(row["restart"]))
            for row in matching_previous.to_dict(orient="records")
        }
        print(f"Resuming with {len(completed)} completed restart runs.")

    for record_index, record in enumerate(records):
        selection_seed = record["selection_seed"]
        excitations = record["excitations"]
        ansatz = build_selected_ansatz(system, excitations)
        resources = circuit_resource_metrics(ansatz, parameter_seed + record_index)

        for restart in range(restarts):
            key = (seed_key(selection_seed), restart)
            if key in completed:
                continue
            rng = np.random.default_rng(
                parameter_seed + 1_000_003 * record_index + 10_007 * restart
            )
            initial = initial_point_for_restart(
                restart, ansatz.num_parameters, rng, initial_scale
            )
            base = {
                "configuration_fingerprint": system.fingerprint,
                "selection_seed": selection_seed,
                "restart": restart,
                "initialization": "zeros" if restart == 0 else "uniform_random",
                "initialization_seed": (
                    None
                    if restart == 0
                    else parameter_seed + 1_000_003 * record_index + 10_007 * restart
                ),
                "selected_count": len(excitations),
                "selected_ids_json": json.dumps(record["selected_ids"]),
                "num_parameters": int(ansatz.num_parameters),
                "phase1_error_ha": record["phase1_error_ha"],
                "exact_total_energy_ha": system.exact_total_ha,
                "optimizer": optimizer,
                "maxiter": maxiter,
                **resources,
            }
            print(
                f"selection_seed={selection_seed} restart={restart + 1}/{restarts} "
                f"parameters={ansatz.num_parameters}",
                flush=True,
            )
            try:
                outcome = run_one_optimization(
                    system,
                    ansatz,
                    initial,
                    optimizer,
                    maxiter,
                    tolerance,
                )
                row = {**base, **outcome, "run_exception": None}
            except Exception as exc:
                row = {
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
                print(f"ERROR: {exc}", file=sys.stderr)
            rows.append(row)
            completed.add(key)
            checkpoint_runs(runs_path, rows)

    runs = pd.DataFrame(rows)
    current = runs[runs["configuration_fingerprint"] == system.fingerprint].copy()
    wanted_seeds = {seed_key(record["selection_seed"]) for record in records}
    current = current[current["selection_seed"].map(seed_key).isin(wanted_seeds)]
    successful = current[np.isfinite(current["absolute_error_ha"])].copy()
    if successful.empty:
        raise RuntimeError("No optimization restart completed successfully.")

    summary_rows: list[dict[str, Any]] = []
    phase1_by_seed = {
        seed_key(record["selection_seed"]): record["phase1_error_ha"] for record in records
    }
    ids_by_seed = {
        seed_key(record["selection_seed"]): record["selected_ids"] for record in records
    }
    for selection_seed, group in successful.groupby("selection_seed", sort=False):
        errors = group["absolute_error_ha"].astype(float)
        phase1_error = phase1_by_seed[seed_key(selection_seed)]
        best = float(errors.min())
        chemical_rate = float((errors <= CHEMICAL_ACCURACY_HA).mean())
        if phase1_error is not None:
            if phase1_error > CHEMICAL_ACCURACY_HA and best <= CHEMICAL_ACCURACY_HA:
                classification = "optimizer_instability_rescued"
            elif best > CHEMICAL_ACCURACY_HA:
                classification = "selection_failure"
            elif phase1_error <= CHEMICAL_ACCURACY_HA and chemical_rate < 0.8:
                classification = "optimizer_sensitive_success"
            else:
                classification = "robust_success"
        else:
            if best > CHEMICAL_ACCURACY_HA:
                classification = "selection_failure"
            elif chemical_rate < 0.8:
                classification = "optimizer_instability"
            else:
                classification = "robust_success"

        summary_rows.append(
            {
                "selection_seed": selection_seed,
                "selected_count": int(group["selected_count"].iloc[0]),
                "selected_ids_json": json.dumps(ids_by_seed[seed_key(selection_seed)]),
                "phase1_error_ha": phase1_error,
                "completed_restarts": int(len(group)),
                "best_error_ha": best,
                "median_error_ha": float(errors.median()),
                "mean_error_ha": float(errors.mean()),
                "std_error_ha": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
                "worst_error_ha": float(errors.max()),
                "chemical_success_rate": chemical_rate,
                "good_success_rate": float((errors <= GOOD_ACCURACY_HA).mean()),
                "acceptable_success_rate": float(
                    (errors <= ACCEPTABLE_ACCURACY_HA).mean()
                ),
                "optimizer_success_rate": float(group["optimizer_success"].astype(bool).mean()),
                "failure_classification": classification,
                "compiled_depth": group["compiled_depth"].iloc[0],
                "compiled_cx": group["compiled_cx"].iloc[0],
                "num_parameters": int(group["num_parameters"].iloc[0]),
            }
        )

    seed_summary = pd.DataFrame(summary_rows)
    seed_summary.to_csv(output_dir / "diagnostic_seed_summary.csv", index=False)
    write_excitation_frequency(records, seed_summary, system, output_dir)
    write_jaccard(records, output_dir)

    counts = seed_summary["failure_classification"].value_counts().to_dict()
    diagnosis = {
        "configuration_fingerprint": system.fingerprint,
        "exact_total_energy_ha": system.exact_total_ha,
        "num_selection_seeds": len(seed_summary),
        "requested_restarts_per_seed": restarts,
        "classification_counts": counts,
        "selection_failure_rate": float(
            (seed_summary["failure_classification"] == "selection_failure").mean()
        ),
        "optimizer_instability_rate": float(
            seed_summary["failure_classification"]
            .isin(
                [
                    "optimizer_instability",
                    "optimizer_instability_rescued",
                    "optimizer_sensitive_success",
                ]
            )
            .mean()
        ),
        "robust_success_rate": float(
            (seed_summary["failure_classification"] == "robust_success").mean()
        ),
        "best_of_restarts_chemical_accuracy_rate": float(
            (seed_summary["best_error_ha"] <= CHEMICAL_ACCURACY_HA).mean()
        ),
        "all_restart_chemical_accuracy_rate": float(
            (successful["absolute_error_ha"] <= CHEMICAL_ACCURACY_HA).mean()
        ),
        "any_variational_violation": bool(successful["variational_violation"].astype(bool).any()),
        "thresholds_ha": {
            "chemical": CHEMICAL_ACCURACY_HA,
            "good": GOOD_ACCURACY_HA,
            "acceptable": ACCEPTABLE_ACCURACY_HA,
        },
        "interpretation_rule": {
            "selection_failure": "No restart reached chemical accuracy.",
            "optimizer_instability": "At least one restart reached chemical accuracy, but fewer than 80% did.",
            "optimizer_instability_rescued": "Phase 1 failed, but a diagnostic restart reached chemical accuracy.",
            "robust_success": "At least 80% of restarts reached chemical accuracy.",
        },
    }
    write_json(output_dir / "diagnostic_overall_summary.json", diagnosis)
    return diagnosis


def write_excitation_frequency(
    records: Sequence[dict[str, Any]],
    seed_summary: pd.DataFrame,
    system: SystemData,
    output_dir: Path,
) -> None:
    summary_lookup = {
        seed_key(row["selection_seed"]): row for _, row in seed_summary.iterrows()
    }
    rows = []
    total = len(records)
    for candidate_id, excitation in enumerate(system.candidate_excitations):
        containing = [record for record in records if candidate_id in record["selected_ids"]]
        best_errors = [
            float(summary_lookup[seed_key(record["selection_seed"])]["best_error_ha"])
            for record in containing
            if seed_key(record["selection_seed"]) in summary_lookup
        ]
        rows.append(
            {
                "candidate_id": candidate_id,
                "excitation_json": json.dumps(excitation_to_json(excitation)),
                "selected_count": len(containing),
                "selection_frequency": len(containing) / total,
                "mean_best_error_when_selected_ha": (
                    float(np.mean(best_errors)) if best_errors else np.nan
                ),
                "median_best_error_when_selected_ha": (
                    float(np.median(best_errors)) if best_errors else np.nan
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "excitation_selection_frequency.csv", index=False)


def write_jaccard(records: Sequence[dict[str, Any]], output_dir: Path) -> None:
    labels = [str(record["selection_seed"]) for record in records]
    matrix = np.zeros((len(records), len(records)), dtype=float)
    sets = [set(record["selected_ids"]) for record in records]
    for i, left in enumerate(sets):
        for j, right in enumerate(sets):
            union = left | right
            matrix[i, j] = 1.0 if not union else len(left & right) / len(union)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "selected_subset_jaccard_matrix.csv", index_label="selection_seed"
    )


def config_from_args(args: argparse.Namespace) -> MolecularConfig:
    atom = f"Li 0 0 0; H 0 0 {args.bond_length}"
    return MolecularConfig(
        atom=atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
        freeze_core=args.freeze_core,
        mapper="JordanWignerMapper",
        excitation_rank=2,
        preserve_spin=args.preserve_spin,
    )


def add_system_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bond-length",
        type=float,
        default=1.595,
        help="Li-H distance in the selected unit (Phase-1 default: 1.595).",
    )
    parser.add_argument("--unit", choices=["ANGSTROM", "BOHR"], default="ANGSTROM")
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0, help="PySCF 2S value (default: singlet 0).")
    parser.add_argument(
        "--freeze-core",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the Li 1s core (default: true).",
    )
    parser.add_argument(
        "--preserve-spin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use spin-preserving double excitations (default: true).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("lih_diagnostics_output"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit LiH reference energies and diagnose selected-UCC VQE failures."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference", help="Run the Hamiltonian/reference audit.")
    add_system_arguments(reference)
    reference.add_argument(
        "--reported-exact",
        type=float,
        action="append",
        default=[],
        help="Historical exact total energy to compare; may be supplied multiple times.",
    )
    reference.add_argument("--audit-tolerance", type=float, default=1e-8)

    template = subparsers.add_parser("template", help="Write candidate pool and selection template.")
    add_system_arguments(template)
    template.add_argument("--num-seeds", type=int, default=20)

    diagnose = subparsers.add_parser("diagnose", help="Run restart-based failure diagnosis.")
    add_system_arguments(diagnose)
    diagnose.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Filled JSON/CSV selection file, including the original "
            "phase1_bayesian_20seed_results.csv format."
        ),
    )
    diagnose.add_argument("--restarts", type=int, default=10)
    diagnose.add_argument(
        "--optimizer", choices=["SLSQP", "COBYLA", "L-BFGS-B"], default="SLSQP"
    )
    diagnose.add_argument("--maxiter", type=int, default=1000)
    diagnose.add_argument("--optimizer-tolerance", type=float, default=1e-9)
    diagnose.add_argument(
        "--initial-scale",
        type=float,
        default=0.1,
        help="Half-width of random UCC parameter perturbations in radians (default: 0.1).",
    )
    diagnose.add_argument("--parameter-seed", type=int, default=20260721)
    diagnose.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed seed/restart pairs from the output CSV (default: true).",
    )

    all_cmd = subparsers.add_parser("all", help="Run audit, then diagnose a filled selection file.")
    add_system_arguments(all_cmd)
    all_cmd.add_argument("--input", type=Path, required=True)
    all_cmd.add_argument("--reported-exact", type=float, action="append", default=[])
    all_cmd.add_argument("--audit-tolerance", type=float, default=1e-8)
    all_cmd.add_argument("--restarts", type=int, default=10)
    all_cmd.add_argument(
        "--optimizer", choices=["SLSQP", "COBYLA", "L-BFGS-B"], default="SLSQP"
    )
    all_cmd.add_argument("--maxiter", type=int, default=1000)
    all_cmd.add_argument("--optimizer-tolerance", type=float, default=1e-9)
    all_cmd.add_argument("--initial-scale", type=float, default=0.1)
    all_cmd.add_argument("--parameter-seed", type=int, default=20260721)
    all_cmd.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def save_reference_outputs(
    system: SystemData,
    output_dir: Path,
    reported_exact: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = validate_reference(system, reported_exact, tolerance)
    write_json(output_dir / "reference_audit.json", audit)
    write_candidate_pool(system, output_dir / "candidate_excitation_pool.csv")
    print(json.dumps(json_ready(audit), indent=2, sort_keys=True))
    if not audit["audit_passed"]:
        raise RuntimeError(
            "Internal reference audit failed. Do not run or interpret VQE diagnostics. "
            f"Inspect {output_dir / 'reference_audit.json'}."
        )
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    print("Building one audited molecular Hamiltonian...", flush=True)
    system = build_system(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command in {"reference", "all"}:
        save_reference_outputs(
            system,
            args.output_dir,
            args.reported_exact,
            args.audit_tolerance,
        )

    if args.command == "template":
        audit = save_reference_outputs(system, args.output_dir, [], 1e-8)
        del audit
        make_selection_template(
            system,
            args.num_seeds,
            args.output_dir / "selected_subsets_template.json",
        )
        print(f"Wrote template to {args.output_dir / 'selected_subsets_template.json'}")

    if args.command in {"diagnose", "all"}:
        # Diagnose also audits internally even when the user did not request an
        # audit printout. This blocks comparisons across mismatched Hamiltonians.
        audit = validate_reference(system, [], 1e-8)
        write_json(args.output_dir / "reference_audit.json", audit)
        write_candidate_pool(system, args.output_dir / "candidate_excitation_pool.csv")
        if not audit["audit_passed"]:
            raise RuntimeError("Reference audit failed; diagnosis aborted.")
        records = read_selection_records(args.input, system)
        diagnosis = diagnose_records(
            system=system,
            records=records,
            output_dir=args.output_dir,
            restarts=args.restarts,
            optimizer=args.optimizer,
            maxiter=args.maxiter,
            tolerance=args.optimizer_tolerance,
            initial_scale=args.initial_scale,
            parameter_seed=args.parameter_seed,
            resume=args.resume,
        )
        print(json.dumps(json_ready(diagnosis), indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed restart runs remain checkpointed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
