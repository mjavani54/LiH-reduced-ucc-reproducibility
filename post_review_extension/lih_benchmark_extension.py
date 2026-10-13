#!/usr/bin/env python3
"""Post-rejection benchmark extension for the LiH reduced-UCC study.

This script does not alter the frozen Phase-7C/7D record.  It adds new,
explicitly labelled analyses requested by peer review:

1. out-of-sample geometries for the frozen supports;
2. an ideal, noiseless fermionic ADAPT-VQE comparator in the same determinant
   sector and with the same ordered excitation factors;
3. an exhaustive Boundary-46 + one-operator control;
4. composition- and spin-partner-closure-matched random 47-operator controls;
5. spin-partner-block augmentation and SCF/ranking sensitivity checks.

The implementation imports the deposited Phase-7 modules so that orbital
construction, excitation indexing, factor conventions, and analytic gradients
remain identical to the archived calculation wherever possible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau


ORIGINAL_BONDS = (1.595, 2.5, 3.0)
HOLDOUT_BONDS = (1.4, 1.8, 2.1, 2.75)
ALL_BONDS = tuple(sorted(set(ORIGINAL_BONDS + HOLDOUT_BONDS)))
CHEMICAL_HA = 1.6e-3
STRICT_HA = 1.0e-6
ENERGY_TARGET_HA = 1.0e-8
GRADIENT_TARGET = 1.0e-7
BASE_SEED = 20261012


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(rows))


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class Modules:
    diag: Any
    p3: Any
    p7: Any
    p7a1: Any
    p7a2: Any
    p7c: Any
    p7d: Any


def import_modules(repository: Path) -> Modules:
    source = repository.resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Repository source directory not found: {source}")
    sys.path.insert(0, str(source))
    import lih_reference_failure_diagnostics as diag
    import lih_phase3_geometry_transfer as p3
    import lih_phase7a_cross_basis_transfer as p7
    import lih_phase7a1_orbital_embedding_diagnostic as p7a1
    import lih_phase7a2_rotated_hamiltonian_audit as p7a2
    import lih_phase7c_expanded_space_augmentation as p7c
    import lih_phase7d_frozen_qiskit_confirmation as p7d

    return Modules(diag, p3, p7, p7a1, p7a2, p7c, p7d)


def geometry_key(value: float) -> str:
    return f"{float(value):.6f}"


def custom_source_snapshot(
    modules: Modules,
    bond: float,
    basis: str,
    active_orbitals: int,
    scf_tolerance: float,
) -> Any:
    from pyscf import gto, scf

    molecule = gto.M(
        atom=f"Li 0 0 0; H 0 0 {bond}",
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    # The managed execution container does not expose /proc/<pid>/statm for
    # nested Python processes.  These LiH integral tensors are tiny, so force
    # the safe in-core path and avoid PySCF's unavailable memory probe.
    molecule.incore_anyway = True
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = float(scf_tolerance)
    mean_field.max_cycle = 300
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.conv_tol = float(scf_tolerance)
        mean_field.max_cycle = 300
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"Source RHF did not converge at R={bond}.")
    frozen_core = 1
    coefficients = np.asarray(
        mean_field.mo_coeff[:, frozen_core : frozen_core + active_orbitals],
        dtype=float,
    )
    energies = np.asarray(
        mean_field.mo_energy[frozen_core : frozen_core + active_orbitals],
        dtype=float,
    )
    occupations = np.asarray(
        mean_field.mo_occ[frozen_core : frozen_core + active_orbitals],
        dtype=float,
    )
    return modules.p3.OrbitalSnapshot(
        bond_length=bond,
        molecule=molecule,
        active_coefficients=coefficients,
        active_energies=energies,
        active_occupations=occupations,
        scf_total_energy_ha=float(mean_field.e_tot),
        num_frozen_core_orbitals=frozen_core,
    )


def custom_target_rhf(
    modules: Modules,
    bond: float,
    basis: str,
    scf_tolerance: float,
) -> Any:
    from pyscf import gto, scf

    molecule = gto.M(
        atom=f"Li 0 0 0; H 0 0 {bond}",
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    molecule.incore_anyway = True
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = float(scf_tolerance)
    mean_field.max_cycle = 300
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.conv_tol = float(scf_tolerance)
        mean_field.max_cycle = 300
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"Target RHF did not converge at R={bond}.")
    return modules.p7.TargetRHFData(
        bond_length=bond,
        molecule=molecule,
        mean_field=mean_field,
        num_frozen_core_orbitals=1,
    )


def build_system_from_embedding(
    modules: Modules,
    bond: float,
    target: Any,
    full_rotation: np.ndarray,
    embedding_quality: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from pyscf import mcscf
    from pyscf.fci import direct_spin1

    physical = np.asarray(target.mean_field.mo_coeff, dtype=float) @ np.asarray(
        full_rotation, dtype=float
    )
    casci = mcscf.CASCI(target.mean_field, 10, (1, 1))
    casci.verbose = 0
    casci.mo_coeff = physical
    h1eff, core_energy = casci.get_h1eff()
    h2eff = casci.get_h2eff()
    addresses, electronic = direct_spin1.pspace(
        np.asarray(h1eff, dtype=float),
        np.asarray(h2eff, dtype=float),
        10,
        (1, 1),
        np=100,
    )
    if not np.array_equal(addresses, np.arange(100)):
        raise RuntimeError("Unexpected PySCF determinant ordering.")
    hamiltonian = np.asarray(electronic, dtype=float)
    hamiltonian += float(core_energy) * np.eye(100)
    hermiticity = float(np.max(np.abs(hamiltonian - hamiltonian.T)))
    exact = float(np.linalg.eigvalsh(hamiltonian)[0])
    return hamiltonian, {
        "bond_length_angstrom": bond,
        "geometry_key": geometry_key(bond),
        "scf_total_energy_ha": float(target.mean_field.e_tot),
        "scf_converged": bool(target.mean_field.converged),
        "embedding_quality": float(embedding_quality),
        "full_rotation_orthogonality_residual": float(
            np.max(np.abs(full_rotation.T @ full_rotation - np.eye(11)))
        ),
        "hamiltonian_hermiticity_residual": hermiticity,
        "exact_2e10o_total_energy_ha": exact,
        "hf_determinant_energy_ha": float(hamiltonian[0, 0]),
        "core_energy_ha": float(core_energy),
    }


def build_systems(
    modules: Modules,
    bonds: Sequence[float],
    scf_tolerance: float = 1e-10,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    tracking_bonds = tuple(sorted(set((1.595,) + tuple(float(x) for x in bonds))))
    source_snapshots = {
        geometry_key(bond): custom_source_snapshot(
            modules, bond, "sto-3g", 5, scf_tolerance
        )
        for bond in tracking_bonds
    }
    groups, _details, summary = modules.p3.track_orbitals(
        source_snapshots,
        tracking_bonds,
        0.75,
        1e-6,
    )
    if not bool(summary["tracking_reliable"].all()):
        raise RuntimeError("Source-orbital tracking failed on the extension grid.")

    systems: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for bond in bonds:
        key = geometry_key(bond)
        target = custom_target_rhf(modules, bond, "6-31g", scf_tolerance)
        cross = modules.p7a1.cross_overlap_matrix(source_snapshots[key], target)
        rotation = modules.p7a1.build_rotated_embedding(
            bond, cross, target, groups, 0.75
        )
        full_rotation, active_residual, full_residual = (
            modules.p7a2.complete_full_rotation(
                rotation.weights_in_full_target_mos, target, 1e-8
            )
        )
        hamiltonian, row = build_system_from_embedding(
            modules,
            bond,
            target,
            full_rotation,
            rotation.minimum_assigned_group_quality,
        )
        row["active_rotation_orthonormality_residual"] = active_residual
        row["full_rotation_orthogonality_residual"] = full_residual
        systems[key] = hamiltonian
        rows.append(row)
        print(
            f"built R={bond:.3f}: exact={row['exact_2e10o_total_energy_ha']:.12f}, "
            f"quality={row['embedding_quality']:.6f}",
            flush=True,
        )
    return systems, rows


def save_systems(
    output_dir: Path,
    systems: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "benchmark_hamiltonians.npz",
        **{f"h_{key.replace('.', '_')}": value for key, value in systems.items()},
    )
    write_csv(output_dir / "benchmark_hamiltonian_audit.csv", rows)
    write_json(output_dir / "benchmark_hamiltonian_audit.json", list(rows))


def load_systems(output_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    audit_rows = json.loads(
        (output_dir / "benchmark_hamiltonian_audit.json").read_text(encoding="utf-8")
    )
    archive = np.load(output_dir / "benchmark_hamiltonians.npz")
    systems = {
        str(row["geometry_key"]): np.asarray(
            archive[f"h_{str(row['geometry_key']).replace('.', '_')}"]
        )
        for row in audit_rows
    }
    audits = {str(row["geometry_key"]): row for row in audit_rows}
    return systems, audits


def support_state(
    modules: Modules,
    parameters: Sequence[float],
    selected_ids: Sequence[int],
    pool: Sequence[Any],
) -> np.ndarray:
    state = np.zeros(100, dtype=float)
    state[0] = 1.0
    for theta, global_id in zip(parameters, selected_ids):
        generator = modules.p7c.generator_for_excitation(
            int(global_id), pool[int(global_id)]
        )
        state = modules.p7c.apply_generator_rotation(state, generator, float(theta))
    return state


def spin_square(state: np.ndarray) -> float:
    from pyscf.fci import spin_op

    value, _multiplicity = spin_op.spin_square0(
        np.asarray(state, dtype=float).reshape(10, 10), 10, (1, 1)
    )
    return float(value)


def optimize_support(
    modules: Modules,
    hamiltonian: np.ndarray,
    pool: Sequence[Any],
    selected_ids: Sequence[int],
    starts: Sequence[np.ndarray],
    maxiter: int = 4000,
) -> dict[str, Any]:
    generators = [
        modules.p7c.generator_for_excitation(int(index), pool[int(index)])
        for index in selected_ids
    ]

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        return modules.p7c.energy_and_gradient(values, hamiltonian, generators)

    runs: list[dict[str, Any]] = []
    for start_id, initial in enumerate(starts):
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (len(selected_ids),):
            raise ValueError("Initial vector does not match support size.")
        first = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": maxiter,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxls": 100,
                "maxcor": 30,
            },
        )
        second = minimize(
            objective,
            np.asarray(first.x, dtype=float),
            method="BFGS",
            jac=True,
            options={
                "maxiter": maxiter,
                "gtol": 1e-10,
                "norm": np.inf,
            },
        )
        candidates = [first, second]
        best_result = min(candidates, key=lambda item: float(item.fun))
        energy, gradient = objective(np.asarray(best_result.x, dtype=float))
        state = support_state(
            modules, best_result.x, selected_ids, pool
        )
        runs.append(
            {
                "start_id": start_id,
                "energy_total_ha": energy,
                "gradient_infinity_norm": float(np.max(np.abs(gradient)))
                if len(gradient)
                else 0.0,
                "parameters": np.asarray(best_result.x, dtype=float),
                "s2": spin_square(state),
                "iterations": int(getattr(best_result, "nit", 0)),
                "function_evaluations": int(getattr(best_result, "nfev", 0)),
                "optimizer_success": bool(best_result.success),
                "optimizer_message": str(best_result.message),
            }
        )
    best = min(runs, key=lambda item: item["energy_total_ha"])
    return {
        **best,
        "restart_energy_spread_ha": float(
            max(row["energy_total_ha"] for row in runs)
            - min(row["energy_total_ha"] for row in runs)
        ),
        "num_starts": len(runs),
    }


def deterministic_starts(size: int, seed: int, count: int = 2) -> list[np.ndarray]:
    result = [np.zeros(size, dtype=float)]
    rng = np.random.default_rng(seed)
    for _ in range(1, count):
        result.append(rng.normal(scale=0.1, size=size))
    return result


def load_frozen_supports(repository: Path) -> tuple[dict[str, list[int]], dict[str, Any]]:
    path = repository / "results" / "phase7d" / "phase7d_frozen_confirmation_support.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = {
        str(item["name"]): [int(value) for value in item["selected_global_uccsd_indices"]]
        for item in payload["variants"]
    }
    return variants, payload


def run_main_supports(
    modules: Modules,
    repository: Path,
    output_dir: Path,
    systems: dict[str, np.ndarray],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants, _payload = load_frozen_supports(repository)
    pool = modules.p7c.expanded_pool()
    rows: list[dict[str, Any]] = []
    for bond in ALL_BONDS:
        key = geometry_key(bond)
        exact = float(audits[key]["exact_2e10o_total_energy_ha"])
        for name in ("compact10", "chemical_boundary46", "spin_complete_candidate47", "full99"):
            ids = variants[name]
            result = optimize_support(
                modules,
                systems[key],
                pool,
                ids,
                deterministic_starts(
                    len(ids), BASE_SEED + int(round(1000 * bond)) + len(ids), 3
                ),
            )
            error = float(result["energy_total_ha"] - exact)
            row = {
                "bond_length_angstrom": bond,
                "geometry_role": "selection_grid" if bond in ORIGINAL_BONDS else "held_out",
                "variant": name,
                "operator_count": len(ids),
                "energy_total_ha": result["energy_total_ha"],
                "exact_total_energy_ha": exact,
                "absolute_error_ha": abs(error),
                "s2": result["s2"],
                "gradient_infinity_norm": result["gradient_infinity_norm"],
                "restart_energy_spread_ha": result["restart_energy_spread_ha"],
                "num_starts": result["num_starts"],
                "optimizer_success": result["optimizer_success"],
                "parameters_json": json.dumps(result["parameters"].tolist()),
            }
            rows.append(row)
            print(
                f"main R={bond:.3f} {name}: error={abs(error):.3e}, "
                f"S2={result['s2']:.3e}, grad={result['gradient_infinity_norm']:.3e}",
                flush=True,
            )
    write_csv(output_dir / "main_and_holdout_support_results.csv", rows)
    write_json(output_dir / "main_and_holdout_support_results.json", rows)
    return rows


def pool_gradient(
    modules: Modules,
    state: np.ndarray,
    hamiltonian: np.ndarray,
    pool: Sequence[Any],
    candidate_ids: Iterable[int],
) -> dict[int, float]:
    h_state = hamiltonian @ state
    result: dict[int, float] = {}
    for global_id in candidate_ids:
        generator = modules.p7c.generator_for_excitation(global_id, pool[global_id])
        derivative = modules.p7c.apply_generator_derivative(state, generator, 0.0)
        result[int(global_id)] = 2.0 * float(h_state @ derivative)
    return result


def run_adapt(
    modules: Modules,
    hamiltonian: np.ndarray,
    exact: float,
    pool: Sequence[Any],
    bond: float,
    max_operators: int = 70,
) -> tuple[list[dict[str, Any]], list[int], np.ndarray]:
    selected: list[int] = []
    parameters = np.zeros(0, dtype=float)
    rows: list[dict[str, Any]] = []
    cumulative_screens = 0
    for iteration in range(max_operators + 1):
        if selected:
            warm = np.asarray(parameters, dtype=float)
            result = optimize_support(
                modules,
                hamiltonian,
                pool,
                selected,
                [warm],
                maxiter=3000,
            )
            parameters = np.asarray(result["parameters"], dtype=float)
            energy = float(result["energy_total_ha"])
            parameter_gradient = float(result["gradient_infinity_norm"])
            s2 = float(result["s2"])
        else:
            energy = float(hamiltonian[0, 0])
            parameter_gradient = 0.0
            s2 = 0.0
        state = support_state(modules, parameters, selected, pool)
        remaining = [index for index in range(len(pool)) if index not in selected]
        gradients = pool_gradient(modules, state, hamiltonian, pool, remaining)
        cumulative_screens += len(remaining)
        pool_norm = float(np.linalg.norm(list(gradients.values()))) if gradients else 0.0
        max_id = min(
            gradients,
            key=lambda index: (-abs(gradients[index]), index),
        ) if gradients else None
        max_gradient = abs(gradients[max_id]) if max_id is not None else 0.0
        error = abs(energy - exact)
        rows.append(
            {
                "bond_length_angstrom": bond,
                "iteration": iteration,
                "operator_count": len(selected),
                "energy_total_ha": energy,
                "absolute_error_ha": error,
                "s2": s2,
                "parameter_gradient_infinity_norm": parameter_gradient,
                "pool_gradient_l2_norm": pool_norm,
                "largest_pool_gradient": max_gradient,
                "next_global_id": max_id,
                "cumulative_pool_gradient_evaluations": cumulative_screens,
                "selected_ids_json": json.dumps(selected),
                "parameters_json": json.dumps(parameters.tolist()),
            }
        )
        print(
            f"ADAPT R={bond:.3f} n={len(selected)} error={error:.3e} "
            f"pool_norm={pool_norm:.3e}",
            flush=True,
        )
        if (error <= ENERGY_TARGET_HA and pool_norm <= 1e-6) or max_id is None:
            break
        selected.append(int(max_id))
        parameters = np.concatenate([parameters, np.zeros(1)])
    return rows, selected, parameters


def run_adapt_benchmarks(
    modules: Modules,
    output_dir: Path,
    systems: dict[str, np.ndarray],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pool = modules.p7c.expanded_pool()
    rows: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    for bond in ORIGINAL_BONDS:
        key = geometry_key(bond)
        exact = float(audits[key]["exact_2e10o_total_energy_ha"])
        history, ids, parameters = run_adapt(
            modules, systems[key], exact, pool, bond
        )
        rows.extend(history)
        final.append(
            {
                "bond_length_angstrom": bond,
                "selected_ids": ids,
                "parameters": parameters,
                "operator_count": len(ids),
                "final_absolute_error_ha": history[-1]["absolute_error_ha"],
                "final_s2": history[-1]["s2"],
                "pool_gradient_evaluations": history[-1][
                    "cumulative_pool_gradient_evaluations"
                ],
            }
        )
    write_csv(output_dir / "adapt_vqe_iteration_history.csv", rows)
    write_json(output_dir / "adapt_vqe_final_supports.json", final)
    return rows


def spin_partner(modules: Modules, pool: Sequence[Any], global_id: int) -> int:
    return int(modules.p7d.spin_partner_id(pool, int(global_id)))


def run_boundary_plus_one_controls(
    modules: Modules,
    repository: Path,
    output_dir: Path,
    systems: dict[str, np.ndarray],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants, _payload = load_frozen_supports(repository)
    pool = modules.p7c.expanded_pool()
    boundary = set(variants["chemical_boundary46"])
    partner_id = 17
    rows: list[dict[str, Any]] = []
    for added in sorted(set(range(99)) - boundary):
        selected = sorted(boundary | {added})
        for bond in ORIGINAL_BONDS:
            key = geometry_key(bond)
            exact = float(audits[key]["exact_2e10o_total_energy_ha"])
            seed = BASE_SEED + 500000 + 1000 * added + int(round(100 * bond))
            result = optimize_support(
                modules,
                systems[key],
                pool,
                selected,
                deterministic_starts(len(selected), seed, 2),
                maxiter=3000,
            )
            rows.append(
                {
                    "added_global_id": added,
                    "added_is_missing_spin_partner": added == partner_id,
                    "bond_length_angstrom": bond,
                    "absolute_error_ha": abs(result["energy_total_ha"] - exact),
                    "s2": result["s2"],
                    "gradient_infinity_norm": result["gradient_infinity_norm"],
                    "restart_energy_spread_ha": result["restart_energy_spread_ha"],
                }
            )
        worst = max(row["absolute_error_ha"] for row in rows if row["added_global_id"] == added)
        print(f"boundary+E{added}: worst error={worst:.3e}", flush=True)
    write_csv(output_dir / "boundary46_plus_one_exhaustive_controls.csv", rows)
    summary: list[dict[str, Any]] = []
    for added in sorted(set(row["added_global_id"] for row in rows)):
        group = [row for row in rows if row["added_global_id"] == added]
        summary.append(
            {
                "added_global_id": added,
                "added_is_missing_spin_partner": added == partner_id,
                "worst_absolute_error_ha": max(row["absolute_error_ha"] for row in group),
                "worst_s2": max(row["s2"] for row in group),
                "all_geometry_chemical": all(row["absolute_error_ha"] <= CHEMICAL_HA for row in group),
                "all_geometry_strict": all(row["absolute_error_ha"] <= STRICT_HA for row in group),
            }
        )
    write_csv(output_dir / "boundary46_plus_one_exhaustive_summary.csv", summary)
    return rows


def choose_weighted(rng: np.random.Generator, values: Sequence[int], weights: Sequence[int]) -> int:
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= np.sum(probabilities)
    return int(rng.choice(np.asarray(values, dtype=int), p=probabilities))


def random_partner_closed_supports(
    modules: Modules,
    compact_ids: Sequence[int],
    count: int,
    seed: int,
) -> list[list[int]]:
    pool = modules.p7c.expanded_pool()
    compact = set(int(value) for value in compact_ids)
    single_pairs = [(a, a + 9) for a in range(9)]
    available_single_pairs = [pair for pair in single_pairs if not set(pair) <= compact]
    diagonal_doubles = [18 + 9 * a + a for a in range(9)]
    available_diagonals = [value for value in diagonal_doubles if value not in compact]
    off_diagonal_pairs = [
        (18 + 9 * a + b, 18 + 9 * b + a)
        for a in range(9)
        for b in range(a + 1, 9)
        if not {18 + 9 * a + b, 18 + 9 * b + a} <= compact
    ]
    feasible_d = [d for d in (1, 3, 5) if d <= len(available_diagonals)]
    weights = [
        math.comb(len(available_diagonals), d)
        * math.comb(len(off_diagonal_pairs), (31 - d) // 2)
        for d in feasible_d
    ]
    rng = np.random.default_rng(seed)
    supports: set[tuple[int, ...]] = set()
    while len(supports) < count:
        selected = set(compact)
        pair_indices = rng.choice(len(available_single_pairs), size=3, replace=False)
        for index in pair_indices:
            selected.update(available_single_pairs[int(index)])
        d = choose_weighted(rng, feasible_d, weights)
        selected.update(
            int(value)
            for value in rng.choice(available_diagonals, size=d, replace=False)
        )
        pair_count = (31 - d) // 2
        double_pair_indices = rng.choice(
            len(off_diagonal_pairs), size=pair_count, replace=False
        )
        for index in double_pair_indices:
            selected.update(off_diagonal_pairs[int(index)])
        ordered = tuple(sorted(selected))
        if len(ordered) != 47:
            raise RuntimeError("Partner-closed control has the wrong size.")
        if any(spin_partner(modules, pool, value) not in selected for value in selected):
            raise RuntimeError("Partner-closed control failed its closure audit.")
        supports.add(ordered)
    return [list(value) for value in sorted(supports)]


def run_partner_closed_controls(
    modules: Modules,
    repository: Path,
    output_dir: Path,
    systems: dict[str, np.ndarray],
    audits: dict[str, dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    variants, _payload = load_frozen_supports(repository)
    pool = modules.p7c.expanded_pool()
    supports = random_partner_closed_supports(
        modules, variants["compact10"], count, BASE_SEED + 700000
    )
    write_json(
        output_dir / "partner_closed_random_supports.json",
        {"seed": BASE_SEED + 700000, "supports": supports},
    )
    rows: list[dict[str, Any]] = []
    for support_id, selected in enumerate(supports):
        for bond in ORIGINAL_BONDS:
            key = geometry_key(bond)
            exact = float(audits[key]["exact_2e10o_total_energy_ha"])
            seed = BASE_SEED + 800000 + 10000 * support_id + int(round(100 * bond))
            result = optimize_support(
                modules,
                systems[key],
                pool,
                selected,
                deterministic_starts(len(selected), seed, 2),
                maxiter=3000,
            )
            rows.append(
                {
                    "support_id": support_id,
                    "bond_length_angstrom": bond,
                    "absolute_error_ha": abs(result["energy_total_ha"] - exact),
                    "s2": result["s2"],
                    "gradient_infinity_norm": result["gradient_infinity_norm"],
                    "restart_energy_spread_ha": result["restart_energy_spread_ha"],
                }
            )
        group = [row for row in rows if row["support_id"] == support_id]
        print(
            f"closed control {support_id + 1}/{count}: "
            f"worst={max(row['absolute_error_ha'] for row in group):.3e}",
            flush=True,
        )
    write_csv(output_dir / "partner_closed_random_control_results.csv", rows)
    summary = []
    for support_id in range(count):
        group = [row for row in rows if row["support_id"] == support_id]
        summary.append(
            {
                "support_id": support_id,
                "worst_absolute_error_ha": max(row["absolute_error_ha"] for row in group),
                "worst_s2": max(row["s2"] for row in group),
                "all_geometry_chemical": all(row["absolute_error_ha"] <= CHEMICAL_HA for row in group),
                "all_geometry_strict": all(row["absolute_error_ha"] <= STRICT_HA for row in group),
            }
        )
    write_csv(output_dir / "partner_closed_random_control_summary.csv", summary)
    return rows


def partner_blocks(
    modules: Modules,
    pool: Sequence[Any],
    candidate_ids: Iterable[int],
) -> list[tuple[int, ...]]:
    candidates = set(int(value) for value in candidate_ids)
    seen: set[int] = set()
    blocks: list[tuple[int, ...]] = []
    for value in sorted(candidates):
        if value in seen:
            continue
        partner = spin_partner(modules, pool, value)
        block = tuple(sorted({value, partner} & candidates))
        if spin_partner(modules, pool, value) in candidates and partner not in block:
            raise RuntimeError("Failed to construct a complete partner block.")
        seen.update(block)
        blocks.append(block)
    return blocks


def run_spin_block_path(
    modules: Modules,
    repository: Path,
    output_dir: Path,
    systems: dict[str, np.ndarray],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants, _payload = load_frozen_supports(repository)
    pool = modules.p7c.expanded_pool()
    compact = set(variants["compact10"])
    internal_ids = set(range(4)) | set(range(9, 13)) | {
        18 + 9 * a + b for a in range(4) for b in range(4)
    }
    external = set(range(99)) - internal_ids
    equilibrium = systems[geometry_key(1.595)]
    expanded = modules.p7c.ExpandedSystem(
        geometry_key(1.595),
        1.595,
        equilibrium,
        0.0,
        float(audits[geometry_key(1.595)]["exact_2e10o_total_energy_ha"]),
        0.0,
        float(np.max(np.abs(equilibrium - equilibrium.T))),
        True,
    )
    score_rows = modules.p7c.rank_external_operators(
        expanded, pool, internal_ids, compact, 1e-8
    )
    score_by_id = {
        int(row["global_uccsd_index"]): float(row["en_amplitude_score"])
        for row in score_rows
    }
    blocks = partner_blocks(modules, pool, external)
    blocks.sort(key=lambda block: (-max(score_by_id[value] for value in block), block))
    write_json(
        output_dir / "spin_partner_block_ranking.json",
        [
            {
                "block_rank": rank,
                "global_ids": block,
                "block_score": max(score_by_id[value] for value in block),
            }
            for rank, block in enumerate(blocks, start=1)
        ],
    )

    selected = set(compact)
    previous_parameters: dict[str, dict[int, float]] = {
        geometry_key(bond): {} for bond in ORIGINAL_BONDS
    }
    rows: list[dict[str, Any]] = []
    for block_rank, block in enumerate(blocks, start=1):
        selected.update(block)
        ordered = sorted(selected)
        for bond in ORIGINAL_BONDS:
            key = geometry_key(bond)
            exact = float(audits[key]["exact_2e10o_total_energy_ha"])
            warm_map = previous_parameters[key]
            warm = np.asarray([warm_map.get(value, 0.0) for value in ordered])
            rng = np.random.default_rng(
                BASE_SEED + 900000 + block_rank * 1000 + int(round(100 * bond))
            )
            result = optimize_support(
                modules,
                systems[key],
                pool,
                ordered,
                [warm, rng.normal(scale=0.1, size=len(ordered))],
                maxiter=3000,
            )
            previous_parameters[key] = {
                value: float(parameter)
                for value, parameter in zip(ordered, result["parameters"])
            }
            rows.append(
                {
                    "block_rank": block_rank,
                    "added_block_json": json.dumps(block),
                    "operator_count": len(ordered),
                    "bond_length_angstrom": bond,
                    "absolute_error_ha": abs(result["energy_total_ha"] - exact),
                    "s2": result["s2"],
                    "gradient_infinity_norm": result["gradient_infinity_norm"],
                    "selected_ids_json": json.dumps(ordered),
                }
            )
        group = [row for row in rows if row["block_rank"] == block_rank]
        worst = max(row["absolute_error_ha"] for row in group)
        print(
            f"block path rank={block_rank} n={len(ordered)} worst={worst:.3e}",
            flush=True,
        )
        if worst <= ENERGY_TARGET_HA:
            break
    write_csv(output_dir / "spin_partner_block_path_results.csv", rows)
    return rows


def run_ranking_sensitivity(
    modules: Modules,
    output_dir: Path,
) -> list[dict[str, Any]]:
    pool = modules.p7c.expanded_pool()
    compact = {0, 3, 9, 12, 18, 21, 28, 38, 45, 48}
    internal = set(range(4)) | set(range(9, 13)) | {
        18 + 9 * a + b for a in range(4) for b in range(4)
    }
    conditions: list[dict[str, Any]] = []
    rankings: dict[str, list[int]] = {}
    for tolerance in (1e-10, 1e-12, 1e-14):
        systems, audits = build_systems(modules, [1.595], tolerance)
        hamiltonian = systems[geometry_key(1.595)]
        expanded = modules.p7c.ExpandedSystem(
            geometry_key(1.595),
            1.595,
            hamiltonian,
            float(audits[0]["core_energy_ha"]),
            float(audits[0]["exact_2e10o_total_energy_ha"]),
            0.0,
            float(audits[0]["hamiltonian_hermiticity_residual"]),
            True,
        )
        for floor in (1e-8, 1e-10, 1e-12):
            rows = modules.p7c.rank_external_operators(
                expanded, pool, internal, compact, floor
            )
            ranking = list(modules.p7c.external_ranking(rows))
            key = f"scf_{tolerance:.0e}_floor_{floor:.0e}"
            rankings[key] = ranking
            score = {
                int(row["global_uccsd_index"]): float(row["en_amplitude_score"])
                for row in rows
            }
            conditions.append(
                {
                    "condition": key,
                    "scf_tolerance": tolerance,
                    "denominator_floor_ha": floor,
                    "e8_rank": ranking.index(8) + 1,
                    "e17_rank": ranking.index(17) + 1,
                    "e8_score": score[8],
                    "e17_score": score[17],
                    "top37_json": json.dumps(ranking[:37]),
                }
            )
    baseline_key = "scf_1e-10_floor_1e-08"
    baseline = rankings[baseline_key]
    baseline_rank = {value: index for index, value in enumerate(baseline)}
    baseline_top = set(baseline[:37])
    for row in conditions:
        ranking = rankings[row["condition"]]
        order_a = [baseline_rank[value] for value in ranking]
        tau = float(kendalltau(np.arange(len(order_a)), order_a).statistic)
        top = set(ranking[:37])
        row["kendall_tau_vs_baseline"] = tau
        row["top37_jaccard_vs_baseline"] = len(top & baseline_top) / len(top | baseline_top)
        row["top37_set_identical_to_baseline"] = top == baseline_top
    write_csv(output_dir / "ranking_sensitivity.csv", conditions)
    write_json(output_dir / "ranking_sensitivity_rankings.json", rankings)
    return conditions


def circuit_metrics(
    modules: Modules,
    selected_ids: Sequence[int],
    pool: Sequence[Any],
    seed: int = 20261012,
) -> dict[str, Any]:
    from qiskit import transpile
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    excitations = [pool[int(value)] for value in selected_ids]

    def generator(
        num_spatial_orbitals: int,
        num_particles: tuple[int, int],
    ) -> list[Any]:
        del num_spatial_orbitals, num_particles
        return list(excitations)

    mapper = JordanWignerMapper()
    initial = HartreeFock(10, (1, 1), mapper)
    ansatz = UCC(
        num_spatial_orbitals=10,
        num_particles=(1, 1),
        excitations=generator,
        qubit_mapper=mapper,
        preserve_spin=True,
        reps=1,
        initial_state=initial,
    )
    _ = ansatz.num_parameters
    compiled = transpile(
        ansatz,
        basis_gates=["u", "cx"],
        optimization_level=3,
        seed_transpiler=seed,
    )
    operations = compiled.count_ops()
    return {
        "operator_count": len(selected_ids),
        "logical_depth": int(ansatz.depth()),
        "logical_size": int(ansatz.size()),
        "compiled_depth": int(compiled.depth()),
        "compiled_size": int(compiled.size()),
        "compiled_cx": int(operations.get("cx", 0)),
        "compiled_u": int(operations.get("u", 0)),
    }


def run_circuit_comparisons(
    modules: Modules,
    repository: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    variants, _payload = load_frozen_supports(repository)
    pool = modules.p7c.expanded_pool()
    supports: list[tuple[str, list[int]]] = [
        ("boundary46", variants["chemical_boundary46"]),
        ("partner47", variants["spin_complete_candidate47"]),
        ("full99", variants["full99"]),
    ]
    adapt_path = output_dir / "adapt_vqe_final_supports.json"
    if adapt_path.is_file():
        adapt = json.loads(adapt_path.read_text(encoding="utf-8"))
        for row in adapt:
            supports.append(
                (f"adapt_R_{float(row['bond_length_angstrom']):.3f}", [int(x) for x in row["selected_ids"]])
            )
    rows = []
    for name, selected in supports:
        metrics = circuit_metrics(modules, selected, pool)
        rows.append({"support": name, **metrics})
        print(
            f"circuit {name}: n={len(selected)}, cx={metrics['compiled_cx']}, "
            f"depth={metrics['compiled_depth']}",
            flush=True,
        )
    write_csv(output_dir / "matched_circuit_resources.csv", rows)
    return rows


def build_protocol(repository: Path, control_count: int) -> dict[str, Any]:
    support_path = repository / "results" / "phase7d" / "phase7d_frozen_confirmation_support.json"
    support_payload = json.loads(support_path.read_text(encoding="utf-8"))
    protocol = {
        "analysis": "LiH post-rejection benchmark extension",
        "date_declared": "2026-10-12",
        "frozen_phase7d_support_fingerprint": support_payload["phase7d_support_fingerprint"],
        "support_reselection_for_holdouts": False,
        "original_selection_geometries_angstrom": list(ORIGINAL_BONDS),
        "held_out_geometries_angstrom": list(HOLDOUT_BONDS),
        "chemical_accuracy_ha": CHEMICAL_HA,
        "strict_accuracy_ha": STRICT_HA,
        "adapt_energy_target_ha": ENERGY_TARGET_HA,
        "reported_gradient_target": GRADIENT_TARGET,
        "partner_closed_random_control_count": control_count,
        "random_seed": BASE_SEED,
        "boundary_plus_one_control": "all 53 single additions to frozen Boundary-46",
        "ranking_sensitivity_scf_tolerances": [1e-10, 1e-12, 1e-14],
        "ranking_sensitivity_denominator_floors_ha": [1e-8, 1e-10, 1e-12],
        "claim_boundary": (
            "These calculations extend the frozen LiH support audit. They are "
            "out-of-sample with respect to support selection but were designed "
            "after peer review and are not claimed as prospectively preregistered "
            "evidence or as evidence of transfer to another molecule."
        ),
    }
    protocol["protocol_fingerprint"] = fingerprint(protocol)
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--group",
        choices=(
            "systems",
            "main",
            "adapt",
            "boundary-controls",
            "closed-controls",
            "block-path",
            "ranking",
            "circuits",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--closed-control-count", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    args.repository = args.repository.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modules = import_modules(args.repository)

    protocol_path = args.output_dir / "extension_protocol.json"
    protocol = build_protocol(args.repository, args.closed_control_count)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing extension protocol differs from requested run.")
    else:
        write_json(protocol_path, protocol)

    if args.group in {"systems", "all"}:
        systems, rows = build_systems(modules, ALL_BONDS)
        save_systems(args.output_dir, systems, rows)
    if args.group == "systems":
        return 0

    systems, audits = load_systems(args.output_dir)
    if args.group in {"main", "all"}:
        run_main_supports(modules, args.repository, args.output_dir, systems, audits)
    if args.group in {"adapt", "all"}:
        run_adapt_benchmarks(modules, args.output_dir, systems, audits)
    if args.group in {"boundary-controls", "all"}:
        run_boundary_plus_one_controls(
            modules, args.repository, args.output_dir, systems, audits
        )
    if args.group in {"closed-controls", "all"}:
        run_partner_closed_controls(
            modules,
            args.repository,
            args.output_dir,
            systems,
            audits,
            args.closed_control_count,
        )
    if args.group in {"block-path", "all"}:
        run_spin_block_path(
            modules, args.repository, args.output_dir, systems, audits
        )
    if args.group in {"ranking", "all"}:
        run_ranking_sensitivity(modules, args.output_dir)
    if args.group in {"circuits", "all"}:
        run_circuit_comparisons(modules, args.repository, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
