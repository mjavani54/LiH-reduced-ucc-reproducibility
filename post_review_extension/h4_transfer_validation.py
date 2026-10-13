#!/usr/bin/env python3
"""Four-electron H4 transfer validation for the revised UCC manuscript.

The calculation is intentionally compact and independent of the archived LiH
workflow.  A fermionic ADAPT support is selected at one linear-H4 geometry,
closed under alpha/beta spin partnership, frozen, and evaluated at three
previously unused H-H spacings.  Full factorized UCCSD and geometry-specific
ADAPT calculations provide same-Hamiltonian references.

All states are represented exactly in the fixed (N_alpha, N_beta)=(2,2)
determinant sector, so the calculation contains neither sampling noise nor a
qubit mapping approximation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize


REFERENCE_SPACING = 1.2
TEST_SPACINGS = (0.8, 1.6, 2.0)
ALL_SPACINGS = (0.8, 1.2, 1.6, 2.0)
STRICT_HA = 1.0e-6
CHEMICAL_HA = 1.6e-3
BASE_SEED = 20261012


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def excitation_label(excitation: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    occupied, virtual = excitation
    return f"{','.join(map(str, occupied))}->{','.join(map(str, virtual))}"


@dataclass(frozen=True)
class Generator:
    source: np.ndarray
    target: np.ndarray
    sign: np.ndarray


@dataclass
class System:
    spacing: float
    hamiltonian: np.ndarray
    exact_energy: float
    hf_energy: float
    determinant_bits: list[int]
    molecule: Any
    mo_coeff: np.ndarray


def annihilate(bits: int, orbital: int) -> tuple[int, int] | None:
    if not (bits >> orbital) & 1:
        return None
    sign = -1 if (bits & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return bits ^ (1 << orbital), sign


def create(bits: int, orbital: int) -> tuple[int, int] | None:
    if (bits >> orbital) & 1:
        return None
    sign = -1 if (bits & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return bits | (1 << orbital), sign


def make_generator(
    excitation: tuple[tuple[int, ...], tuple[int, ...]],
    determinant_bits: Sequence[int],
) -> Generator:
    occupied, virtual = excitation
    bit_to_index = {bits: index for index, bits in enumerate(determinant_bits)}
    sources: list[int] = []
    targets: list[int] = []
    signs: list[int] = []
    for source_index, source_bits in enumerate(determinant_bits):
        bits = source_bits
        total_sign = 1
        valid = True
        for orbital in occupied:
            result = annihilate(bits, orbital)
            if result is None:
                valid = False
                break
            bits, sign = result
            total_sign *= sign
        if not valid:
            continue
        for orbital in reversed(virtual):
            result = create(bits, orbital)
            if result is None:
                valid = False
                break
            bits, sign = result
            total_sign *= sign
        if not valid or bits not in bit_to_index:
            continue
        sources.append(source_index)
        targets.append(bit_to_index[bits])
        signs.append(total_sign)
    return Generator(
        np.asarray(sources, dtype=int),
        np.asarray(targets, dtype=int),
        np.asarray(signs, dtype=float),
    )


def generator_action(state: np.ndarray, generator: Generator) -> np.ndarray:
    result = np.zeros_like(state)
    result[generator.target] += generator.sign * state[generator.source]
    result[generator.source] -= generator.sign * state[generator.target]
    return result


def rotate(state: np.ndarray, generator: Generator, angle: float) -> np.ndarray:
    result = np.array(state, copy=True)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    source_values = state[generator.source]
    target_values = state[generator.target]
    result[generator.source] = (
        cosine * source_values - generator.sign * sine * target_values
    )
    result[generator.target] = (
        cosine * target_values + generator.sign * sine * source_values
    )
    return result


def state_and_gradient(
    parameters: np.ndarray,
    hamiltonian: np.ndarray,
    generators: Sequence[Generator],
    reference_index: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    initial = np.zeros(hamiltonian.shape[0])
    initial[reference_index] = 1.0
    forward = [initial]
    for angle, generator in zip(parameters, generators):
        forward.append(rotate(forward[-1], generator, float(angle)))
    state = forward[-1]
    h_state = hamiltonian @ state
    energy = float(state @ h_state)
    gradient = np.zeros(len(generators))
    backward = h_state
    for index in range(len(generators) - 1, -1, -1):
        gradient[index] = 2.0 * float(
            backward @ generator_action(forward[index + 1], generators[index])
        )
        backward = rotate(backward, generators[index], -float(parameters[index]))
    return energy, gradient, state


def deterministic_starts(size: int, seed: int, count: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.zeros(size)]
    for _ in range(count - 1):
        starts.append(rng.normal(0.0, 0.02, size=size))
    return starts


def optimize(
    hamiltonian: np.ndarray,
    support: Sequence[int],
    pool_generators: Sequence[Generator],
    reference_index: int,
    starts: Sequence[np.ndarray],
    maxiter: int = 4000,
) -> dict[str, Any]:
    generators = [pool_generators[index] for index in support]

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        energy, gradient, _state = state_and_gradient(
            values, hamiltonian, generators, reference_index
        )
        return energy, gradient

    runs: list[dict[str, Any]] = []
    for start_id, start in enumerate(starts):
        first = minimize(
            objective,
            np.asarray(start),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10, "maxls": 100},
        )
        second = minimize(
            objective,
            np.asarray(first.x),
            method="BFGS",
            jac=True,
            options={"maxiter": maxiter, "gtol": 1e-10, "norm": np.inf},
        )
        best = min((first, second), key=lambda item: float(item.fun))
        energy, gradient, state = state_and_gradient(
            np.asarray(best.x), hamiltonian, generators, reference_index
        )
        runs.append(
            {
                "start_id": start_id,
                "energy": energy,
                "gradient_infinity_norm": float(np.max(np.abs(gradient))) if len(gradient) else 0.0,
                "parameters": np.asarray(best.x),
                "state": state,
                "success": bool(best.success),
                "message": str(best.message),
                "nfev": int(best.nfev),
            }
        )
    best_run = min(runs, key=lambda item: item["energy"])
    return {
        "energy": best_run["energy"],
        "gradient_infinity_norm": best_run["gradient_infinity_norm"],
        "parameters": best_run["parameters"],
        "state": best_run["state"],
        "success": best_run["success"],
        "message": best_run["message"],
        "nfev": best_run["nfev"],
        "restart_energy_spread": float(max(r["energy"] for r in runs) - min(r["energy"] for r in runs)),
    }


def build_systems(spacings: Sequence[float]) -> tuple[dict[float, System], list[dict[str, Any]]]:
    from pyscf import ao2mo, fci, gto, scf
    from pyscf.fci import cistring, direct_spin1

    systems: dict[float, System] = {}
    rows: list[dict[str, Any]] = []
    for spacing in spacings:
        coordinates = [(-1.5 * spacing), (-0.5 * spacing), (0.5 * spacing), (1.5 * spacing)]
        molecule = gto.M(
            atom="; ".join(f"H 0 0 {z:.12f}" for z in coordinates),
            basis="sto-3g",
            charge=0,
            spin=0,
            unit="Angstrom",
            verbose=0,
        )
        molecule.incore_anyway = True
        mean_field = scf.RHF(molecule)
        mean_field.conv_tol = 1e-12
        mean_field.max_cycle = 300
        mean_field.kernel()
        if not mean_field.converged:
            mean_field = mean_field.newton()
            mean_field.conv_tol = 1e-12
            mean_field.kernel()
        if not mean_field.converged:
            raise RuntimeError(f"H4 RHF failed at spacing {spacing}")
        coefficients = np.asarray(mean_field.mo_coeff)
        h1 = coefficients.T @ mean_field.get_hcore() @ coefficients
        h2 = ao2mo.restore(1, ao2mo.kernel(molecule, coefficients), 4)
        addresses, electronic = direct_spin1.pspace(h1, h2, 4, (2, 2), np=36)
        if not np.array_equal(addresses, np.arange(36)):
            raise RuntimeError("Unexpected H4 determinant ordering")
        hamiltonian = np.asarray(electronic) + molecule.energy_nuc() * np.eye(36)
        alpha = cistring.addrs2str(4, 2, np.arange(6))
        beta = cistring.addrs2str(4, 2, np.arange(6))
        determinant_bits = [int(a) | (int(b) << 4) for a in alpha for b in beta]
        reference_bits = 0b0011 | (0b0011 << 4)
        reference_index = determinant_bits.index(reference_bits)
        exact = float(np.linalg.eigvalsh(hamiltonian)[0])
        system = System(
            float(spacing),
            hamiltonian,
            exact,
            float(hamiltonian[reference_index, reference_index]),
            determinant_bits,
            molecule,
            coefficients,
        )
        systems[float(spacing)] = system
        rows.append(
            {
                "spacing_angstrom": spacing,
                "rhf_converged": bool(mean_field.converged),
                "rhf_total_energy_ha": float(mean_field.e_tot),
                "exact_fci_total_energy_ha": exact,
                "hf_determinant_energy_ha": float(hamiltonian[reference_index, reference_index]),
                "hamiltonian_hermiticity_residual": float(np.max(np.abs(hamiltonian - hamiltonian.T))),
            }
        )
        print(f"built H4 d={spacing:.2f}: exact={exact:.12f}", flush=True)
    return systems, rows


def orbital_overlap_audit(systems: dict[float, System]) -> list[dict[str, Any]]:
    from pyscf import gto
    from scipy.optimize import linear_sum_assignment

    reference = systems[REFERENCE_SPACING]
    rows: list[dict[str, Any]] = []
    for spacing in ALL_SPACINGS:
        target = systems[spacing]
        cross_overlap = gto.intor_cross(
            "int1e_ovlp", reference.molecule, target.molecule
        )
        molecular_overlap = (
            reference.mo_coeff.T @ cross_overlap @ target.mo_coeff
        )
        record: dict[str, Any] = {"spacing_angstrom": spacing}
        for label, start, stop in (("occupied", 0, 2), ("virtual", 2, 4)):
            block = molecular_overlap[start:stop, start:stop]
            row_ids, column_ids = linear_sum_assignment(-np.abs(block))
            assigned = np.abs(block[row_ids, column_ids])
            off_diagonal = block - np.diag(np.diag(block))
            record[f"{label}_assignment_zero_based"] = json.dumps(column_ids.tolist())
            record[f"minimum_{label}_matched_overlap"] = float(np.min(assigned))
            record[f"maximum_{label}_within_block_off_diagonal_overlap"] = float(
                np.max(np.abs(off_diagonal))
            )
        record["identity_assignment_in_both_blocks"] = bool(
            record["occupied_assignment_zero_based"] == "[0, 1]"
            and record["virtual_assignment_zero_based"] == "[0, 1]"
        )
        rows.append(record)
    return rows


def generate_pool() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations

    pool: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for rank in (1, 2):
        pool.extend(
            generate_fermionic_excitations(
                rank,
                num_spatial_orbitals=4,
                num_particles=(2, 2),
                preserve_spin=True,
            )
        )
    return pool


def spin_partner(
    excitation: tuple[tuple[int, ...], tuple[int, ...]]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    def flip(index: int) -> int:
        return index + 4 if index < 4 else index - 4

    return (
        tuple(sorted(flip(index) for index in excitation[0])),
        tuple(sorted(flip(index) for index in excitation[1])),
    )


def close_support(
    support: Sequence[int],
    pool: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> list[int]:
    lookup = {excitation: index for index, excitation in enumerate(pool)}
    closed = set(int(index) for index in support)
    for index in list(closed):
        partner = spin_partner(pool[index])
        if partner not in lookup:
            raise RuntimeError(f"Missing spin partner for {pool[index]}")
        closed.add(lookup[partner])
    return [index for index in range(len(pool)) if index in closed]


def run_adapt(
    system: System,
    pool_generators: Sequence[Generator],
    reference_index: int,
    seed: int,
) -> dict[str, Any]:
    support: list[int] = []
    parameters = np.zeros(0)
    rows: list[dict[str, Any]] = []
    state = np.zeros(36)
    state[reference_index] = 1.0
    for iteration in range(len(pool_generators) + 1):
        energy = float(state @ system.hamiltonian @ state)
        error = energy - system.exact_energy
        h_state = system.hamiltonian @ state
        gradients = np.asarray(
            [
                0.0 if index in support else 2.0 * float(h_state @ generator_action(state, generator))
                for index, generator in enumerate(pool_generators)
            ]
        )
        rows.append(
            {
                "iteration": iteration,
                "operator_count": len(support),
                "energy_total_ha": energy,
                "absolute_error_ha": abs(error),
                "maximum_unused_gradient": float(np.max(np.abs(gradients))) if iteration < len(pool_generators) else 0.0,
                "selected_operator": support[-1] if support else "",
            }
        )
        if abs(error) <= 1e-8 or len(support) == len(pool_generators):
            break
        selected = int(np.argmax(np.abs(gradients)))
        support.append(selected)
        starts = [np.concatenate((parameters, [0.0]))]
        if len(support) <= 8:
            starts.extend(deterministic_starts(len(support), seed + iteration, 2)[1:])
        result = optimize(
            system.hamiltonian,
            support,
            pool_generators,
            reference_index,
            starts,
        )
        parameters = np.asarray(result["parameters"])
        state = np.asarray(result["state"])
    return {"support": support, "parameters": parameters, "state": state, "history": rows}


def finite_difference_check(
    hamiltonian: np.ndarray,
    generators: Sequence[Generator],
    reference_index: int,
) -> dict[str, float]:
    rng = np.random.default_rng(BASE_SEED)
    selected = [0, 4, 8, 13, 20]
    chosen = [generators[index] for index in selected]
    parameters = rng.normal(0.0, 0.05, size=len(selected))
    energy, gradient, _state = state_and_gradient(parameters, hamiltonian, chosen, reference_index)
    numerical = np.zeros_like(gradient)
    step = 1e-6
    for index in range(len(parameters)):
        plus = np.array(parameters, copy=True)
        minus = np.array(parameters, copy=True)
        plus[index] += step
        minus[index] -= step
        e_plus = state_and_gradient(plus, hamiltonian, chosen, reference_index)[0]
        e_minus = state_and_gradient(minus, hamiltonian, chosen, reference_index)[0]
        numerical[index] = (e_plus - e_minus) / (2.0 * step)
    return {
        "test_energy_ha": energy,
        "maximum_analytic_finite_difference_difference": float(np.max(np.abs(gradient - numerical))),
    }


def spin_square(state: np.ndarray) -> float:
    from pyscf.fci import spin_op

    value, _multiplicity = spin_op.spin_square0(
        np.asarray(state).reshape(6, 6), 4, (2, 2)
    )
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper_improvement/results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    systems, audit_rows = build_systems(ALL_SPACINGS)
    overlap_rows = orbital_overlap_audit(systems)
    pool = generate_pool()
    reference_bits = 0b0011 | (0b0011 << 4)
    reference_index = systems[REFERENCE_SPACING].determinant_bits.index(reference_bits)
    pool_generators = [make_generator(excitation, systems[REFERENCE_SPACING].determinant_bits) for excitation in pool]
    check = finite_difference_check(
        systems[REFERENCE_SPACING].hamiltonian, pool_generators, reference_index
    )
    if check["maximum_analytic_finite_difference_difference"] > 1e-7:
        raise RuntimeError(f"Gradient validation failed: {check}")

    adapt_results: dict[float, dict[str, Any]] = {}
    adapt_rows: list[dict[str, Any]] = []
    for geometry_index, spacing in enumerate(ALL_SPACINGS):
        result = run_adapt(
            systems[spacing], pool_generators, reference_index, BASE_SEED + 1000 * geometry_index
        )
        adapt_results[spacing] = result
        for row in result["history"]:
            adapt_rows.append({"spacing_angstrom": spacing, **row})
        print(
            f"ADAPT d={spacing:.2f}: {len(result['support'])} operators, "
            f"error={result['history'][-1]['absolute_error_ha']:.3e}",
            flush=True,
        )

    reference_history = adapt_results[REFERENCE_SPACING]["history"]
    chemical_row = next(
        row for row in reference_history if row["absolute_error_ha"] <= CHEMICAL_HA
    )
    raw_support = adapt_results[REFERENCE_SPACING]["support"][
        : int(chemical_row["operator_count"])
    ]
    closed_support = close_support(raw_support, pool)
    full_support = list(range(len(pool)))

    evaluation_rows: list[dict[str, Any]] = []
    for geometry_index, spacing in enumerate(ALL_SPACINGS):
        system = systems[spacing]
        supports = {
            "reference_ADAPT_prefix_raw": raw_support,
            "reference_ADAPT_prefix_spin_closed": closed_support,
            "full_factorized_UCCSD": full_support,
        }
        for support_name, support in supports.items():
            result = optimize(
                system.hamiltonian,
                support,
                pool_generators,
                reference_index,
                deterministic_starts(len(support), BASE_SEED + 10000 + geometry_index * 100 + len(support), 3),
            )
            evaluation_rows.append(
                {
                    "spacing_angstrom": spacing,
                    "geometry_role": "selection" if spacing == REFERENCE_SPACING else "held_out",
                    "support_name": support_name,
                    "operator_count": len(support),
                    "energy_total_ha": result["energy"],
                    "exact_energy_total_ha": system.exact_energy,
                    "absolute_error_ha": abs(result["energy"] - system.exact_energy),
                    "spin_squared": spin_square(np.asarray(result["state"])),
                    "gradient_infinity_norm": result["gradient_infinity_norm"],
                    "restart_energy_spread_ha": result["restart_energy_spread"],
                }
            )
            print(
                f"eval d={spacing:.2f} {support_name}: "
                f"n={len(support)}, error={evaluation_rows[-1]['absolute_error_ha']:.3e}",
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for support_name in sorted(set(row["support_name"] for row in evaluation_rows)):
        selected = [row for row in evaluation_rows if row["support_name"] == support_name]
        held = [row for row in selected if row["geometry_role"] == "held_out"]
        summary_rows.append(
            {
                "support_name": support_name,
                "operator_count": selected[0]["operator_count"],
                "maximum_error_all_geometries_ha": max(row["absolute_error_ha"] for row in selected),
                "maximum_error_held_out_geometries_ha": max(row["absolute_error_ha"] for row in held),
                "all_geometries_chemical_accuracy": all(row["absolute_error_ha"] <= CHEMICAL_HA for row in selected),
                "all_geometries_strict_accuracy": all(row["absolute_error_ha"] <= STRICT_HA for row in selected),
                "maximum_gradient_infinity_norm": max(row["gradient_infinity_norm"] for row in selected),
                "maximum_spin_squared": max(row["spin_squared"] for row in selected),
            }
        )

    protocol = {
        "design_date": "2026-10-12",
        "system": "linear H4/STO-3G, four electrons in four spatial orbitals",
        "selection_geometry_spacing_angstrom": REFERENCE_SPACING,
        "held_out_spacings_angstrom": TEST_SPACINGS,
        "selection_rule": (
            "fermionic ADAPT until 1.6e-3 Ha at the reference geometry, then "
            "alpha/beta spin-partner closure"
        ),
        "claim_boundary": (
            "Post-review independent validation on a second, four-electron model; "
            "it tests transfer across H4 spacings but does not establish molecular universality."
        ),
        "pool_size": len(pool),
        "raw_support": raw_support,
        "closed_support": closed_support,
        "raw_support_fingerprint": hashlib.sha256(json.dumps(raw_support).encode()).hexdigest(),
        "closed_support_fingerprint": hashlib.sha256(json.dumps(closed_support).encode()).hexdigest(),
        "gradient_check": check,
    }
    write_csv(args.output_dir / "h4_hamiltonian_audit.csv", audit_rows)
    write_csv(args.output_dir / "h4_orbital_overlap_audit.csv", overlap_rows)
    write_csv(args.output_dir / "h4_adapt_history.csv", adapt_rows)
    write_csv(args.output_dir / "h4_transfer_results.csv", evaluation_rows)
    write_csv(args.output_dir / "h4_transfer_summary.csv", summary_rows)
    write_json(
        args.output_dir / "h4_supports.json",
        {
            "pool": [excitation_label(excitation) for excitation in pool],
            "raw_support": raw_support,
            "closed_support": closed_support,
            "geometry_specific_adapt_supports": {
                str(spacing): result["support"] for spacing, result in adapt_results.items()
            },
        },
    )
    write_json(args.output_dir / "h4_validation_protocol.json", protocol)
    print(json.dumps(summary_rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
