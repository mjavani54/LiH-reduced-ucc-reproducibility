#!/usr/bin/env python3
"""Phase 6: mechanism analysis of the compact LiH UCCSD support.

Phase 5 established that ten reference operators reproduce full 24-operator
UCCSD across the audited LiH/STO-3G bond-length grid.  This program asks why.
It combines five complementary diagnostics:

1. optimized-amplitude trajectories from the five-restart confirmation;
2. tracked-orbital energies and Mulliken orbital populations;
3. HF finite-difference response and frozen-core MP2 double amplitudes;
4. conditional spin-paired-single response on the optimized D6 state; and
5. checkpointed leave-one-out reoptimization of the compact10 support.

The HF response is deliberately not treated as a sufficient singles ranking.
Canonical-RHF singles obey Brillouin's theorem and can have zero first-order
gradient even when they become essential after doubles generate correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import lih_phase2b_exhaustive_support as p2b
    import lih_phase3_geometry_transfer as p3
    import lih_phase4_ansatz_diagnosis as p4
    import lih_phase5_compact_uccsd_ablation as p5
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place the Phase-6 script beside the Phase-2B, Phase-3, Phase-4, "
        "Phase-5, and reference-diagnostic scripts."
    ) from exc


ALL_BOND_LENGTHS: tuple[float, ...] = p5.FULL_BOND_LENGTHS
PILOT_LOO_BOND_LENGTHS: tuple[float, ...] = (1.595, 2.5, 3.0)
COMPACT_LABELS: tuple[str, ...] = tuple(
    p5.operator_label(index)
    for index in p5.VARIANT_BY_NAME["compact10"].reference_global_ids
)
D6_LABELS: tuple[str, ...] = tuple(
    p5.operator_label(index)
    for index in p5.VARIANT_BY_NAME["D6_only"].reference_global_ids
)
SINGLE_CHANNELS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("V1", ("S0", "S4")),
    ("V2", ("S1", "S5")),
    ("V3", ("S2", "S6")),
    ("V4", ("S3", "S7")),
)
ACTIVE_SINGLE_CHANNELS = {"V1", "V4"}


@dataclass
class GeometryMechanismData:
    bond_length: float
    mean_field: Any
    snapshot: p3.OrbitalSnapshot
    mp2_correlation_energy_ha: float
    mp2_t2: np.ndarray


def geometry_key(bond_length: float) -> str:
    return p5.geometry_key(bond_length)


def json_list(value: Any, field: str) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not parse {field}.") from exc
    if not isinstance(parsed, list):
        raise RuntimeError(f"{field} must contain a JSON list.")
    return parsed


def normalized_values(values: Iterable[float]) -> list[float]:
    result = sorted({round(float(value), 9) for value in values})
    if not result or any(value <= 0 for value in result):
        raise ValueError("At least one positive bond length is required.")
    if any(
        not any(abs(value - allowed) <= 1e-9 for allowed in ALL_BOND_LENGTHS)
        for value in result
    ):
        raise ValueError(
            "Leave-one-out geometries must belong to the audited Phase-5 grid."
        )
    return result


def resolved_loo_bond_lengths(
    profile: str,
    overrides: Sequence[float] | None,
) -> list[float]:
    if overrides:
        return normalized_values(overrides)
    if profile == "pilot":
        return list(PILOT_LOO_BOND_LENGTHS)
    return list(ALL_BOND_LENGTHS)


def resolved_loo_operators(
    names: Sequence[str] | None,
) -> list[str]:
    if not names:
        return list(COMPACT_LABELS)
    result = list(dict.fromkeys(str(name) for name in names))
    invalid = [name for name in result if name not in COMPACT_LABELS]
    if invalid:
        raise ValueError(
            "Leave-one-out operators must belong to compact10: "
            + ", ".join(invalid)
        )
    return result


def read_phase5_inputs(
    directory: Path,
    required_variants: Sequence[str],
    minimum_restarts: int,
    role: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runs_path = directory / "phase5_restart_runs.csv"
    conclusions_path = directory / "phase5_causal_conclusions.json"
    if not runs_path.is_file() or not conclusions_path.is_file():
        raise FileNotFoundError(
            f"{role} directory {directory} must contain "
            "phase5_restart_runs.csv and phase5_causal_conclusions.json."
        )
    conclusions = json.loads(conclusions_path.read_text(encoding="utf-8"))
    protocol = str(conclusions.get("protocol_fingerprint", ""))
    if not protocol:
        raise RuntimeError(f"{role} conclusions have no protocol fingerprint.")
    runs = p5.canonicalize_checkpoint(pd.read_csv(runs_path))
    current = runs[
        runs["protocol_fingerprint"].astype(str) == protocol
    ].copy()
    if current.empty:
        raise RuntimeError(
            f"No {role} checkpoint rows match the conclusions fingerprint."
        )
    required_bonds = {geometry_key(value) for value in ALL_BOND_LENGTHS}
    for variant in required_variants:
        selected = current[current["variant"].astype(str) == variant]
        found_bonds = set(selected["geometry_key"].astype(str))
        if found_bonds != required_bonds:
            raise RuntimeError(
                f"{role} variant {variant} does not cover all seven geometries."
            )
        for key, group in selected.groupby("geometry_key"):
            if len(group) < minimum_restarts:
                raise RuntimeError(
                    f"{role} {variant} at {key} has {len(group)} restarts; "
                    f"at least {minimum_restarts} are required."
                )
            energies = p2b.finite_series(group, "vqe_total_energy_ha")
            if energies.isna().any():
                raise RuntimeError(f"{role} contains a nonfinite energy.")
            if float(energies.max() - energies.min()) > 1e-7:
                raise RuntimeError(
                    f"{role} is not numerically stable for {variant} at {key}."
                )
            if not group["optimizer_success"].map(p2b.as_bool).all():
                raise RuntimeError(
                    f"{role} contains an optimizer failure for {variant} at {key}."
                )
            if group["variational_violation"].map(p2b.as_bool).any():
                raise RuntimeError(
                    f"{role} contains a variational violation for {variant} at {key}."
                )
    return current, conclusions


def best_run(
    runs: pd.DataFrame,
    bond_length: float,
    variant: str,
) -> pd.Series:
    selected = runs[
        (runs["geometry_key"].astype(str) == geometry_key(bond_length))
        & (runs["variant"].astype(str) == variant)
    ].copy()
    selected["_energy"] = pd.to_numeric(
        selected["vqe_total_energy_ha"], errors="coerce"
    )
    selected = selected[np.isfinite(selected["_energy"])]
    if selected.empty:
        raise RuntimeError(f"No finite {variant} run exists at R={bond_length}.")
    return selected.sort_values("_energy").iloc[0]


def parameter_vector(row: pd.Series, expected_labels: Sequence[str]) -> np.ndarray:
    labels = [str(value) for value in json_list(row["reference_labels_json"], "labels")]
    if labels != list(expected_labels):
        raise RuntimeError(
            f"Checkpoint operator order {labels} differs from {list(expected_labels)}."
        )
    values = np.asarray(
        json_list(row["optimal_parameters_json"], "optimal parameters"),
        dtype=float,
    )
    if len(values) != len(labels) or not np.isfinite(values).all():
        raise RuntimeError("Checkpoint optimal parameters are invalid.")
    return values


def build_mechanism_geometry(
    bond_length: float,
    basis: str,
) -> GeometryMechanismData:
    from pyscf import gto, mp, scf

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
        raise RuntimeError(f"PySCF RHF failed at R={bond_length}.")

    frozen_core = (int(molecule.nelectron) - p5.EXPECTED_ACTIVE_ELECTRONS) // 2
    start = frozen_core
    stop = start + p5.EXPECTED_ACTIVE_SPATIAL_ORBITALS
    snapshot = p3.OrbitalSnapshot(
        bond_length=bond_length,
        molecule=molecule,
        active_coefficients=np.asarray(
            mean_field.mo_coeff[:, start:stop], dtype=float
        ),
        active_energies=np.asarray(
            mean_field.mo_energy[start:stop], dtype=float
        ),
        active_occupations=np.asarray(
            mean_field.mo_occ[start:stop], dtype=float
        ),
        scf_total_energy_ha=float(mean_field.e_tot),
        num_frozen_core_orbitals=frozen_core,
    )
    mp2_solver = mp.MP2(mean_field, frozen=frozen_core)
    correlation_energy, t2 = mp2_solver.kernel()
    if not bool(getattr(mp2_solver, "converged", True)):
        raise RuntimeError(f"Frozen-core MP2 failed at R={bond_length}.")
    t2_array = np.asarray(t2, dtype=float)
    if t2_array.ndim != 4:
        raise RuntimeError("Unexpected MP2 t2 tensor shape.")
    return GeometryMechanismData(
        bond_length=bond_length,
        mean_field=mean_field,
        snapshot=snapshot,
        mp2_correlation_energy_ha=float(correlation_energy),
        mp2_t2=t2_array,
    )


def ao_angular_momentum(label: Any) -> str:
    text = " ".join(str(part) for part in label).lower()
    match = re.search(r"[0-9]([spdfg])", text)
    return match.group(1) if match else "unknown"


def make_orbital_descriptors(
    geometries: dict[str, GeometryMechanismData],
    bond_lengths: Sequence[float],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bond_length in sorted(bond_lengths):
        data = geometries[geometry_key(bond_length)]
        snapshot = data.snapshot
        if snapshot.reference_to_canonical is None:
            raise RuntimeError("Orbital tracking has not been applied.")
        overlap = np.asarray(
            data.mean_field.get_ovlp(data.mean_field.mol), dtype=float
        )
        labels = data.mean_field.mol.ao_labels(fmt=False)
        atom_slices = data.mean_field.mol.aoslice_by_atom()
        for reference_orbital, current_orbital in enumerate(
            snapshot.reference_to_canonical
        ):
            coefficients = snapshot.active_coefficients[:, int(current_orbital)]
            gross = coefficients * (overlap @ coefficients)
            atom_populations: dict[str, float] = {}
            for atom_index, atom_slice in enumerate(atom_slices):
                start, stop = int(atom_slice[2]), int(atom_slice[3])
                symbol = str(data.mean_field.mol.atom_symbol(atom_index))
                atom_populations[symbol] = atom_populations.get(symbol, 0.0) + float(
                    gross[start:stop].sum()
                )
            angular: dict[str, float] = {}
            for ao_index, label in enumerate(labels):
                kind = ao_angular_momentum(label)
                angular[kind] = angular.get(kind, 0.0) + float(gross[ao_index])
            rows.append(
                {
                    "bond_length_angstrom": bond_length,
                    "geometry_key": geometry_key(bond_length),
                    "reference_active_orbital": reference_orbital,
                    "current_canonical_active_orbital": int(current_orbital),
                    "orbital_energy_ha": float(
                        snapshot.active_energies[int(current_orbital)]
                    ),
                    "occupation": float(
                        snapshot.active_occupations[int(current_orbital)]
                    ),
                    "li_mulliken_population": atom_populations.get("Li", np.nan),
                    "h_mulliken_population": atom_populations.get("H", np.nan),
                    "s_mulliken_population": angular.get("s", 0.0),
                    "p_mulliken_population": angular.get("p", 0.0),
                    "population_sum": float(gross.sum()),
                    "matched_individual_abs_overlap": float(
                        snapshot.matched_individual_overlaps[reference_orbital]
                    ),
                    "matched_group_min_singular_value": float(
                        snapshot.group_quality_by_reference_orbital[
                            reference_orbital
                        ]
                    ),
                    "tracking_reliable": snapshot.tracking_reliable,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "phase6_orbital_descriptors.csv", index=False)
    return frame


def extract_amplitudes(
    confirmation_runs: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for _, run in confirmation_runs.iterrows():
        variant = str(run["variant"])
        if variant not in {"full24", "compact10"}:
            continue
        labels = [
            str(value)
            for value in json_list(run["reference_labels_json"], "operator labels")
        ]
        parameters = parameter_vector(run, labels)
        for position, (label, value) in enumerate(zip(labels, parameters)):
            rows.append(
                {
                    "bond_length_angstrom": float(run["bond_length_angstrom"]),
                    "geometry_key": geometry_key(run["bond_length_angstrom"]),
                    "variant": variant,
                    "restart": int(run["restart"]),
                    "restart_stage": str(run["restart_stage"]),
                    "operator_position": position,
                    "operator_label": label,
                    "global_reference_id": int(
                        label[1:]
                        if label.startswith("S")
                        else p5.EXPECTED_SINGLE_COUNT + int(label[1:])
                    ),
                    "parameter_value": float(value),
                    "absolute_parameter_value": abs(float(value)),
                    "operator_in_compact10": label in COMPACT_LABELS,
                }
            )
    long_frame = pd.DataFrame(rows).sort_values(
        ["bond_length_angstrom", "variant", "operator_position", "restart"]
    )
    if long_frame.empty:
        raise RuntimeError("No Phase-5 amplitudes were extracted.")
    long_frame.to_csv(output_dir / "phase6_optimized_amplitudes.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "bond_length_angstrom",
        "geometry_key",
        "variant",
        "operator_label",
        "global_reference_id",
        "operator_in_compact10",
    ]
    for keys, group in long_frame.groupby(group_columns, dropna=False):
        values = pd.to_numeric(group["parameter_value"], errors="coerce")
        signs = np.sign(values[np.abs(values) > 1e-12])
        summary_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "num_restarts": int(len(group)),
                "mean_parameter": float(values.mean()),
                "mean_absolute_parameter": float(np.abs(values).mean()),
                "median_absolute_parameter": float(np.abs(values).median()),
                "std_parameter": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "maximum_absolute_parameter": float(np.abs(values).max()),
                "sign_consistency": (
                    np.nan if len(signs) == 0 else float(abs(signs.mean()))
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["bond_length_angstrom", "variant", "global_reference_id"]
    )
    summary.to_csv(output_dir / "phase6_amplitude_summary.csv", index=False)
    return long_frame, summary


def circuit_energy(
    system: diag.SystemData,
    ansatz: Any,
    parameter_values: Sequence[float],
) -> float:
    from qiskit.quantum_info import Statevector

    parameters = list(ansatz.parameters)
    values = np.asarray(parameter_values, dtype=float)
    if len(parameters) != len(values):
        raise ValueError("Circuit parameter-vector length mismatch.")
    bound = ansatz.assign_parameters(
        dict(zip(parameters, values)),
        inplace=False,
    )
    state = Statevector.from_instruction(bound)
    active = float(np.real(state.expectation_value(system.qubit_op)))
    return active + system.total_offset_ha


def finite_difference_response(
    system: diag.SystemData,
    excitation: diag.Excitation,
    step: float,
) -> tuple[float, float, float]:
    ansatz = p4.build_ansatz(system, [excitation], reps=1)
    zero = circuit_energy(system, ansatz, [0.0])
    plus = circuit_energy(system, ansatz, [step])
    minus = circuit_energy(system, ansatz, [-step])
    gradient = (plus - minus) / (2.0 * step)
    curvature = (plus - 2.0 * zero + minus) / (step * step)
    return gradient, curvature, zero


def mp2_amplitude(
    data: GeometryMechanismData,
    excitation: diag.Excitation,
) -> float | None:
    occupied, virtual = excitation
    if len(occupied) != 2:
        return None
    n_spatial = p5.EXPECTED_ACTIVE_SPATIAL_ORBITALS
    virtual_spatial = [int(index) % n_spatial for index in virtual]
    virtual_indices = [index - 1 for index in virtual_spatial]
    if any(index < 0 for index in virtual_indices):
        return None
    t2 = data.mp2_t2
    if (
        t2.shape[0] < 1
        or t2.shape[1] < 1
        or max(virtual_indices) >= t2.shape[2]
        or max(virtual_indices) >= t2.shape[3]
    ):
        raise RuntimeError(
            f"MP2 tensor shape {t2.shape} is incompatible with {excitation}."
        )
    return float(t2[0, 0, virtual_indices[0], virtual_indices[1]])


def make_operator_mechanism_table(
    systems: dict[str, diag.SystemData],
    geometries: dict[str, GeometryMechanismData],
    mappings: dict[tuple[str, str], dict[str, Any]],
    amplitude_summary: pd.DataFrame,
    bond_lengths: Sequence[float],
    finite_difference_step: float,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    compact_ids = set(
        p5.VARIANT_BY_NAME["compact10"].reference_global_ids
    )
    for bond_length in sorted(bond_lengths):
        key = geometry_key(bond_length)
        system = systems[key]
        data = geometries[key]
        mapping = mappings[(key, "full24")]
        reference_mapping = mappings[
            (geometry_key(p5.REFERENCE_BOND_LENGTH), "full24")
        ]
        for position, global_id in enumerate(mapping["reference_global_ids"]):
            excitation = mapping["mapped_excitations"][position]
            mapped_id = mapping["mapped_current_global_ids"][position]
            occupied, virtual = excitation
            n_spatial = system.problem.num_spatial_orbitals
            occupied_spatial = [int(index) % n_spatial for index in occupied]
            virtual_spatial = [int(index) % n_spatial for index in virtual]
            gap = float(
                sum(data.snapshot.active_energies[index] for index in virtual_spatial)
                - sum(
                    data.snapshot.active_energies[index]
                    for index in occupied_spatial
                )
            )
            gradient, curvature, hf_from_circuit = finite_difference_response(
                system,
                excitation,
                finite_difference_step,
            )
            label = p5.operator_label(global_id)
            full_amplitude = amplitude_summary[
                (amplitude_summary["geometry_key"] == key)
                & (amplitude_summary["variant"] == "full24")
                & (amplitude_summary["operator_label"] == label)
            ]
            compact_amplitude = amplitude_summary[
                (amplitude_summary["geometry_key"] == key)
                & (amplitude_summary["variant"] == "compact10")
                & (amplitude_summary["operator_label"] == label)
            ]
            rows.append(
                {
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "operator_label": label,
                    "global_reference_id": global_id,
                    "mapped_current_global_id": mapped_id,
                    "excitation_type": p5.operator_type(global_id),
                    "operator_in_compact10": global_id in compact_ids,
                    "reference_excitation_json": json.dumps(
                        diag.excitation_to_json(
                            reference_mapping["mapped_excitations"][global_id]
                        )
                    ),
                    "mapped_excitation_json": json.dumps(
                        diag.excitation_to_json(excitation)
                    ),
                    "current_occupied_spatial_json": json.dumps(occupied_spatial),
                    "current_virtual_spatial_json": json.dumps(virtual_spatial),
                    "orbital_energy_denominator_ha": gap,
                    "hf_finite_difference_gradient_ha": gradient,
                    "hf_finite_difference_curvature_ha": curvature,
                    "hf_energy_residual_ha": (
                        hf_from_circuit - system.hf_total_from_qubit_ha
                    ),
                    "frozen_core_mp2_t2": mp2_amplitude(data, excitation),
                    "full24_mean_absolute_parameter": (
                        np.nan
                        if full_amplitude.empty
                        else float(
                            full_amplitude.iloc[0][
                                "mean_absolute_parameter"
                            ]
                        )
                    ),
                    "full24_std_parameter": (
                        np.nan
                        if full_amplitude.empty
                        else float(full_amplitude.iloc[0]["std_parameter"])
                    ),
                    "compact10_mean_absolute_parameter": (
                        np.nan
                        if compact_amplitude.empty
                        else float(
                            compact_amplitude.iloc[0][
                                "mean_absolute_parameter"
                            ]
                        )
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["bond_length_angstrom", "global_reference_id"]
    )
    if float(frame["hf_energy_residual_ha"].abs().max()) > 1e-8:
        raise RuntimeError("Single-operator HF circuits do not reproduce the HF energy.")
    frame.to_csv(output_dir / "phase6_operator_mechanism.csv", index=False)
    return frame


def conditional_channel_response(
    system: diag.SystemData,
    channel_excitations: Sequence[diag.Excitation],
    d6_excitations: Sequence[diag.Excitation],
    d6_parameters: np.ndarray,
    finite_difference_step: float,
    optimizer_tolerance: float,
    maxiter: int,
) -> dict[str, Any]:
    excitations = [*channel_excitations, *d6_excitations]
    ansatz = p4.build_ansatz(system, excitations, reps=1)
    if len(d6_parameters) != len(d6_excitations):
        raise RuntimeError("D6 parameter-vector length mismatch.")

    def values(single_values: Sequence[float]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(single_values, dtype=float), d6_parameters]
        )

    zero = np.zeros(len(channel_excitations), dtype=float)
    base_energy = circuit_energy(system, ansatz, values(zero))
    gradients: list[float] = []
    curvatures: list[float] = []
    for index in range(len(channel_excitations)):
        plus = zero.copy()
        minus = zero.copy()
        plus[index] = finite_difference_step
        minus[index] = -finite_difference_step
        e_plus = circuit_energy(system, ansatz, values(plus))
        e_minus = circuit_energy(system, ansatz, values(minus))
        gradients.append(
            (e_plus - e_minus) / (2.0 * finite_difference_step)
        )
        curvatures.append(
            (e_plus - 2.0 * base_energy + e_minus)
            / (finite_difference_step * finite_difference_step)
        )

    evaluations = 0

    def objective(single_values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return circuit_energy(system, ansatz, values(single_values))

    result = minimize(
        objective,
        zero,
        method="SLSQP",
        bounds=[(-math.pi, math.pi)] * len(zero),
        options={"maxiter": maxiter, "ftol": optimizer_tolerance},
        tol=optimizer_tolerance,
    )
    best_energy = float(result.fun)
    return {
        "d6_base_energy_from_channel_circuit_ha": base_energy,
        "optimized_channel_energy_ha": best_energy,
        "conditional_energy_lowering_ha": base_energy - best_energy,
        "conditional_gradient_norm_ha": float(np.linalg.norm(gradients)),
        "conditional_gradients_json": json.dumps(gradients),
        "conditional_curvatures_json": json.dumps(curvatures),
        "optimized_channel_parameters_json": json.dumps(
            np.asarray(result.x, dtype=float).tolist()
        ),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "num_objective_evaluations": int(
            getattr(result, "nfev", evaluations)
        ),
        "num_iterations": int(getattr(result, "nit", -1)),
    }


def make_conditional_single_channels(
    systems: dict[str, diag.SystemData],
    mappings: dict[tuple[str, str], dict[str, Any]],
    ablation_runs: pd.DataFrame,
    bond_lengths: Sequence[float],
    finite_difference_step: float,
    optimizer_tolerance: float,
    maxiter: int,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_reference_ids = list(
        p5.VARIANT_BY_NAME["full24"].reference_global_ids
    )
    for bond_length in sorted(bond_lengths):
        key = geometry_key(bond_length)
        system = systems[key]
        d6_run = best_run(ablation_runs, bond_length, "D6_only")
        d6_parameters = parameter_vector(d6_run, D6_LABELS)
        d6_mapping = mappings[(key, "D6_only")]
        d6_excitations = d6_mapping["mapped_excitations"]
        full_mapping = mappings[(key, "full24")]
        excitation_by_label = {
            p5.operator_label(global_id): excitation
            for global_id, excitation in zip(
                full_reference_ids,
                full_mapping["mapped_excitations"],
            )
        }
        for channel, labels in SINGLE_CHANNELS:
            response = conditional_channel_response(
                system,
                [excitation_by_label[label] for label in labels],
                d6_excitations,
                d6_parameters,
                finite_difference_step,
                optimizer_tolerance,
                maxiter,
            )
            d6_checkpoint_energy = float(d6_run["vqe_total_energy_ha"])
            residual = (
                response["d6_base_energy_from_channel_circuit_ha"]
                - d6_checkpoint_energy
            )
            if abs(residual) > 1e-7:
                raise RuntimeError(
                    f"Conditional {channel} circuit at R={bond_length} does "
                    f"not reproduce the D6 checkpoint ({residual:.3e} Ha)."
                )
            rows.append(
                {
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "single_channel": channel,
                    "single_labels_json": json.dumps(list(labels)),
                    "phase5_empirically_active_channel": (
                        channel in ACTIVE_SINGLE_CHANNELS
                    ),
                    "d6_checkpoint_restart": int(d6_run["restart"]),
                    "d6_checkpoint_energy_ha": d6_checkpoint_energy,
                    "d6_circuit_reconstruction_residual_ha": residual,
                    "exact_total_energy_ha": system.exact_total_ha,
                    **response,
                    "conditional_best_error_exact_ha": abs(
                        response["optimized_channel_energy_ha"]
                        - system.exact_total_ha
                    ),
                    "conditional_channel_reaches_chemical_accuracy": bool(
                        abs(
                            response["optimized_channel_energy_ha"]
                            - system.exact_total_ha
                        )
                        <= diag.CHEMICAL_ACCURACY_HA
                    ),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["bond_length_angstrom", "single_channel"]
    )
    frame.to_csv(
        output_dir / "phase6_conditional_single_channels.csv",
        index=False,
    )
    return frame


def loo_protocol_fingerprint(
    confirmation_protocol: str,
    target_bond_lengths: Sequence[float],
    operator_labels: Sequence[str],
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    random_restarts: int,
) -> str:
    payload = {
        "analysis_phase": "Phase 6 compact10 leave-one-out",
        "confirmation_protocol_fingerprint": confirmation_protocol,
        "target_bond_lengths_angstrom": list(target_bond_lengths),
        "compact_labels": list(COMPACT_LABELS),
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        "parameter_seed": parameter_seed,
        "warm_start_restarts": 1,
        "random_restarts": random_restarts,
        "checkpoint_geometry_key_policy": (
            "canonicalize_from_bond_length_angstrom_to_six_decimals"
        ),
    }
    # Preserve the original pilot fingerprint when all compact10 operators are
    # selected so an existing 60-run checkpoint remains resumable.
    if list(operator_labels) != list(COMPACT_LABELS):
        payload["selected_operator_labels"] = list(operator_labels)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def canonicalize_loo_checkpoint(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {
        "protocol_fingerprint",
        "bond_length_angstrom",
        "omitted_operator_label",
        "restart",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "The Phase-6 checkpoint is missing columns: " + ", ".join(missing)
        )
    result = frame.copy()
    bonds = pd.to_numeric(result["bond_length_angstrom"], errors="coerce")
    restarts = pd.to_numeric(result["restart"], errors="coerce")
    if bonds.isna().any() or restarts.isna().any():
        raise RuntimeError("The Phase-6 checkpoint contains invalid numeric keys.")
    if not np.allclose(restarts, np.round(restarts)):
        raise RuntimeError("The Phase-6 checkpoint contains noninteger restart IDs.")
    result["bond_length_angstrom"] = bonds.astype(float)
    result["geometry_key"] = bonds.map(geometry_key)
    result["restart"] = restarts.astype(int)
    key_columns = [
        "protocol_fingerprint",
        "geometry_key",
        "omitted_operator_label",
        "restart",
    ]
    duplicates = result.duplicated(key_columns, keep=False)
    if duplicates.any():
        values = (
            result.loc[duplicates, key_columns]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "The Phase-6 checkpoint contains duplicate canonical run keys: "
            f"{values[:5]}"
        )
    return result


def run_leave_one_out(
    systems: dict[str, diag.SystemData],
    mappings: dict[tuple[str, str], dict[str, Any]],
    confirmation_runs: pd.DataFrame,
    confirmation_protocol: str,
    target_bond_lengths: Sequence[float],
    operator_labels: Sequence[str],
    output_dir: Path,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    random_restarts: int,
    resume: bool,
) -> tuple[pd.DataFrame, str]:
    protocol = loo_protocol_fingerprint(
        confirmation_protocol,
        target_bond_lengths,
        operator_labels,
        optimizer,
        maxiter,
        optimizer_tolerance,
        initial_scale,
        parameter_seed,
        random_restarts,
    )
    path = output_dir / "phase6_leave_one_out_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if resume and path.exists():
        previous = canonicalize_loo_checkpoint(pd.read_csv(path))
        rows = previous.to_dict(orient="records")
        matching = previous[
            previous["protocol_fingerprint"].astype(str) == protocol
        ]
        completed = {
            (
                geometry_key(row["bond_length_angstrom"]),
                str(row["omitted_operator_label"]),
                int(row["restart"]),
            )
            for row in matching.to_dict(orient="records")
        }
        print(f"Resuming with {len(completed)} completed Phase-6 LOO runs.")

    total_restarts = 1 + random_restarts
    for geometry_index, bond_length in enumerate(sorted(target_bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        compact_run = best_run(confirmation_runs, bond_length, "compact10")
        compact_parameters = parameter_vector(compact_run, COMPACT_LABELS)
        compact_mapping = mappings[(key, "compact10")]
        compact_excitations = compact_mapping["mapped_excitations"]
        for omitted_label in operator_labels:
            omitted_position = COMPACT_LABELS.index(omitted_label)
            retained_labels = [
                label for label in COMPACT_LABELS if label != omitted_label
            ]
            retained_excitations = [
                excitation
                for index, excitation in enumerate(compact_excitations)
                if index != omitted_position
            ]
            warm_start = np.delete(compact_parameters, omitted_position)
            ansatz = p4.build_ansatz(system, retained_excitations, reps=1)
            resources = diag.circuit_resource_metrics(
                ansatz,
                parameter_seed
                + 10_000 * geometry_index
                + omitted_position,
            )
            for restart in range(total_restarts):
                completed_key = (key, omitted_label, restart)
                if completed_key in completed:
                    continue
                restart_seed = (
                    parameter_seed
                    + 1_000_003 * geometry_index
                    + 100_003 * omitted_position
                    + 10_007 * restart
                )
                if restart == 0:
                    initial = warm_start
                    initialization = "compact10_parameter_deletion_warm_start"
                    initialization_seed: int | None = None
                else:
                    rng = np.random.default_rng(restart_seed)
                    initial = rng.uniform(
                        -initial_scale,
                        initial_scale,
                        size=ansatz.num_parameters,
                    )
                    initialization = "uniform_random"
                    initialization_seed = restart_seed
                base = {
                    "protocol_fingerprint": protocol,
                    "confirmation_protocol_fingerprint": confirmation_protocol,
                    "configuration_fingerprint": system.fingerprint,
                    "geometry_index": geometry_index,
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "omitted_operator_label": omitted_label,
                    "omitted_operator_type": (
                        "single" if omitted_label.startswith("S") else "double"
                    ),
                    "retained_labels_json": json.dumps(retained_labels),
                    "num_parameters": int(ansatz.num_parameters),
                    "restart": restart,
                    "restart_stage": (
                        "warm_start" if restart == 0 else "random_confirmation"
                    ),
                    "initialization": initialization,
                    "initialization_seed": initialization_seed,
                    "compact10_baseline_restart": int(compact_run["restart"]),
                    "compact10_baseline_energy_ha": float(
                        compact_run["vqe_total_energy_ha"]
                    ),
                    "exact_total_energy_ha": system.exact_total_ha,
                    "optimizer": optimizer,
                    "maxiter": maxiter,
                    "optimizer_tolerance": optimizer_tolerance,
                    "initial_scale": initial_scale,
                    **resources,
                }
                print(
                    f"R={bond_length:.3f} omit={omitted_label} "
                    f"restart={restart + 1}/{total_restarts} "
                    f"parameters={ansatz.num_parameters} "
                    f"initialization={initialization}",
                    flush=True,
                )
                try:
                    outcome = diag.run_one_optimization(
                        system,
                        ansatz,
                        initial,
                        optimizer,
                        maxiter,
                        optimizer_tolerance,
                    )
                    row = {**base, **outcome, "run_exception": None}
                except Exception as exc:
                    print(
                        f"ERROR in LOO R={bond_length}, omit={omitted_label}, "
                        f"restart={restart}: {exc}",
                        file=sys.stderr,
                    )
                    row = p2b.failure_row(base, exc)
                rows.append(row)
                completed.add(completed_key)
                p2b.checkpoint(path, rows)
    current = canonicalize_loo_checkpoint(pd.DataFrame(rows))
    return (
        current[
            current["protocol_fingerprint"].astype(str) == protocol
        ].copy(),
        protocol,
    )


def summarize_leave_one_out(
    runs: pd.DataFrame,
    target_bond_lengths: Sequence[float],
    operator_labels: Sequence[str],
    random_restarts: int,
    disagreement_tolerance_ha: float,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected = 1 + random_restarts
    for bond_length in sorted(target_bond_lengths):
        key = geometry_key(bond_length)
        for omitted_label in operator_labels:
            group = runs[
                (runs["geometry_key"].astype(str) == key)
                & (
                    runs["omitted_operator_label"].astype(str)
                    == omitted_label
                )
            ].sort_values("restart")
            if len(group) != expected:
                raise RuntimeError(
                    f"LOO {omitted_label} at R={bond_length} has "
                    f"{len(group)}/{expected} attempts."
                )
            energies = p2b.finite_series(group, "vqe_total_energy_ha")
            errors = p2b.finite_series(group, "absolute_error_ha")
            success = group["optimizer_success"].map(p2b.as_bool).all()
            finite = energies.notna().all() and errors.notna().all()
            variational_violation = bool(
                group["variational_violation"].map(p2b.as_bool).any()
            )
            runs_valid = bool(success and finite and not variational_violation)
            spread = (
                float(energies.max() - energies.min())
                if finite
                else np.nan
            )
            energy_stable = bool(
                runs_valid
                and spread <= disagreement_tolerance_ha
            )
            baseline = float(group["compact10_baseline_energy_ha"].iloc[0])
            baseline_values = pd.to_numeric(
                group["compact10_baseline_energy_ha"],
                errors="coerce",
            )
            if (
                baseline_values.isna().any()
                or float(baseline_values.max() - baseline_values.min()) > 1e-10
            ):
                raise RuntimeError(
                    f"LOO {omitted_label} at R={bond_length} has inconsistent "
                    "compact10 baseline energies."
                )
            best_energy = float(energies.min()) if finite else np.nan
            worst_energy = float(energies.max()) if finite else np.nan
            best_error = float(errors.min()) if finite else np.nan
            best_penalty = best_energy - baseline if finite else np.nan
            worst_penalty = worst_energy - baseline if finite else np.nan
            strict_flags = (
                (energies - baseline).abs()
                <= p5.STRICT_FULL24_TOLERANCE_HA
            )
            chemical_flags = errors <= diag.CHEMICAL_ACCURACY_HA
            strict_classification_stable = bool(
                runs_valid and strict_flags.nunique() == 1
            )
            chemical_classification_stable = bool(
                runs_valid and chemical_flags.nunique() == 1
            )
            strict_retained: bool | float = np.nan
            chemical_retained: bool | float = np.nan
            if strict_classification_stable:
                strict_retained = bool(strict_flags.iloc[0])
            if chemical_classification_stable:
                chemical_retained = bool(chemical_flags.iloc[0])
            if energy_stable:
                resolution = "energy_and_threshold_classifications_stable"
            elif (
                strict_classification_stable
                and chemical_classification_stable
            ):
                resolution = (
                    "threshold_classifications_stable_energy_penalty_unresolved"
                )
            else:
                resolution = "threshold_classification_unresolved"
            first = group.iloc[0]
            rows.append(
                {
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "omitted_operator_label": omitted_label,
                    "omitted_operator_type": first["omitted_operator_type"],
                    "num_parameters": int(first["num_parameters"]),
                    "completed_attempts": int(len(group)),
                    "optimizer_success_all_restarts": bool(success),
                    "all_runs_valid": runs_valid,
                    "analysis_stable": energy_stable,
                    "energy_stable": energy_stable,
                    "chemical_classification_stable": (
                        chemical_classification_stable
                    ),
                    "strict_classification_stable": (
                        strict_classification_stable
                    ),
                    "resolution_status": resolution,
                    "energy_spread_ha": spread,
                    "compact10_baseline_energy_ha": baseline,
                    "best_leave_one_out_energy_ha": best_energy,
                    "worst_leave_one_out_energy_ha": worst_energy,
                    "best_leave_one_out_error_exact_ha": best_error,
                    "best_energy_penalty_vs_compact10_ha": best_penalty,
                    "worst_energy_penalty_vs_compact10_ha": worst_penalty,
                    # Backward-compatible alias used by the Phase-6 report.
                    "energy_penalty_vs_compact10_ha": best_penalty,
                    "strict_equivalence_retained": strict_retained,
                    "chemical_accuracy_retained": chemical_retained,
                    "operator_required_for_strict_equivalence": (
                        np.nan
                        if not strict_classification_stable
                        else not bool(strict_retained)
                    ),
                    "operator_required_for_chemical_accuracy": (
                        np.nan
                        if not chemical_classification_stable
                        else not bool(chemical_retained)
                    ),
                    "compiled_depth": first.get("compiled_depth"),
                    "compiled_cx": first.get("compiled_cx"),
                    "any_variational_violation": variational_violation,
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["bond_length_angstrom", "omitted_operator_label"]
    )
    frame.to_csv(
        output_dir / "phase6_leave_one_out_summary.csv",
        index=False,
    )
    return frame


def nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def make_conclusions(
    confirmation_conclusions: dict[str, Any],
    ablation_conclusions: dict[str, Any],
    orbital_descriptors: pd.DataFrame,
    operator_table: pd.DataFrame,
    conditional_channels: pd.DataFrame,
    loo_summary: pd.DataFrame | None,
    loo_protocol: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    active = operator_table[
        operator_table["operator_in_compact10"].map(p2b.as_bool)
    ]
    inactive = operator_table[
        ~operator_table["operator_in_compact10"].map(p2b.as_bool)
    ]
    active_amp = pd.to_numeric(
        active["full24_mean_absolute_parameter"], errors="coerce"
    )
    inactive_amp = pd.to_numeric(
        inactive["full24_mean_absolute_parameter"], errors="coerce"
    )
    single_rows = operator_table[
        operator_table["excitation_type"] == "single"
    ]
    double_rows = operator_table[
        operator_table["excitation_type"] == "double"
    ]
    channel_results: dict[str, Any] = {}
    for channel, group in conditional_channels.groupby("single_channel"):
        lowerings = pd.to_numeric(
            group["conditional_energy_lowering_ha"], errors="coerce"
        )
        channel_results[str(channel)] = {
            "phase5_empirically_active_channel": bool(
                group["phase5_empirically_active_channel"]
                .map(p2b.as_bool)
                .iloc[0]
            ),
            "maximum_conditional_energy_lowering_ha": nullable_float(
                lowerings.max()
            ),
            "median_conditional_energy_lowering_ha": nullable_float(
                lowerings.median()
            ),
            "all_channel_optimizations_successful": bool(
                group["optimizer_success"].map(p2b.as_bool).all()
            ),
            "chemical_accuracy_bond_lengths_angstrom": group[
                group[
                    "conditional_channel_reaches_chemical_accuracy"
                ].map(p2b.as_bool)
            ]["bond_length_angstrom"].astype(float).tolist(),
        }

    loo_results: dict[str, Any] | None = None
    if loo_summary is not None:
        energy_stable = loo_summary["energy_stable"].map(p2b.as_bool)
        chemical_stable = loo_summary[
            "chemical_classification_stable"
        ].map(p2b.as_bool)
        strict_stable = loo_summary[
            "strict_classification_stable"
        ].map(p2b.as_bool)
        threshold_classifications_resolved = bool(
            chemical_stable.all() and strict_stable.all()
        )
        loo_results = {
            "protocol_fingerprint": loo_protocol,
            "num_tests": int(len(loo_summary)),
            # Backward-compatible fields: "stable" here means energy stable.
            "num_stable": int(energy_stable.sum()),
            "all_tests_stable": bool(energy_stable.all()),
            "num_energy_stable": int(energy_stable.sum()),
            "all_tests_energy_stable": bool(energy_stable.all()),
            "num_chemical_classifications_stable": int(
                chemical_stable.sum()
            ),
            "all_chemical_classifications_stable": bool(
                chemical_stable.all()
            ),
            "num_strict_classifications_stable": int(strict_stable.sum()),
            "all_strict_classifications_stable": bool(strict_stable.all()),
            "all_threshold_classifications_resolved": (
                threshold_classifications_resolved
            ),
            "energy_unresolved_tests": [
                {
                    "bond_length_angstrom": float(
                        row["bond_length_angstrom"]
                    ),
                    "omitted_operator_label": str(
                        row["omitted_operator_label"]
                    ),
                    "energy_spread_ha": float(row["energy_spread_ha"]),
                    "chemical_classification_stable": p2b.as_bool(
                        row["chemical_classification_stable"]
                    ),
                    "strict_classification_stable": p2b.as_bool(
                        row["strict_classification_stable"]
                    ),
                }
                for _, row in loo_summary[
                    ~energy_stable
                ].iterrows()
            ],
            "operators_required_for_strict_equivalence_at_any_tested_geometry": sorted(
                loo_summary[
                    loo_summary[
                        "operator_required_for_strict_equivalence"
                    ].map(p2b.as_bool)
                ]["omitted_operator_label"].astype(str).unique().tolist()
            ),
            "operators_required_for_chemical_accuracy_at_any_tested_geometry": sorted(
                loo_summary[
                    loo_summary[
                        "operator_required_for_chemical_accuracy"
                    ].map(p2b.as_bool)
                ]["omitted_operator_label"].astype(str).unique().tolist()
            ),
            "maximum_energy_penalty_by_operator_ha": {
                str(label): nullable_float(
                    pd.to_numeric(
                        group["energy_penalty_vs_compact10_ha"],
                        errors="coerce",
                    ).max()
                )
                for label, group in loo_summary.groupby(
                    "omitted_operator_label"
                )
            },
        }

    if loo_summary is None:
        analysis_status = "mechanism_descriptors_complete"
    elif loo_results is not None and loo_results["all_tests_energy_stable"]:
        analysis_status = "complete_with_leave_one_out"
    elif (
        loo_results is not None
        and loo_results["all_threshold_classifications_resolved"]
    ):
        analysis_status = (
            "complete_for_threshold_classification_"
            "energy_penalties_partially_unresolved"
        )
    else:
        analysis_status = "leave_one_out_partially_unresolved"

    conclusions = {
        "analysis_status": analysis_status,
        "confirmation_protocol_fingerprint": confirmation_conclusions[
            "protocol_fingerprint"
        ],
        "ablation_protocol_fingerprint": ablation_conclusions[
            "protocol_fingerprint"
        ],
        "all_orbital_mappings_reliable": bool(
            orbital_descriptors["tracking_reliable"].map(p2b.as_bool).all()
        ),
        "candidate_invariant_rule": {
            "spin_complete_singles_to_reference_virtual_orbitals": [1, 4],
            "paired_doubles_to_each_reference_virtual_orbital": [1, 2, 3, 4],
            "complementary_cross_doubles_between_reference_virtual_orbitals": [
                [1, 4],
                [4, 1],
            ],
            "compact_operator_labels": list(COMPACT_LABELS),
        },
        "optimized_amplitude_diagnostic": {
            "median_full24_absolute_amplitude_active_operators": nullable_float(
                active_amp.median()
            ),
            "median_full24_absolute_amplitude_inactive_operators": nullable_float(
                inactive_amp.median()
            ),
            "active_to_inactive_median_ratio": (
                None
                if nullable_float(inactive_amp.median()) in {None, 0.0}
                else float(active_amp.median() / inactive_amp.median())
            ),
            "warning": (
                "Optimized amplitudes depend on parameterization and operator "
                "ordering; they are descriptive evidence, not standalone causality."
            ),
        },
        "hf_response_diagnostic": {
            "maximum_absolute_single_gradient_ha": nullable_float(
                pd.to_numeric(
                    single_rows["hf_finite_difference_gradient_ha"],
                    errors="coerce",
                ).abs().max()
            ),
            "maximum_absolute_double_gradient_ha": nullable_float(
                pd.to_numeric(
                    double_rows["hf_finite_difference_gradient_ha"],
                    errors="coerce",
                ).abs().max()
            ),
            "interpretation": (
                "Canonical-HF single gradients are not a valid standalone "
                "screen because Brillouin's theorem suppresses their first-order "
                "response. Conditional correlated-state tests carry the relevant "
                "singles evidence."
            ),
        },
        "conditional_single_channel_results": channel_results,
        "leave_one_out_results": loo_results,
        "claim_boundary": (
            "This mechanism analysis concerns the audited frozen-core "
            "LiH/STO-3G Hamiltonians, canonical RHF orbitals tracked by maximum "
            "overlap, Jordan-Wigner mapping, product-form one-repetition UCC, "
            "statevector energies, and local numerical optimization. Mulliken "
            "populations, amplitudes, finite-difference responses, and MP2 "
            "amplitudes are diagnostics rather than basis-invariant proofs. "
            "Leave-one-out reoptimization supplies direct within-model causal "
            "evidence but does not establish transfer to another basis, active "
            "space, molecule, or hardware setting."
        ),
    }
    diag.write_json(
        output_dir / "phase6_mechanism_conclusions.json",
        conclusions,
    )
    return conclusions


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    conclusions: dict[str, Any],
    conditional_channels: pd.DataFrame,
    loo_summary: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    lines = [
        "# LiH Phase 6 Mechanism Report",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        "## Candidate invariant support rule",
        "",
        (
            "Retain spin-complete singles into tracked reference virtual "
            "orbitals 1 and 4; paired doubles into all four reference virtual "
            "orbitals; and the complementary 1→4 and 4→1 cross doubles."
        ),
        "",
        "## Conditional spin-paired-single channels",
        "",
        "| Channel | Operators | Phase-5 active | Maximum lowering (Ha) | "
        "Chemical geometries |",
        "|---|---|:---:|---:|---|",
    ]
    for channel, labels in SINGLE_CHANNELS:
        result = conclusions["conditional_single_channel_results"][channel]
        lines.append(
            f"| {channel} | {', '.join(labels)} | "
            f"{result['phase5_empirically_active_channel']} | "
            f"{fmt(result['maximum_conditional_energy_lowering_ha'], 9)} | "
            f"{result['chemical_accuracy_bond_lengths_angstrom']} |"
        )
    if loo_summary is not None:
        lines.extend(
            [
                "",
                "## Compact10 leave-one-out results",
                "",
                "| R (Å) | Omitted | Penalty (Ha) | Chemical retained | "
                "Strict retained | Energy stable | Thresholds stable |",
                "|---:|---|---:|:---:|:---:|:---:|:---:|",
            ]
        )
        for _, row in loo_summary.iterrows():
            lines.append(
                f"| {float(row['bond_length_angstrom']):.3f} | "
                f"{row['omitted_operator_label']} | "
                f"{fmt(row['energy_penalty_vs_compact10_ha'], 9)} | "
                f"{row['chemical_accuracy_retained']} | "
                f"{row['strict_equivalence_retained']} | "
                f"{row['energy_stable']} | "
                f"{bool(row['chemical_classification_stable']) and bool(row['strict_classification_stable'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            conclusions["claim_boundary"],
            "",
        ]
    )
    (output_dir / "phase6_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explain the orbital, amplitude, perturbative, conditional, and "
            "leave-one-out mechanism behind the compact10 LiH support."
        )
    )
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        default=Path("lih_phase5_confirmatory_5restart"),
        help="Phase-5 five-restart full24/compact10 output directory.",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("lih_phase5_full"),
        help="Phase-5 seven-variant full-grid output directory.",
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="pilot",
        help=(
            "Controls leave-one-out geometries only. pilot: 1.595, 2.5, 3.0; "
            "full: all seven. Descriptive diagnostics always use all seven."
        ),
    )
    parser.add_argument(
        "--loo-bond-length",
        action="append",
        type=float,
        help="Override leave-one-out geometries; repeat as needed.",
    )
    parser.add_argument(
        "--loo-operator",
        action="append",
        choices=list(COMPACT_LABELS),
        help=(
            "Restrict leave-one-out to this compact10 operator; repeat as "
            "needed. Default: all ten operators."
        ),
    )
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--finite-difference-step", type=float, default=1e-5)
    parser.add_argument("--conditional-maxiter", type=int, default=300)
    parser.add_argument(
        "--loo-random-restarts",
        type=int,
        default=1,
        help="Random confirmations in addition to one compact10 warm start.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["SLSQP", "COBYLA", "L-BFGS-B"],
        default="SLSQP",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-9)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--parameter-seed", type=int, default=20260728)
    parser.add_argument("--disagreement-tolerance", type=float, default=1e-7)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument("--orbital-overlap-threshold", type=float, default=0.75)
    parser.add_argument("--degeneracy-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--skip-leave-one-out",
        action="store_true",
        help="Generate mechanism descriptors without LOO VQE optimizations.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit Phase-5 inputs, Hamiltonians, pools, and mappings, then stop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lih_phase6_mechanism"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume matching leave-one-out checkpoint runs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.finite_difference_step <= 0:
        raise ValueError("The finite-difference step must be positive.")
    if args.conditional_maxiter < 1 or args.maxiter < 1:
        raise ValueError("Iteration limits must be positive.")
    if args.loo_random_restarts < 0:
        raise ValueError("LOO random restarts cannot be negative.")
    if args.optimizer_tolerance <= 0 or args.disagreement_tolerance <= 0:
        raise ValueError("Optimization tolerances must be positive.")
    if args.initial_scale <= 0:
        raise ValueError("The initial scale must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirmation_runs, confirmation_conclusions = read_phase5_inputs(
        args.confirmation_dir,
        ["full24", "compact10"],
        minimum_restarts=5,
        role="five-restart confirmation",
    )
    ablation_runs, ablation_conclusions = read_phase5_inputs(
        args.ablation_dir,
        [variant.name for variant in p5.VARIANTS],
        minimum_restarts=2,
        role="seven-variant ablation",
    )
    if (
        confirmation_conclusions["operator_pool_fingerprint"]
        != ablation_conclusions["operator_pool_fingerprint"]
    ):
        raise RuntimeError("Phase-5 input operator-pool fingerprints differ.")

    systems: dict[str, diag.SystemData] = {}
    pools: dict[str, list[diag.Excitation]] = {}
    geometries: dict[str, GeometryMechanismData] = {}
    audits: list[dict[str, Any]] = []
    print("Building and auditing Phase-6 Hamiltonians and orbitals...", flush=True)
    for bond_length in ALL_BOND_LENGTHS:
        key = geometry_key(bond_length)
        system = diag.build_system(p5.build_config(bond_length, args.basis))
        _, _, combined = p5.full_pool(system)
        reported = (
            [p5.EXPECTED_REFERENCE_EXACT_TOTAL_HA]
            if abs(bond_length - p5.REFERENCE_BOND_LENGTH) <= 1e-9
            else []
        )
        audit = diag.validate_reference(
            system,
            reported,
            args.audit_tolerance,
        )
        if not audit["audit_passed"]:
            raise RuntimeError(f"Hamiltonian audit failed at R={bond_length}.")
        phase5_rows = confirmation_runs[
            confirmation_runs["geometry_key"] == key
        ]
        fingerprints = set(
            phase5_rows["configuration_fingerprint"].astype(str)
        )
        if fingerprints != {system.fingerprint}:
            raise RuntimeError(
                f"Phase-5/6 configuration fingerprint mismatch at R={bond_length}."
            )
        data = build_mechanism_geometry(bond_length, args.basis)
        hf_residual = abs(
            data.mean_field.e_tot - system.hf_total_driver_ha
        )
        if hf_residual > args.audit_tolerance:
            raise RuntimeError(
                f"PySCF/Qiskit HF mismatch {hf_residual:.3e} Ha "
                f"at R={bond_length}."
            )
        systems[key] = system
        pools[key] = combined
        geometries[key] = data
        audits.append(audit)

    snapshots = {
        key: data.snapshot for key, data in geometries.items()
    }
    reference_groups, orbital_tracking, orbital_mapping_summary = (
        p3.track_orbitals(
            snapshots,
            ALL_BOND_LENGTHS,
            args.orbital_overlap_threshold,
            args.degeneracy_tolerance,
        )
    )
    if not orbital_mapping_summary["tracking_reliable"].map(p2b.as_bool).all():
        raise RuntimeError("At least one Phase-6 orbital mapping is unreliable.")
    mappings, excitation_mapping = p5.build_excitation_mappings(
        systems,
        pools,
        snapshots,
        ALL_BOND_LENGTHS,
        reference_groups,
    )
    orbital_tracking.to_csv(
        args.output_dir / "phase6_orbital_tracking.csv",
        index=False,
    )
    orbital_mapping_summary.to_csv(
        args.output_dir / "phase6_orbital_mapping_summary.csv",
        index=False,
    )
    excitation_mapping.to_csv(
        args.output_dir / "phase6_excitation_mapping.csv",
        index=False,
    )
    diag.write_json(args.output_dir / "phase6_reference_audits.json", audits)
    input_audit = {
        "audit_passed": True,
        "confirmation_protocol_fingerprint": confirmation_conclusions[
            "protocol_fingerprint"
        ],
        "ablation_protocol_fingerprint": ablation_conclusions[
            "protocol_fingerprint"
        ],
        "operator_pool_fingerprint": confirmation_conclusions[
            "operator_pool_fingerprint"
        ],
        "bond_lengths_angstrom": list(ALL_BOND_LENGTHS),
        "confirmation_restarts_per_geometry_variant": 5,
        "minimum_ablation_restarts_per_geometry_variant": 2,
        "all_orbital_mappings_reliable": True,
        "compact_labels": list(COMPACT_LABELS),
        "software_versions": diag.dependency_versions(),
    }
    diag.write_json(
        args.output_dir / "phase6_input_audit.json",
        input_audit,
    )
    if args.audit_only:
        print(json.dumps(input_audit, indent=2, sort_keys=True))
        return 0

    orbital_descriptors = make_orbital_descriptors(
        geometries,
        ALL_BOND_LENGTHS,
        args.output_dir,
    )
    _, amplitude_summary = extract_amplitudes(
        confirmation_runs,
        args.output_dir,
    )
    operator_table = make_operator_mechanism_table(
        systems,
        geometries,
        mappings,
        amplitude_summary,
        ALL_BOND_LENGTHS,
        args.finite_difference_step,
        args.output_dir,
    )
    conditional_channels = make_conditional_single_channels(
        systems,
        mappings,
        ablation_runs,
        ALL_BOND_LENGTHS,
        args.finite_difference_step,
        args.optimizer_tolerance,
        args.conditional_maxiter,
        args.output_dir,
    )

    loo_summary: pd.DataFrame | None = None
    loo_protocol: str | None = None
    if not args.skip_leave_one_out:
        loo_bond_lengths = resolved_loo_bond_lengths(
            args.profile,
            args.loo_bond_length,
        )
        loo_operators = resolved_loo_operators(args.loo_operator)
        loo_runs, loo_protocol = run_leave_one_out(
            systems,
            mappings,
            confirmation_runs,
            confirmation_conclusions["protocol_fingerprint"],
            loo_bond_lengths,
            loo_operators,
            args.output_dir,
            args.optimizer,
            args.maxiter,
            args.optimizer_tolerance,
            args.initial_scale,
            args.parameter_seed,
            args.loo_random_restarts,
            args.resume,
        )
        loo_summary = summarize_leave_one_out(
            loo_runs,
            loo_bond_lengths,
            loo_operators,
            args.loo_random_restarts,
            args.disagreement_tolerance,
            args.output_dir,
        )

    conclusions = make_conclusions(
        confirmation_conclusions,
        ablation_conclusions,
        orbital_descriptors,
        operator_table,
        conditional_channels,
        loo_summary,
        loo_protocol,
        args.output_dir,
    )
    write_report(
        conclusions,
        conditional_channels,
        loo_summary,
        args.output_dir,
    )
    print("\nPhase 6 conditional single-channel summary:")
    print(
        conditional_channels[
            [
                "bond_length_angstrom",
                "single_channel",
                "phase5_empirically_active_channel",
                "conditional_energy_lowering_ha",
                "conditional_best_error_exact_ha",
                "optimizer_success",
            ]
        ].to_string(index=False)
    )
    if loo_summary is not None:
        print("\nPhase 6 leave-one-out summary:")
        print(
            loo_summary[
                [
                    "bond_length_angstrom",
                    "omitted_operator_label",
                    "energy_penalty_vs_compact10_ha",
                    "chemical_accuracy_retained",
                    "strict_equivalence_retained",
                    "energy_stable",
                    "chemical_classification_stable",
                    "strict_classification_stable",
                ]
            ].to_string(index=False)
        )
    print("\nPhase 6 conclusions:")
    print(json.dumps(conclusions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
