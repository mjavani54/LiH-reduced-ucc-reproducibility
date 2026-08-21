#!/usr/bin/env python3
"""Phase 4: diagnose the stretched-bond failure of the LiH parent ansatz.

Phase 3 showed that a six-double support reproduces the optimized energy of the
full sixteen-double, one-repetition parent ansatz from 1.2 to 3.0 Angstrom. Both
supports nevertheless miss chemical accuracy at 2.5 and 3.0 Angstrom. This
program tests two candidate causes without changing the molecular Hamiltonian:

1. excitation content: doubles only versus singles plus doubles; and
2. repetition depth: one versus two UCC repetitions.

The resulting 2 x 2 factorial contains:

* doubles_r1: complete spin-preserving doubles, one repetition;
* doubles_r2: complete spin-preserving doubles, two repetitions;
* uccsd_r1:   complete spin-preserving singles and doubles, one repetition;
* uccsd_r2:   complete spin-preserving singles and doubles, two repetitions.

Every Hamiltonian is independently audited. Every optimization is checkpointed.
Base restarts use zero, geometry-continuation when available, and deterministic
random initializations. Rescue restarts are triggered by optimizer failure,
nonfinite energy, variational violation, excessive restart disagreement, or
disagreement on chemical-accuracy classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import lih_phase2b_exhaustive_support as p2b
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place lih_phase4_ansatz_diagnosis.py, "
        "lih_phase2b_exhaustive_support.py, and "
        "lih_reference_failure_diagnostics.py in the same directory."
    ) from exc


REFERENCE_BOND_LENGTH = 1.595
EXPECTED_REFERENCE_EXACT_TOTAL_HA = -7.8821745057672805
EXPECTED_ACTIVE_SPATIAL_ORBITALS = 5
EXPECTED_ACTIVE_ELECTRONS = 2
EXPECTED_SINGLE_COUNT = 8
EXPECTED_DOUBLE_COUNT = 16

PILOT_BOND_LENGTHS: tuple[float, ...] = (2.5,)
FULL_BOND_LENGTHS: tuple[float, ...] = (1.595, 2.1, 2.5, 3.0)


@dataclass(frozen=True)
class AnsatzVariant:
    name: str
    excitation_content: str
    reps: int
    scientific_role: str


VARIANTS: tuple[AnsatzVariant, ...] = (
    AnsatzVariant(
        "doubles_r1",
        "doubles",
        1,
        "Phase-3 parent control: complete spin-preserving doubles at one repetition.",
    ),
    AnsatzVariant(
        "doubles_r2",
        "doubles",
        2,
        "Tests whether additional UCC repetition depth repairs the parent failure.",
    ),
    AnsatzVariant(
        "uccsd_r1",
        "singles_and_doubles",
        1,
        "Tests whether adding spin-preserving singles repairs the parent failure.",
    ),
    AnsatzVariant(
        "uccsd_r2",
        "singles_and_doubles",
        2,
        "Tests the interaction between added singles and additional repetition depth.",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def geometry_key(bond_length: float) -> str:
    return f"{float(bond_length):.6f}"


def normalized_bond_lengths(
    profile: str,
    values: Sequence[float] | None,
) -> list[float]:
    defaults = PILOT_BOND_LENGTHS if profile == "pilot" else FULL_BOND_LENGTHS
    raw = list(defaults if not values else values)
    result = sorted({round(float(value), 9) for value in raw})
    if not result or any(value <= 0 for value in result):
        raise ValueError("At least one positive bond length is required.")
    keys = [geometry_key(value) for value in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Bond lengths must remain unique at six-decimal precision.")
    return result


def selected_variants(names: Sequence[str] | None) -> list[AnsatzVariant]:
    if not names:
        return list(VARIANTS)
    return [VARIANT_BY_NAME[name] for name in dict.fromkeys(names)]


def resolved_restart_count(profile: str, value: int | None) -> int:
    if value is not None:
        return int(value)
    return 2 if profile == "pilot" else 5


def build_config(bond_length: float, basis: str) -> diag.MolecularConfig:
    return diag.MolecularConfig(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        charge=0,
        spin=0,
        unit="ANGSTROM",
        freeze_core=True,
        mapper="JordanWignerMapper",
        excitation_rank=2,
        preserve_spin=True,
    )


def single_excitations(system: diag.SystemData) -> list[diag.Excitation]:
    from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
        generate_fermionic_excitations,
    )

    generated = generate_fermionic_excitations(
        num_excitations=1,
        num_spatial_orbitals=system.problem.num_spatial_orbitals,
        num_particles=system.problem.num_particles,
        preserve_spin=system.config.preserve_spin,
    )
    return [diag.canonical_excitation(item) for item in generated]


def excitation_pools(
    system: diag.SystemData,
) -> tuple[list[diag.Excitation], list[diag.Excitation]]:
    singles = single_excitations(system)
    doubles = diag.candidate_double_excitations(
        system.problem,
        preserve_spin=system.config.preserve_spin,
    )
    if system.problem.num_spatial_orbitals != EXPECTED_ACTIVE_SPATIAL_ORBITALS:
        raise RuntimeError("Phase 4 requires exactly five active spatial orbitals.")
    if sum(system.problem.num_particles) != EXPECTED_ACTIVE_ELECTRONS:
        raise RuntimeError("Phase 4 requires exactly two active electrons.")
    if len(singles) != EXPECTED_SINGLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SINGLE_COUNT} spin-preserving singles, "
            f"but generated {len(singles)}."
        )
    if len(doubles) != EXPECTED_DOUBLE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DOUBLE_COUNT} spin-preserving doubles, "
            f"but generated {len(doubles)}."
        )
    if len(set(singles)) != len(singles) or len(set(doubles)) != len(doubles):
        raise RuntimeError("The generated excitation pools contain duplicates.")
    if set(singles).intersection(doubles):
        raise RuntimeError("The generated single and double pools overlap.")
    return singles, doubles


def excitations_for_variant(
    variant: AnsatzVariant,
    singles: Sequence[diag.Excitation],
    doubles: Sequence[diag.Excitation],
) -> list[diag.Excitation]:
    if variant.excitation_content == "doubles":
        return list(doubles)
    if variant.excitation_content == "singles_and_doubles":
        return [*singles, *doubles]
    raise ValueError(f"Unknown excitation content: {variant.excitation_content}")


def build_ansatz(
    system: diag.SystemData,
    excitations: Sequence[diag.Excitation],
    reps: int,
) -> Any:
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
        preserve_spin=system.config.preserve_spin,
        reps=reps,
        initial_state=initial_state,
    )
    _ = ansatz.num_parameters
    expected_parameters = len(excitations) * reps
    if ansatz.num_parameters != expected_parameters:
        raise RuntimeError(
            f"Variant has {ansatz.num_parameters} parameters; "
            f"expected {expected_parameters}."
        )
    return ansatz


def pool_fingerprint(
    singles: Sequence[diag.Excitation],
    doubles: Sequence[diag.Excitation],
) -> str:
    payload = {
        "singles": [diag.excitation_to_json(item) for item in singles],
        "doubles": [diag.excitation_to_json(item) for item in doubles],
        "ordering": "singles_then_doubles_generator_order",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_operator_pool(
    singles: Sequence[diag.Excitation],
    doubles: Sequence[diag.Excitation],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for excitation_type, pool in (("single", singles), ("double", doubles)):
        prefix = "S" if excitation_type == "single" else "D"
        for index, excitation in enumerate(pool):
            rows.append(
                {
                    "operator_label": f"{prefix}{index}",
                    "excitation_type": excitation_type,
                    "type_local_index": index,
                    "excitation_json": json.dumps(
                        diag.excitation_to_json(excitation)
                    ),
                    "global_uccsd_order_index": (
                        index
                        if excitation_type == "single"
                        else len(singles) + index
                    ),
                    "included_in_doubles_variants": excitation_type == "double",
                    "included_in_uccsd_variants": True,
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "phase4_operator_pool.csv",
        index=False,
    )


def protocol_fingerprint(
    systems: dict[str, diag.SystemData],
    bond_lengths: Sequence[float],
    pool_hash: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
) -> str:
    payload = {
        "analysis_phase": "Phase 4 ansatz diagnosis",
        "bond_lengths_angstrom": list(bond_lengths),
        "configuration_fingerprints": {
            key: systems[key].fingerprint for key in sorted(systems)
        },
        "pool_fingerprint": pool_hash,
        "variants": [
            {
                "name": variant.name,
                "excitation_content": variant.excitation_content,
                "reps": variant.reps,
            }
            for variant in VARIANTS
        ],
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        "parameter_seed": parameter_seed,
        "base_restarts": base_restarts,
        "rescue_restarts": rescue_restarts,
        "disagreement_tolerance_ha": disagreement_tolerance_ha,
        "initialization_policy": (
            "zero_then_previous_geometry_continuation_when_available_then_random"
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def common_run_fields(
    system: diag.SystemData,
    geometry_index: int,
    bond_length: float,
    variant: AnsatzVariant,
    excitations: Sequence[diag.Excitation],
    protocol: str,
    pool_hash: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    resources: dict[str, int | None],
    num_parameters: int,
) -> dict[str, Any]:
    return {
        "protocol_fingerprint": protocol,
        "configuration_fingerprint": system.fingerprint,
        "operator_pool_fingerprint": pool_hash,
        "geometry_index": geometry_index,
        "bond_length_angstrom": bond_length,
        "geometry_key": geometry_key(bond_length),
        "variant": variant.name,
        "excitation_content": variant.excitation_content,
        "reps": variant.reps,
        "scientific_role": variant.scientific_role,
        "operator_ordering": "singles_then_doubles_generator_order",
        "num_unique_excitations": len(excitations),
        "num_parameters": num_parameters,
        "excitations_json": json.dumps(
            [diag.excitation_to_json(item) for item in excitations]
        ),
        "exact_total_energy_ha": system.exact_total_ha,
        "hf_total_energy_ha": system.hf_total_driver_ha,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        **resources,
    }


def best_previous_geometry_parameters(
    rows: Sequence[dict[str, Any]],
    protocol: str,
    variant_name: str,
    current_geometry_index: int,
    expected_parameters: int,
) -> tuple[np.ndarray, float] | None:
    if not rows or current_geometry_index <= 0:
        return None
    frame = pd.DataFrame(rows)
    required = {
        "protocol_fingerprint",
        "variant",
        "geometry_index",
        "vqe_total_energy_ha",
        "optimal_parameters_json",
    }
    if not required.issubset(frame.columns):
        return None
    candidates = frame[
        (frame["protocol_fingerprint"].astype(str) == protocol)
        & (frame["variant"].astype(str) == variant_name)
        & (
            pd.to_numeric(frame["geometry_index"], errors="coerce")
            == current_geometry_index - 1
        )
    ].copy()
    candidates["_energy"] = pd.to_numeric(
        candidates["vqe_total_energy_ha"],
        errors="coerce",
    )
    candidates = candidates[np.isfinite(candidates["_energy"])]
    if candidates.empty:
        return None
    best = candidates.sort_values("_energy").iloc[0]
    try:
        values = np.asarray(
            json.loads(str(best["optimal_parameters_json"])),
            dtype=float,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(values) != expected_parameters or not np.isfinite(values).all():
        return None
    return values, float(best["bond_length_angstrom"])


def run_experiment(
    systems: dict[str, diag.SystemData],
    pools: dict[str, tuple[list[diag.Excitation], list[diag.Excitation]]],
    bond_lengths: Sequence[float],
    variants_to_run: Sequence[AnsatzVariant],
    output_dir: Path,
    protocol: str,
    pool_hash: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
    resume: bool,
) -> pd.DataFrame:
    if base_restarts < 2:
        raise ValueError("At least two base restarts are required.")
    if rescue_restarts < 1:
        raise ValueError("At least one rescue restart is required.")
    if disagreement_tolerance_ha <= 0:
        raise ValueError("The disagreement tolerance must be positive.")

    runs_path = output_dir / "phase4_restart_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if resume and runs_path.exists():
        previous = pd.read_csv(runs_path)
        rows = previous.to_dict(orient="records")
        matching = previous[
            previous["protocol_fingerprint"].astype(str) == protocol
        ]
        completed = {
            (str(row["geometry_key"]), str(row["variant"]), int(row["restart"]))
            for row in matching.to_dict(orient="records")
        }
        print(f"Resuming with {len(completed)} completed Phase-4 attempts.")

    def current_runs(key: str, variant_name: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        return frame[
            (frame["protocol_fingerprint"].astype(str) == protocol)
            & (frame["geometry_key"].astype(str) == key)
            & (frame["variant"].astype(str) == variant_name)
        ].copy()

    for geometry_index, bond_length in enumerate(sorted(bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        singles, doubles = pools[key]
        for variant_index, variant in enumerate(variants_to_run):
            existing = current_runs(key, variant.name)
            if not existing.empty:
                existing_base = existing[
                    pd.to_numeric(existing["restart"], errors="coerce")
                    < base_restarts
                ].sort_values("restart")
                if len(existing_base) == base_restarts:
                    reasons = p2b.rescue_reasons(
                        existing_base,
                        base_restarts,
                        disagreement_tolerance_ha,
                    )
                    expected = base_restarts + (
                        rescue_restarts if reasons else 0
                    )
                    if all(
                        (key, variant.name, restart) in completed
                        for restart in range(expected)
                    ):
                        continue

            excitations = excitations_for_variant(
                variant,
                singles,
                doubles,
            )
            ansatz = build_ansatz(
                system,
                excitations,
                variant.reps,
            )
            expected_parameters = len(excitations) * variant.reps
            if ansatz.num_parameters != expected_parameters:
                raise RuntimeError(
                    f"{variant.name} parameter audit failed at R={bond_length}."
                )
            resource_seed = parameter_seed + 10_000 * geometry_index + variant_index
            resources = diag.circuit_resource_metrics(ansatz, resource_seed)
            common = common_run_fields(
                system,
                geometry_index,
                bond_length,
                variant,
                excitations,
                protocol,
                pool_hash,
                optimizer,
                maxiter,
                optimizer_tolerance,
                initial_scale,
                resources,
                int(ansatz.num_parameters),
            )

            def execute_restart(
                restart: int,
                stage: str,
                reasons: Sequence[str] | None,
            ) -> None:
                completed_key = (key, variant.name, restart)
                if completed_key in completed:
                    return
                restart_seed = (
                    parameter_seed
                    + 1_000_003 * geometry_index
                    + 100_003 * variant_index
                    + 10_007 * restart
                )
                rng = np.random.default_rng(restart_seed)
                continuation_source: float | None = None
                if restart == 0:
                    initial = np.zeros(ansatz.num_parameters, dtype=float)
                    initialization = "zeros"
                    initialization_seed: int | None = None
                elif restart == 1:
                    previous = best_previous_geometry_parameters(
                        rows,
                        protocol,
                        variant.name,
                        geometry_index,
                        int(ansatz.num_parameters),
                    )
                    if previous is None:
                        initial = rng.uniform(
                            -initial_scale,
                            initial_scale,
                            size=ansatz.num_parameters,
                        )
                        initialization = "uniform_random"
                        initialization_seed = restart_seed
                    else:
                        initial, continuation_source = previous
                        initialization = "previous_geometry_continuation"
                        initialization_seed = None
                else:
                    initial = rng.uniform(
                        -initial_scale,
                        initial_scale,
                        size=ansatz.num_parameters,
                    )
                    initialization = "uniform_random"
                    initialization_seed = restart_seed

                base = {
                    **common,
                    "restart": restart,
                    "restart_stage": stage,
                    "initialization": initialization,
                    "initialization_seed": initialization_seed,
                    "continuation_source_bond_length_angstrom": (
                        continuation_source
                    ),
                    "rescue_trigger_reasons": (
                        None if not reasons else json.dumps(list(reasons))
                    ),
                    "evaluation_mode": "statevector_vqe",
                }
                print(
                    f"R={bond_length:.3f} variant={variant.name} "
                    f"{stage}_restart={restart + 1} "
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
                        f"ERROR at R={bond_length}, {variant.name}, "
                        f"restart {restart}: {exc}",
                        file=sys.stderr,
                    )
                    row = p2b.failure_row(base, exc)
                rows.append(row)
                completed.add(completed_key)
                p2b.checkpoint(runs_path, rows)

            for restart in range(base_restarts):
                execute_restart(restart, "base", None)

            base_frame = current_runs(key, variant.name)
            base_frame = base_frame[
                pd.to_numeric(base_frame["restart"], errors="coerce")
                < base_restarts
            ].sort_values("restart")
            reasons = p2b.rescue_reasons(
                base_frame,
                base_restarts,
                disagreement_tolerance_ha,
            )
            if reasons:
                print(
                    f"R={bond_length:.3f} variant={variant.name} rescue: "
                    + ", ".join(reasons),
                    flush=True,
                )
                for restart in range(
                    base_restarts,
                    base_restarts + rescue_restarts,
                ):
                    execute_restart(restart, "rescue", reasons)

    all_runs = pd.DataFrame(rows)
    return all_runs[
        all_runs["protocol_fingerprint"].astype(str) == protocol
    ].copy()


def summarize_runs(
    runs: pd.DataFrame,
    systems: dict[str, diag.SystemData],
    bond_lengths: Sequence[float],
    variants_to_run: Sequence[AnsatzVariant],
    output_dir: Path,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for geometry_index, bond_length in enumerate(sorted(bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        for variant in variants_to_run:
            group = runs[
                (runs["geometry_key"].astype(str) == key)
                & (runs["variant"].astype(str) == variant.name)
            ].copy()
            group = group.sort_values("restart")
            if group.empty:
                raise RuntimeError(
                    f"No Phase-4 runs exist for R={bond_length}, {variant.name}."
                )
            rescue = group[group["restart_stage"] == "rescue"].copy()
            rescue_triggered = not rescue.empty
            if rescue_triggered:
                analysis_runs = rescue
                expected_analysis_runs = rescue_restarts
                analysis_scope = "rescue_only"
            else:
                analysis_runs = group[group["restart_stage"] == "base"].copy()
                expected_analysis_runs = base_restarts
                analysis_scope = "base"
            stable, warnings = p2b.assess_analysis_runs(
                analysis_runs,
                expected_analysis_runs,
                disagreement_tolerance_ha,
            )
            expected_attempts = base_restarts + (
                rescue_restarts if rescue_triggered else 0
            )
            attempts_complete = len(group) == expected_attempts
            valid = analysis_runs[
                p2b.finite_series(analysis_runs, "absolute_error_ha").notna()
            ].copy()
            errors = p2b.finite_series(valid, "absolute_error_ha")
            energies = p2b.finite_series(valid, "vqe_total_energy_ha")
            robust_chemical: bool | float = np.nan
            if stable and not errors.empty:
                robust_chemical = bool(
                    (errors <= diag.CHEMICAL_ACCURACY_HA).all()
                )
            if not attempts_complete:
                classification = "incomplete"
            elif not stable:
                classification = "unresolved_optimizer_behavior"
            elif bool(robust_chemical):
                classification = "stable_chemical_accuracy"
            else:
                classification = "stable_nonchemical"
            first = group.iloc[0]
            rescue_reason_values = (
                group["rescue_trigger_reasons"].dropna().astype(str)
            )
            rows.append(
                {
                    "geometry_index": geometry_index,
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "configuration_fingerprint": system.fingerprint,
                    "variant": variant.name,
                    "excitation_content": variant.excitation_content,
                    "reps": variant.reps,
                    "scientific_role": variant.scientific_role,
                    "num_unique_excitations": int(
                        first["num_unique_excitations"]
                    ),
                    "num_parameters": int(first["num_parameters"]),
                    "completed_attempts": int(len(group)),
                    "expected_attempts": expected_attempts,
                    "analysis_restart_scope": analysis_scope,
                    "analysis_restarts": int(len(analysis_runs)),
                    "rescue_triggered": rescue_triggered,
                    "rescue_trigger_reasons": (
                        rescue_reason_values.iloc[0]
                        if not rescue_reason_values.empty
                        else None
                    ),
                    "analysis_complete": attempts_complete,
                    "analysis_stable": stable,
                    "stability_warnings_json": json.dumps(warnings),
                    "classification": classification,
                    "exact_total_energy_ha": system.exact_total_ha,
                    "hf_total_energy_ha": system.hf_total_driver_ha,
                    "hf_error_ha": (
                        system.hf_total_driver_ha - system.exact_total_ha
                    ),
                    "best_total_energy_ha": (
                        float(energies.min()) if not energies.empty else np.nan
                    ),
                    "mean_total_energy_ha": (
                        float(energies.mean()) if not energies.empty else np.nan
                    ),
                    "best_error_exact_ha": (
                        float(errors.min()) if not errors.empty else np.nan
                    ),
                    "median_error_exact_ha": (
                        float(errors.median()) if not errors.empty else np.nan
                    ),
                    "mean_error_exact_ha": (
                        float(errors.mean()) if not errors.empty else np.nan
                    ),
                    "std_error_exact_ha": (
                        float(errors.std(ddof=1)) if len(errors) > 1 else 0.0
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
                        float(
                            (
                                errors <= diag.CHEMICAL_ACCURACY_HA
                            ).mean()
                        )
                        if not errors.empty
                        else np.nan
                    ),
                    "robust_chemical_accuracy": robust_chemical,
                    "optimizer_success_rate": (
                        float(
                            analysis_runs["optimizer_success"]
                            .map(p2b.as_bool)
                            .mean()
                        )
                        if not analysis_runs.empty
                        else np.nan
                    ),
                    "median_iterations": (
                        float(
                            pd.to_numeric(
                                valid["num_iterations"],
                                errors="coerce",
                            ).median()
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "median_elapsed_seconds": (
                        float(
                            pd.to_numeric(
                                valid["elapsed_seconds"],
                                errors="coerce",
                            ).median()
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "total_elapsed_seconds_all_attempts": float(
                        pd.to_numeric(
                            group["elapsed_seconds"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .sum()
                    ),
                    "logical_depth": first.get("logical_depth"),
                    "compiled_depth": first.get("compiled_depth"),
                    "compiled_cx": first.get("compiled_cx"),
                    "any_variational_violation": bool(
                        group["variational_violation"]
                        .map(p2b.as_bool)
                        .any()
                    ),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["geometry_index", "variant"]
    )
    summary = add_baseline_comparisons(summary)
    summary.to_csv(
        output_dir / "phase4_variant_geometry_summary.csv",
        index=False,
    )
    return summary


def add_baseline_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    for column in (
        "doubles_r1_mean_error_exact_ha",
        "error_reduction_vs_doubles_r1_ha",
        "compiled_depth_change_vs_doubles_r1",
        "compiled_cx_change_vs_doubles_r1",
        "num_parameters_change_vs_doubles_r1",
    ):
        result[column] = np.nan

    for _, indices in result.groupby("geometry_key").groups.items():
        geometry = result.loc[indices]
        baseline_rows = geometry[geometry["variant"] == "doubles_r1"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        if not p2b.as_bool(baseline["analysis_stable"]):
            continue
        baseline_error = float(baseline["mean_error_exact_ha"])
        for index in indices:
            row = result.loc[index]
            result.at[index, "doubles_r1_mean_error_exact_ha"] = baseline_error
            if p2b.as_bool(row["analysis_stable"]):
                result.at[
                    index,
                    "error_reduction_vs_doubles_r1_ha",
                ] = baseline_error - float(row["mean_error_exact_ha"])
            for resource in (
                "compiled_depth",
                "compiled_cx",
                "num_parameters",
            ):
                denominator = baseline.get(resource)
                value = row.get(resource)
                output_column = f"{resource}_change_vs_doubles_r1"
                if (
                    pd.notna(denominator)
                    and float(denominator) != 0
                    and pd.notna(value)
                ):
                    result.at[index, output_column] = (
                        float(value) - float(denominator)
                    ) / float(denominator)
    return result


def causal_diagnosis(group: pd.DataFrame) -> str:
    stable = {
        str(row["variant"]): row
        for _, row in group.iterrows()
        if p2b.as_bool(row["analysis_stable"])
    }
    if set(stable) != set(VARIANT_BY_NAME):
        return "incomplete_factorial"
    chemical = {
        name: p2b.as_bool(row["robust_chemical_accuracy"])
        for name, row in stable.items()
    }
    if chemical["doubles_r1"]:
        return "baseline_control_chemical"
    depth = chemical["doubles_r2"]
    singles = chemical["uccsd_r1"]
    combined = chemical["uccsd_r2"]
    if depth and singles:
        return "either_depth_or_singles_repairs"
    if depth and not singles:
        return "repetition_depth_repairs"
    if singles and not depth:
        return "singles_repair"
    if combined:
        return "combined_singles_and_depth_required"
    return "tested_hierarchy_insufficient"


def make_geometry_summary(
    variant_summary: pd.DataFrame,
    systems: dict[str, diag.SystemData],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in variant_summary.groupby("geometry_key", sort=False):
        bond_length = float(group["bond_length_angstrom"].iloc[0])
        system = systems[str(key)]
        stable = group[group["analysis_stable"].map(p2b.as_bool)]
        best = (
            stable.sort_values("mean_error_exact_ha").iloc[0]
            if not stable.empty
            else None
        )
        row: dict[str, Any] = {
            "geometry_index": int(group["geometry_index"].iloc[0]),
            "bond_length_angstrom": bond_length,
            "configuration_fingerprint": system.fingerprint,
            "exact_total_energy_ha": system.exact_total_ha,
            "hf_total_energy_ha": system.hf_total_driver_ha,
            "hf_error_ha": system.hf_total_driver_ha - system.exact_total_ha,
            "factorial_diagnosis": causal_diagnosis(group),
            "best_variant": None if best is None else best["variant"],
            "best_mean_error_exact_ha": (
                np.nan if best is None else float(best["mean_error_exact_ha"])
            ),
        }
        for variant in VARIANTS:
            selected = group[group["variant"] == variant.name]
            row[f"{variant.name}_mean_error_exact_ha"] = (
                np.nan
                if selected.empty
                else selected.iloc[0].get("mean_error_exact_ha", np.nan)
            )
            row[f"{variant.name}_robust_chemical_accuracy"] = (
                pd.NA
                if selected.empty
                else selected.iloc[0].get("robust_chemical_accuracy", pd.NA)
            )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("geometry_index")
    result.to_csv(
        output_dir / "phase4_geometry_summary.csv",
        index=False,
    )
    return result


def nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def make_conclusions(
    profile: str,
    protocol: str,
    pool_hash: str,
    variant_summary: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    variants_to_run: Sequence[AnsatzVariant],
    output_dir: Path,
) -> dict[str, Any]:
    requested_names = {variant.name for variant in variants_to_run}
    all_stable = bool(
        variant_summary["analysis_complete"].map(p2b.as_bool).all()
        and variant_summary["analysis_stable"].map(p2b.as_bool).all()
    )
    factorial_requested = requested_names == set(VARIANT_BY_NAME)
    full_grid = set(
        geometry_summary["bond_length_angstrom"].astype(float).tolist()
    ) == set(FULL_BOND_LENGTHS)
    if all_stable and factorial_requested and full_grid:
        analysis_status = "complete"
    elif all_stable and factorial_requested and profile == "pilot":
        analysis_status = "pilot_complete"
    elif all_stable:
        analysis_status = "partial_factorial"
    else:
        analysis_status = "partial_or_unresolved"

    variant_results: dict[str, Any] = {}
    for variant in VARIANTS:
        group = variant_summary[
            variant_summary["variant"] == variant.name
        ].sort_values("geometry_index")
        stable = group[group["analysis_stable"].map(p2b.as_bool)]
        chemical = stable[
            stable["robust_chemical_accuracy"].map(p2b.as_bool)
        ]
        variant_results[variant.name] = {
            "excitation_content": variant.excitation_content,
            "reps": variant.reps,
            "num_geometries_requested": int(len(group)),
            "num_geometries_stable": int(len(stable)),
            "num_geometries_chemical_accuracy": int(len(chemical)),
            "chemical_accuracy_bond_lengths_angstrom": (
                chemical["bond_length_angstrom"].astype(float).tolist()
            ),
            "worst_mean_error_exact_ha": nullable_float(
                pd.to_numeric(
                    stable["mean_error_exact_ha"],
                    errors="coerce",
                ).max()
            ),
            "median_compiled_depth": nullable_float(
                pd.to_numeric(
                    stable["compiled_depth"],
                    errors="coerce",
                ).median()
            ),
            "median_compiled_cx": nullable_float(
                pd.to_numeric(
                    stable["compiled_cx"],
                    errors="coerce",
                ).median()
            ),
            "num_parameters": (
                None
                if stable.empty
                else int(stable["num_parameters"].iloc[0])
            ),
        }

    diagnoses = {
        geometry_key(row["bond_length_angstrom"]): str(
            row["factorial_diagnosis"]
        )
        for _, row in geometry_summary.iterrows()
    }
    conclusions = {
        "analysis_status": analysis_status,
        "profile": profile,
        "protocol_fingerprint": protocol,
        "operator_pool_fingerprint": pool_hash,
        "bond_lengths_angstrom": (
            geometry_summary["bond_length_angstrom"].astype(float).tolist()
        ),
        "variants_requested": [
            variant.name for variant in variants_to_run
        ],
        "num_geometry_variant_combinations_expected": (
            len(geometry_summary) * len(variants_to_run)
        ),
        "num_geometry_variant_combinations_stable": int(
            variant_summary["analysis_stable"].map(p2b.as_bool).sum()
        ),
        "num_combinations_requiring_rescue": int(
            variant_summary["rescue_triggered"].map(p2b.as_bool).sum()
        ),
        "any_variational_violation": bool(
            variant_summary["any_variational_violation"]
            .map(p2b.as_bool)
            .any()
        ),
        "causal_diagnosis_by_geometry": diagnoses,
        "variant_results": variant_results,
        "thresholds_ha": {
            "chemical": diag.CHEMICAL_ACCURACY_HA,
            "good": diag.GOOD_ACCURACY_HA,
            "acceptable": diag.ACCEPTABLE_ACCURACY_HA,
        },
        "interpretation_rule": {
            "baseline_control_chemical": (
                "The one-repetition doubles parent already reaches chemical "
                "accuracy; this geometry is a positive control."
            ),
            "repetition_depth_repairs": (
                "Doubles at two repetitions reaches chemical accuracy while "
                "one-repetition UCCSD does not."
            ),
            "singles_repair": (
                "One-repetition UCCSD reaches chemical accuracy while "
                "two-repetition doubles does not."
            ),
            "either_depth_or_singles_repairs": (
                "Both two-repetition doubles and one-repetition UCCSD repair "
                "the baseline failure."
            ),
            "combined_singles_and_depth_required": (
                "Only two-repetition UCCSD repairs the baseline failure."
            ),
            "tested_hierarchy_insufficient": (
                "None of the tested ansatz variants reaches chemical accuracy."
            ),
            "incomplete_factorial": (
                "One or more treatments are absent or unresolved, so the "
                "causal comparison is incomplete."
            ),
        },
        "claim_boundary": (
            "This factorial diagnoses the tested frozen-core LiH/STO-3G "
            "Hamiltonians with canonical RHF orbitals, Jordan-Wigner mapping, "
            "statevector energies, product-form UCC circuits, and the declared "
            "local-optimization protocol. Agreement across restarts is strong "
            "numerical evidence but is not a mathematical proof of global "
            "optimality or ansatz representability. The experiment does not "
            "establish transfer to other basis sets, active spaces, molecules, "
            "hardware noise models, or ansatz families."
        ),
    }
    diag.write_json(
        output_dir / "phase4_ansatz_conclusions.json",
        conclusions,
    )
    return conclusions


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    variant_summary: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    conclusions: dict[str, Any],
    output_dir: Path,
) -> None:
    lines = [
        "# LiH Phase 4 Ansatz-Diagnosis Report",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        (
            "Stable geometry-variant combinations: "
            f"**{conclusions['num_geometry_variant_combinations_stable']}/"
            f"{conclusions['num_geometry_variant_combinations_expected']}**"
        ),
        "",
        (
            "Rescue-triggered combinations: "
            f"**{conclusions['num_combinations_requiring_rescue']}**"
        ),
        "",
        "## Geometry-level causal diagnosis",
        "",
        (
            "| R (A) | HF error | D-r1 error | D-r2 error | "
            "UCCSD-r1 error | UCCSD-r2 error | Diagnosis |"
        ),
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in geometry_summary.iterrows():
        lines.append(
            f"| {fmt(row['bond_length_angstrom'], 3)} | "
            f"{fmt(row['hf_error_ha'])} | "
            f"{fmt(row.get('doubles_r1_mean_error_exact_ha'))} | "
            f"{fmt(row.get('doubles_r2_mean_error_exact_ha'))} | "
            f"{fmt(row.get('uccsd_r1_mean_error_exact_ha'))} | "
            f"{fmt(row.get('uccsd_r2_mean_error_exact_ha'))} | "
            f"{row['factorial_diagnosis']} |"
        )

    lines.extend(
        [
            "",
            "## Variant-level results",
            "",
            (
                "| R (A) | Variant | Parameters | Error vs exact | "
                "Error reduction vs D-r1 | Chemical | Depth | CNOT |"
            ),
            "|---:|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for _, row in variant_summary.iterrows():
        lines.append(
            f"| {fmt(row['bond_length_angstrom'], 3)} | {row['variant']} | "
            f"{int(row['num_parameters'])} | "
            f"{fmt(row['mean_error_exact_ha'])} | "
            f"{fmt(row.get('error_reduction_vs_doubles_r1_ha'))} | "
            f"{row['robust_chemical_accuracy']} | "
            f"{fmt(row.get('compiled_depth'), 0)} | "
            f"{fmt(row.get('compiled_cx'), 0)} |"
        )

    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            conclusions["claim_boundary"],
            "",
        ]
    )
    (output_dir / "phase4_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the stretched-bond failure of the audited LiH parent "
            "ansatz using an excitation-content by repetition-depth factorial."
        )
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="pilot",
        help=(
            "pilot: R=2.5 with two base restarts by default; "
            "full: R=1.595, 2.1, 2.5, 3.0 with five base restarts by default."
        ),
    )
    parser.add_argument(
        "--bond-length",
        action="append",
        type=float,
        help="Override the profile grid; repeat for multiple bond lengths.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANT_BY_NAME),
        help="Run only this treatment; repeat for multiple treatments.",
    )
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument(
        "--base-restarts",
        type=int,
        default=None,
        help="Default: 2 for pilot, 5 for full.",
    )
    parser.add_argument("--rescue-restarts", type=int, default=3)
    parser.add_argument("--disagreement-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--optimizer",
        choices=["SLSQP", "COBYLA", "L-BFGS-B"],
        default="SLSQP",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-9)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--parameter-seed", type=int, default=20260723)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build and audit Hamiltonians and operator pools, then stop.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: lih_phase4_pilot for the pilot profile "
            "and lih_phase4_full for the full profile."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume matching checkpointed runs (default: true).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_restarts = resolved_restart_count(
        args.profile,
        args.base_restarts,
    )
    if base_restarts < 2:
        raise ValueError("At least two base restarts are required.")
    if args.rescue_restarts < 1:
        raise ValueError("At least one rescue restart is required.")
    if args.maxiter < 1:
        raise ValueError("maxiter must be positive.")
    if args.disagreement_tolerance <= 0:
        raise ValueError("The disagreement tolerance must be positive.")
    if args.optimizer_tolerance <= 0:
        raise ValueError("The optimizer tolerance must be positive.")
    if args.initial_scale <= 0:
        raise ValueError("The initial scale must be positive.")

    bond_lengths = normalized_bond_lengths(
        args.profile,
        args.bond_length,
    )
    variants_to_run = selected_variants(args.variant)
    output_dir = args.output_dir or Path(
        "lih_phase4_pilot"
        if args.profile == "pilot"
        else "lih_phase4_full"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    systems: dict[str, diag.SystemData] = {}
    pools: dict[
        str,
        tuple[list[diag.Excitation], list[diag.Excitation]],
    ] = {}
    audits: list[dict[str, Any]] = []
    print("Building and auditing all Phase-4 Hamiltonians...", flush=True)
    for bond_length in bond_lengths:
        key = geometry_key(bond_length)
        system = diag.build_system(build_config(bond_length, args.basis))
        singles, doubles = excitation_pools(system)
        reported = (
            [EXPECTED_REFERENCE_EXACT_TOTAL_HA]
            if abs(bond_length - REFERENCE_BOND_LENGTH) <= 1e-9
            else []
        )
        audit = diag.validate_reference(
            system,
            reported,
            args.audit_tolerance,
        )
        if not audit["audit_passed"]:
            raise RuntimeError(
                f"The Hamiltonian audit failed at R={bond_length}."
            )
        if reported and (
            abs(system.exact_total_ha - reported[0]) > args.audit_tolerance
        ):
            raise RuntimeError(
                "The equilibrium exact energy does not match Phases 1-3."
            )
        systems[key] = system
        pools[key] = (singles, doubles)
        audits.append(audit)

    reference_key = geometry_key(bond_lengths[0])
    reference_singles, reference_doubles = pools[reference_key]
    for key, (singles, doubles) in pools.items():
        if singles != reference_singles or doubles != reference_doubles:
            raise RuntimeError(
                f"The generated operator pool changed at geometry key {key}."
            )
    pool_hash = pool_fingerprint(
        reference_singles,
        reference_doubles,
    )
    write_operator_pool(
        reference_singles,
        reference_doubles,
        output_dir,
    )
    audit_bundle = {
        "analysis_phase": "Phase 4 ansatz diagnosis",
        "profile": args.profile,
        "bond_lengths_angstrom": bond_lengths,
        "num_spin_preserving_singles": len(reference_singles),
        "num_spin_preserving_doubles": len(reference_doubles),
        "operator_ordering": "singles_then_doubles_generator_order",
        "operator_pool_fingerprint": pool_hash,
        "geometry_audits": audits,
    }
    diag.write_json(
        output_dir / "phase4_geometry_audits.json",
        audit_bundle,
    )

    if args.audit_only:
        print("\nPhase 4 audit summary:")
        print(
            json.dumps(
                diag.json_ready(
                    {
                        "bond_lengths_angstrom": bond_lengths,
                        "num_spin_preserving_singles": len(reference_singles),
                        "num_spin_preserving_doubles": len(reference_doubles),
                        "operator_pool_fingerprint": pool_hash,
                        "all_hamiltonian_audits_passed": all(
                            bool(audit["audit_passed"])
                            for audit in audits
                        ),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        print(f"\nAudit-only outputs saved in: {output_dir.resolve()}")
        return 0

    protocol = protocol_fingerprint(
        systems,
        bond_lengths,
        pool_hash,
        args.optimizer,
        args.maxiter,
        args.optimizer_tolerance,
        args.initial_scale,
        args.parameter_seed,
        base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
    )
    runs = run_experiment(
        systems,
        pools,
        bond_lengths,
        variants_to_run,
        output_dir,
        protocol,
        pool_hash,
        args.optimizer,
        args.maxiter,
        args.optimizer_tolerance,
        args.initial_scale,
        args.parameter_seed,
        base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
        args.resume,
    )
    variant_summary = summarize_runs(
        runs,
        systems,
        bond_lengths,
        variants_to_run,
        output_dir,
        base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
    )
    geometry_summary = make_geometry_summary(
        variant_summary,
        systems,
        output_dir,
    )
    conclusions = make_conclusions(
        args.profile,
        protocol,
        pool_hash,
        variant_summary,
        geometry_summary,
        variants_to_run,
        output_dir,
    )
    write_report(
        variant_summary,
        geometry_summary,
        conclusions,
        output_dir,
    )

    print("\nPhase 4 geometry summary:")
    display_columns = [
        "bond_length_angstrom",
        "hf_error_ha",
        "doubles_r1_mean_error_exact_ha",
        "doubles_r2_mean_error_exact_ha",
        "uccsd_r1_mean_error_exact_ha",
        "uccsd_r2_mean_error_exact_ha",
        "best_variant",
        "best_mean_error_exact_ha",
        "factorial_diagnosis",
    ]
    print(
        geometry_summary.reindex(columns=display_columns).to_string(
            index=False
        )
    )
    print("\nPhase 4 conclusions:")
    print(
        json.dumps(
            diag.json_ready(conclusions),
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nSaved results in: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed Phase-4 attempts remain checkpointed.",
            file=sys.stderr,
        )
