#!/usr/bin/env python3
"""Phase 2B: exhaustive closure of the six-operator LiH support.

Phase 2A established single-deletion effects inside the empirically discovered
support ``{0, 3, 5, 10, 12, 15}``. Those ablations do not prove that an omitted
operator cannot be compensated by a different subset. This program closes that
logical gap by optimizing all 2**6 = 64 subsets of the discovered support.

The empty subset is evaluated directly as the Hartree--Fock reference. Every
non-empty subset receives two base restarts by default: one zero initialization
and one seeded random initialization. Three additional rescue restarts are run
only when the base runs fail, disagree by more than the declared energy
tolerance, violate the variational bound, or disagree on chemical-accuracy
classification.

All raw attempts are checkpointed. A protocol fingerprint prevents results
from different Hamiltonians or optimization settings from being combined.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place lih_phase2b_exhaustive_support.py in the same directory as "
        "lih_reference_failure_diagnostics.py, then run it again."
    ) from exc


ACTIVE_IDS: tuple[int, ...] = (0, 3, 5, 10, 12, 15)
EXPECTED_EXACT_TOTAL_HA = -7.8821745057672805
OPERATOR_ORDERING = "ascending_candidate_id"


@dataclass(frozen=True)
class SupportSubset:
    mask: int
    selected_ids: tuple[int, ...]

    @property
    def name(self) -> str:
        suffix = "empty" if not self.selected_ids else "_".join(map(str, self.selected_ids))
        return f"m{self.mask:02d}_{suffix}"


def all_support_subsets() -> list[SupportSubset]:
    """Enumerate masks 0--63 using ACTIVE_IDS as the bit-position map."""
    subsets = []
    for mask in range(1 << len(ACTIVE_IDS)):
        selected = tuple(
            candidate_id
            for bit, candidate_id in enumerate(ACTIVE_IDS)
            if mask & (1 << bit)
        )
        subsets.append(SupportSubset(mask=mask, selected_ids=selected))
    return subsets


ALL_SUBSETS: tuple[SupportSubset, ...] = tuple(all_support_subsets())
SUBSET_BY_MASK = {subset.mask: subset for subset in ALL_SUBSETS}


def selected_subsets(masks: Sequence[int] | None) -> list[SupportSubset]:
    if not masks:
        return list(ALL_SUBSETS)
    return [SUBSET_BY_MASK[mask] for mask in dict.fromkeys(masks)]


def expected_candidate_pool() -> list[diag.Excitation]:
    """Expected Qiskit ordering for the audited 2e/5o opposite-spin doubles."""
    return [
        ((0, 5), (alpha_virtual, beta_virtual))
        for alpha_virtual in range(1, 5)
        for beta_virtual in range(6, 10)
    ]


def validate_candidate_pool(system: diag.SystemData) -> None:
    expected = expected_candidate_pool()
    actual = list(system.candidate_excitations)
    if actual != expected:
        raise RuntimeError(
            "The generated excitation pool or ordering differs from the audited "
            "Phase-1 pool. Candidate IDs cannot be interpreted safely."
        )


def protocol_fingerprint(
    system: diag.SystemData,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
) -> str:
    # The requested subset-mask selection is intentionally excluded. A partial
    # run can therefore be resumed later as part of the complete 64-subset run.
    payload = {
        "configuration_fingerprint": system.fingerprint,
        "active_ids": list(ACTIVE_IDS),
        "operator_ordering": OPERATOR_ORDERING,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        "parameter_seed": parameter_seed,
        "base_restarts": base_restarts,
        "rescue_restarts": rescue_restarts,
        "disagreement_tolerance_ha": disagreement_tolerance_ha,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def rescue_reasons(
    base_runs: pd.DataFrame,
    base_restarts: int,
    disagreement_tolerance_ha: float,
) -> list[str]:
    reasons: list[str] = []
    if len(base_runs) != base_restarts:
        reasons.append("incomplete_base_restarts")
        return reasons

    energies = finite_series(base_runs, "vqe_total_energy_ha")
    if energies.isna().any():
        reasons.append("nonfinite_base_energy")
    if not base_runs["optimizer_success"].map(as_bool).all():
        reasons.append("optimizer_failure")
    if base_runs["variational_violation"].map(as_bool).any():
        reasons.append("variational_violation")
    if energies.notna().all() and float(energies.max() - energies.min()) > disagreement_tolerance_ha:
        reasons.append("energy_disagreement")
    errors = finite_series(base_runs, "absolute_error_ha")
    if errors.notna().all():
        classes = errors <= diag.CHEMICAL_ACCURACY_HA
        if classes.nunique() > 1:
            reasons.append("chemical_classification_disagreement")
    return reasons


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


def common_run_fields(
    system: diag.SystemData,
    subset: SupportSubset,
    fingerprint: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    resources: dict[str, int | None],
    num_parameters: int,
) -> dict[str, Any]:
    excitations = [system.candidate_excitations[i] for i in subset.selected_ids]
    return {
        "protocol_fingerprint": fingerprint,
        "configuration_fingerprint": system.fingerprint,
        "subset_mask": subset.mask,
        "subset_name": subset.name,
        "selected_count": len(subset.selected_ids),
        "selected_ids_json": json.dumps(list(subset.selected_ids)),
        "selected_excitations_json": json.dumps(
            [diag.excitation_to_json(excitation) for excitation in excitations]
        ),
        "operator_ordering": OPERATOR_ORDERING,
        "num_parameters": num_parameters,
        "exact_total_energy_ha": system.exact_total_ha,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        **resources,
    }


def hartree_fock_row(
    system: diag.SystemData,
    subset: SupportSubset,
    fingerprint: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
) -> dict[str, Any]:
    from qiskit_nature.second_q.circuit.library import HartreeFock

    circuit = HartreeFock(
        system.problem.num_spatial_orbitals,
        system.problem.num_particles,
        system.mapper,
    )
    resources = diag.circuit_resource_metrics(circuit, parameter_seed)
    base = common_run_fields(
        system,
        subset,
        fingerprint,
        optimizer,
        maxiter,
        optimizer_tolerance,
        initial_scale,
        resources,
        num_parameters=0,
    )
    energy = float(system.hf_total_from_qubit_ha)
    signed_error = energy - system.exact_total_ha
    return {
        **base,
        "restart": 0,
        "restart_stage": "direct_hf",
        "initialization": "not_applicable",
        "initialization_seed": None,
        "rescue_trigger_reasons": None,
        "evaluation_mode": "direct_hartree_fock",
        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_message": "Direct Hartree--Fock reference; no optimization.",
        "num_objective_evaluations": 0,
        "num_iterations": 0,
        "elapsed_seconds": 0.0,
        "vqe_total_energy_ha": energy,
        "signed_error_ha": signed_error,
        "absolute_error_ha": abs(signed_error),
        "variational_violation": bool(signed_error < -1e-8),
        "chemical_accuracy": bool(abs(signed_error) <= diag.CHEMICAL_ACCURACY_HA),
        "good_accuracy": bool(abs(signed_error) <= diag.GOOD_ACCURACY_HA),
        "acceptable_accuracy": bool(abs(signed_error) <= diag.ACCEPTABLE_ACCURACY_HA),
        "optimal_parameters_json": "[]",
        "run_exception": None,
    }


def run_experiment(
    system: diag.SystemData,
    subsets_to_run: Sequence[SupportSubset],
    output_dir: Path,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
    resume: bool,
) -> tuple[pd.DataFrame, str]:
    if base_restarts < 2:
        raise ValueError("At least two base restarts are required.")
    if rescue_restarts < 1:
        raise ValueError("At least one rescue restart is required.")
    if disagreement_tolerance_ha <= 0:
        raise ValueError("The disagreement tolerance must be positive.")

    fingerprint = protocol_fingerprint(
        system,
        optimizer,
        maxiter,
        optimizer_tolerance,
        initial_scale,
        parameter_seed,
        base_restarts,
        rescue_restarts,
        disagreement_tolerance_ha,
    )
    runs_path = output_dir / "phase2b_exhaustive_restart_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()

    if resume and runs_path.exists():
        previous = pd.read_csv(runs_path)
        rows = previous.to_dict(orient="records")
        if "protocol_fingerprint" in previous.columns:
            matching = previous[previous["protocol_fingerprint"].astype(str) == fingerprint]
            completed = {
                (int(row["subset_mask"]), int(row["restart"]))
                for row in matching.to_dict(orient="records")
            }
        print(f"Resuming with {len(completed)} completed Phase-2B evaluations.")

    def current_subset_runs(mask: int) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        return frame[
            (frame["protocol_fingerprint"].astype(str) == fingerprint)
            & (pd.to_numeric(frame["subset_mask"], errors="coerce") == mask)
        ].copy()

    for subset in subsets_to_run:
        if not subset.selected_ids:
            key = (subset.mask, 0)
            if key not in completed:
                print("subset=0/63 IDs=[] direct Hartree--Fock evaluation", flush=True)
                rows.append(
                    hartree_fock_row(
                        system,
                        subset,
                        fingerprint,
                        optimizer,
                        maxiter,
                        optimizer_tolerance,
                        initial_scale,
                        parameter_seed,
                    )
                )
                completed.add(key)
                checkpoint(runs_path, rows)
            continue

        existing = current_subset_runs(subset.mask)
        if not existing.empty:
            existing_base = existing[
                pd.to_numeric(existing["restart"], errors="coerce") < base_restarts
            ].sort_values("restart")
            if len(existing_base) == base_restarts:
                existing_reasons = rescue_reasons(
                    existing_base,
                    base_restarts,
                    disagreement_tolerance_ha,
                )
                expected_restarts = base_restarts + (
                    rescue_restarts if existing_reasons else 0
                )
                if all(
                    (subset.mask, restart) in completed
                    for restart in range(expected_restarts)
                ):
                    continue

        excitations = [system.candidate_excitations[i] for i in subset.selected_ids]
        ansatz = diag.build_selected_ansatz(system, excitations)
        resources = diag.circuit_resource_metrics(ansatz, parameter_seed + subset.mask)
        common = common_run_fields(
            system,
            subset,
            fingerprint,
            optimizer,
            maxiter,
            optimizer_tolerance,
            initial_scale,
            resources,
            num_parameters=int(ansatz.num_parameters),
        )

        def execute_restart(restart: int, stage: str, reasons: Sequence[str] | None) -> None:
            key = (subset.mask, restart)
            if key in completed:
                return
            restart_seed = parameter_seed + 1_000_003 * subset.mask + 10_007 * restart
            rng = np.random.default_rng(restart_seed)
            initial = diag.initial_point_for_restart(
                restart,
                ansatz.num_parameters,
                rng,
                initial_scale,
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
                "evaluation_mode": "statevector_vqe",
            }
            print(
                f"subset={subset.mask}/63 IDs={list(subset.selected_ids)} "
                f"{stage}_restart={restart + 1} parameters={ansatz.num_parameters}",
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
                    f"ERROR in subset {subset.mask}, restart {restart}: {exc}",
                    file=sys.stderr,
                )
                row = failure_row(base, exc)
            rows.append(row)
            completed.add(key)
            checkpoint(runs_path, rows)

        for restart in range(base_restarts):
            execute_restart(restart, "base", None)

        base_frame = current_subset_runs(subset.mask)
        base_frame = base_frame[
            pd.to_numeric(base_frame["restart"], errors="coerce") < base_restarts
        ].sort_values("restart")
        reasons = rescue_reasons(
            base_frame,
            base_restarts,
            disagreement_tolerance_ha,
        )
        if reasons:
            print(
                f"subset={subset.mask}/63 rescue triggered: {', '.join(reasons)}",
                flush=True,
            )
            for restart in range(base_restarts, base_restarts + rescue_restarts):
                execute_restart(restart, "rescue", reasons)

    all_runs = pd.DataFrame(rows)
    current = all_runs[
        all_runs["protocol_fingerprint"].astype(str) == fingerprint
    ].copy()
    return current, fingerprint


def assess_analysis_runs(
    analysis_runs: pd.DataFrame,
    expected_runs: int,
    disagreement_tolerance_ha: float,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if len(analysis_runs) != expected_runs:
        warnings.append("incomplete_analysis_restarts")
        return False, warnings
    energies = finite_series(analysis_runs, "vqe_total_energy_ha")
    if energies.isna().any():
        warnings.append("nonfinite_energy")
    if not analysis_runs["optimizer_success"].map(as_bool).all():
        warnings.append("optimizer_failure")
    if analysis_runs["variational_violation"].map(as_bool).any():
        warnings.append("variational_violation")
    if energies.notna().all() and float(energies.max() - energies.min()) > disagreement_tolerance_ha:
        warnings.append("energy_disagreement")
    errors = finite_series(analysis_runs, "absolute_error_ha")
    if errors.notna().all() and (errors <= diag.CHEMICAL_ACCURACY_HA).nunique() > 1:
        warnings.append("chemical_classification_disagreement")
    return not warnings, warnings


def summarize_subsets(
    runs: pd.DataFrame,
    output_dir: Path,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for subset in ALL_SUBSETS:
        group = runs[
            pd.to_numeric(runs["subset_mask"], errors="coerce") == subset.mask
        ].copy()
        group = group.sort_values("restart") if not group.empty else group
        if group.empty:
            summary_rows.append(
                {
                    "subset_mask": subset.mask,
                    "subset_name": subset.name,
                    "selected_ids_json": json.dumps(list(subset.selected_ids)),
                    "selected_count": len(subset.selected_ids),
                    "analysis_complete": False,
                    "analysis_stable": False,
                    "classification": "not_run",
                    "completed_attempts": 0,
                }
            )
            continue

        if not subset.selected_ids:
            analysis_runs = group[group["restart_stage"] == "direct_hf"].copy()
            expected_analysis_runs = 1
            analysis_scope = "direct_hf"
            rescue_triggered = False
        else:
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

        stable, warnings = assess_analysis_runs(
            analysis_runs,
            expected_analysis_runs,
            disagreement_tolerance_ha,
        )
        expected_attempts = (
            1
            if not subset.selected_ids
            else base_restarts + (rescue_restarts if rescue_triggered else 0)
        )
        attempts_complete = len(group) == expected_attempts
        valid = analysis_runs[
            finite_series(analysis_runs, "absolute_error_ha").notna()
        ].copy()
        errors = finite_series(valid, "absolute_error_ha")
        energies = finite_series(valid, "vqe_total_energy_ha")
        chemical_rate = (
            float((errors <= diag.CHEMICAL_ACCURACY_HA).mean())
            if not errors.empty
            else np.nan
        )
        robust_chemical: bool | float = np.nan
        if stable and not errors.empty:
            robust_chemical = bool((errors <= diag.CHEMICAL_ACCURACY_HA).all())
        if not attempts_complete:
            classification = "incomplete"
        elif not stable:
            classification = "unresolved_optimizer_behavior"
        elif bool(robust_chemical):
            classification = "robust_chemical_accuracy"
        else:
            classification = "stable_nonchemical"

        first = group.iloc[0]
        rescue_reason_values = group["rescue_trigger_reasons"].dropna().astype(str)
        summary_rows.append(
            {
                "subset_mask": subset.mask,
                "subset_name": subset.name,
                "selected_ids_json": json.dumps(list(subset.selected_ids)),
                "selected_count": len(subset.selected_ids),
                "contains_0": 0 in subset.selected_ids,
                "contains_3": 3 in subset.selected_ids,
                "contains_5": 5 in subset.selected_ids,
                "contains_10": 10 in subset.selected_ids,
                "contains_12": 12 in subset.selected_ids,
                "contains_15": 15 in subset.selected_ids,
                "completed_attempts": int(len(group)),
                "expected_attempts": expected_attempts,
                "analysis_restart_scope": analysis_scope,
                "analysis_restarts": int(len(analysis_runs)),
                "rescue_triggered": rescue_triggered,
                "rescue_trigger_reasons": (
                    rescue_reason_values.iloc[0] if not rescue_reason_values.empty else None
                ),
                "analysis_complete": attempts_complete,
                "analysis_stable": stable,
                "stability_warnings_json": json.dumps(warnings),
                "classification": classification,
                "best_total_energy_ha": float(energies.min()) if not energies.empty else np.nan,
                "mean_total_energy_ha": float(energies.mean()) if not energies.empty else np.nan,
                "best_error_ha": float(errors.min()) if not errors.empty else np.nan,
                "median_error_ha": float(errors.median()) if not errors.empty else np.nan,
                "mean_error_ha": float(errors.mean()) if not errors.empty else np.nan,
                "std_error_ha": (
                    float(errors.std(ddof=1)) if len(errors) > 1 else 0.0
                ),
                "worst_error_ha": float(errors.max()) if not errors.empty else np.nan,
                "energy_spread_ha": (
                    float(energies.max() - energies.min()) if not energies.empty else np.nan
                ),
                "chemical_success_rate": chemical_rate,
                "robust_chemical_accuracy": robust_chemical,
                "good_success_rate": (
                    float((errors <= diag.GOOD_ACCURACY_HA).mean())
                    if not errors.empty
                    else np.nan
                ),
                "acceptable_success_rate": (
                    float((errors <= diag.ACCEPTABLE_ACCURACY_HA).mean())
                    if not errors.empty
                    else np.nan
                ),
                "optimizer_success_rate": (
                    float(analysis_runs["optimizer_success"].map(as_bool).mean())
                    if not analysis_runs.empty
                    else np.nan
                ),
                "median_iterations": (
                    float(pd.to_numeric(valid["num_iterations"], errors="coerce").median())
                    if not valid.empty
                    else np.nan
                ),
                "median_elapsed_seconds": (
                    float(pd.to_numeric(valid["elapsed_seconds"], errors="coerce").median())
                    if not valid.empty
                    else np.nan
                ),
                "total_elapsed_seconds_all_attempts": float(
                    pd.to_numeric(group["elapsed_seconds"], errors="coerce").fillna(0).sum()
                ),
                "logical_depth": first.get("logical_depth"),
                "compiled_depth": first.get("compiled_depth"),
                "compiled_cx": first.get("compiled_cx"),
                "num_parameters": int(first.get("num_parameters", len(subset.selected_ids))),
                "any_variational_violation": bool(
                    group["variational_violation"].map(as_bool).any()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["selected_count", "subset_mask"]
    )
    summary.to_csv(output_dir / "phase2b_subset_summary.csv", index=False)
    return summary


def robust_chemical_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        summary["analysis_stable"].map(as_bool)
        & summary["robust_chemical_accuracy"].map(as_bool)
    ].copy()


def make_operator_necessity(
    summary: pd.DataFrame,
    analysis_complete: bool,
    output_dir: Path,
) -> pd.DataFrame:
    chemical = robust_chemical_rows(summary)
    rows = []
    for candidate_id in ACTIVE_IDS:
        contains = summary[f"contains_{candidate_id}"].map(as_bool)
        with_operator = summary[contains & summary["analysis_stable"].map(as_bool)]
        without_operator = summary[~contains & summary["analysis_stable"].map(as_bool)]
        chemical_with = chemical[chemical[f"contains_{candidate_id}"].map(as_bool)]
        chemical_without = chemical[~chemical[f"contains_{candidate_id}"].map(as_bool)]
        rows.append(
            {
                "candidate_id": candidate_id,
                "chemical_subsets_with_operator": int(len(chemical_with)),
                "chemical_subsets_without_operator": int(len(chemical_without)),
                "best_error_with_operator_ha": pd.to_numeric(
                    with_operator["mean_error_ha"], errors="coerce"
                ).min(),
                "best_error_without_operator_ha": pd.to_numeric(
                    without_operator["mean_error_ha"], errors="coerce"
                ).min(),
                "present_in_every_chemical_subset": (
                    bool(len(chemical) > 0 and len(chemical_without) == 0)
                    if analysis_complete
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "phase2b_operator_necessity.csv", index=False)
    return result


def pareto_front(summary: pd.DataFrame, error_tolerance_ha: float = 1e-10) -> pd.DataFrame:
    candidates = summary[
        summary["analysis_stable"].map(as_bool)
        & np.isfinite(pd.to_numeric(summary["mean_error_ha"], errors="coerce"))
        & np.isfinite(pd.to_numeric(summary["compiled_cx"], errors="coerce"))
    ].copy()
    keep = []
    for index, row in candidates.iterrows():
        error = float(row["mean_error_ha"])
        cx = float(row["compiled_cx"])
        dominated = False
        for other_index, other in candidates.iterrows():
            if other_index == index:
                continue
            other_error = float(other["mean_error_ha"])
            other_cx = float(other["compiled_cx"])
            no_worse = other_error <= error + error_tolerance_ha and other_cx <= cx
            strictly_better = (
                other_error < error - error_tolerance_ha or other_cx < cx
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(index)
    return candidates.loc[keep].sort_values(["compiled_cx", "mean_error_ha"])


def mask_for_ids(ids: Iterable[int]) -> int:
    wanted = set(ids)
    return sum(1 << bit for bit, candidate_id in enumerate(ACTIVE_IDS) if candidate_id in wanted)


def make_pair_interactions(summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute the optimized-energy nonadditivity of pair 3/12 in 16 contexts."""
    by_mask = summary.set_index("subset_mask")
    background_ids = tuple(candidate_id for candidate_id in ACTIVE_IDS if candidate_id not in {3, 12})
    rows: list[dict[str, Any]] = []
    for size in range(len(background_ids) + 1):
        for background in itertools.combinations(background_ids, size):
            masks = {
                "neither": mask_for_ids(background),
                "only_3": mask_for_ids((*background, 3)),
                "only_12": mask_for_ids((*background, 12)),
                "both": mask_for_ids((*background, 3, 12)),
            }
            selected_rows = {name: by_mask.loc[mask] for name, mask in masks.items()}
            stable = all(as_bool(row["analysis_stable"]) for row in selected_rows.values())
            energies = {
                name: float(row["mean_total_energy_ha"])
                if stable and pd.notna(row["mean_total_energy_ha"])
                else np.nan
                for name, row in selected_rows.items()
            }
            nonadditivity = (
                energies["both"]
                - energies["only_3"]
                - energies["only_12"]
                + energies["neither"]
                if stable
                else np.nan
            )
            rows.append(
                {
                    "background_ids_json": json.dumps(list(background)),
                    "background_count": len(background),
                    "mask_neither": masks["neither"],
                    "mask_only_3": masks["only_3"],
                    "mask_only_12": masks["only_12"],
                    "mask_both": masks["both"],
                    "all_four_subsets_stable": stable,
                    "energy_neither_ha": energies["neither"],
                    "energy_only_3_ha": energies["only_3"],
                    "energy_only_12_ha": energies["only_12"],
                    "energy_both_ha": energies["both"],
                    "energy_lowering_3_without_12_ha": (
                        energies["only_3"] - energies["neither"] if stable else np.nan
                    ),
                    "energy_lowering_12_without_3_ha": (
                        energies["only_12"] - energies["neither"] if stable else np.nan
                    ),
                    "energy_lowering_3_with_12_ha": (
                        energies["both"] - energies["only_12"] if stable else np.nan
                    ),
                    "energy_lowering_12_with_3_ha": (
                        energies["both"] - energies["only_3"] if stable else np.nan
                    ),
                    "pair_nonadditivity_ha": nonadditivity,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "phase2b_pair_3_12_interactions.csv", index=False)
    return result


def make_conclusions(
    system: diag.SystemData,
    protocol: str,
    summary: pd.DataFrame,
    necessity: pd.DataFrame,
    pareto: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    analysis_complete = bool(
        len(summary) == len(ALL_SUBSETS)
        and summary["analysis_complete"].map(as_bool).all()
        and summary["analysis_stable"].map(as_bool).all()
    )
    chemical = robust_chemical_rows(summary)
    minimum_count = (
        int(chemical["selected_count"].min()) if not chemical.empty else None
    )
    minimal = (
        chemical[chemical["selected_count"] == minimum_count]
        if minimum_count is not None
        else chemical.iloc[0:0]
    )
    required_ids: list[int] | None = None
    if analysis_complete and not chemical.empty:
        required_ids = [
            int(row["candidate_id"])
            for _, row in necessity.iterrows()
            if as_bool(row["present_in_every_chemical_subset"])
        ]
    counts_by_size = {
        str(size): int((chemical["selected_count"] == size).sum())
        for size in range(len(ACTIVE_IDS) + 1)
    }
    conclusions = {
        "analysis_status": "complete" if analysis_complete else "partial_or_unstable",
        "protocol_fingerprint": protocol,
        "configuration_fingerprint": system.fingerprint,
        "exact_total_energy_ha": system.exact_total_ha,
        "active_operator_universe": list(ACTIVE_IDS),
        "operator_ordering": OPERATOR_ORDERING,
        "num_subsets_expected": len(ALL_SUBSETS),
        "num_subsets_stable": int(summary["analysis_stable"].map(as_bool).sum()),
        "num_subsets_requiring_rescue": int(summary["rescue_triggered"].map(as_bool).sum()),
        "num_robust_chemical_subsets": int(len(chemical)),
        "chemical_subsets_by_cardinality": counts_by_size,
        "minimum_chemical_subset_cardinality": minimum_count if analysis_complete else None,
        "minimal_chemical_subsets": (
            [
                {
                    "subset_mask": int(row["subset_mask"]),
                    "selected_ids": json.loads(row["selected_ids_json"]),
                    "mean_error_ha": float(row["mean_error_ha"]),
                    "compiled_depth": nullable_int(row["compiled_depth"]),
                    "compiled_cx": nullable_int(row["compiled_cx"]),
                }
                for _, row in minimal.iterrows()
            ]
            if analysis_complete
            else None
        ),
        "operators_present_in_every_chemical_subset": required_ids,
        "exhaustive_tests_within_active_six_under_protocol": {
            "candidate_3_required_for_observed_chemical_accuracy": (
                bool(required_ids is not None and 3 in required_ids)
                if analysis_complete
                else None
            ),
            "candidate_12_required_for_observed_chemical_accuracy": (
                bool(required_ids is not None and 12 in required_ids)
                if analysis_complete
                else None
            ),
            "complementary_pair_3_12_required_for_observed_chemical_accuracy": (
                bool(required_ids is not None and {3, 12}.issubset(required_ids))
                if analysis_complete
                else None
            ),
            "candidate_15_required_for_observed_chemical_accuracy": (
                bool(required_ids is not None and 15 in required_ids)
                if analysis_complete
                else None
            ),
        },
        "pareto_front_masks_error_vs_cx": [
            int(value) for value in pareto["subset_mask"].tolist()
        ],
        "thresholds_ha": {
            "chemical": diag.CHEMICAL_ACCURACY_HA,
            "good": diag.GOOD_ACCURACY_HA,
            "acceptable": diag.ACCEPTABLE_ACCURACY_HA,
        },
        "claim_boundary": (
            "The support enumeration covers all 64 subsets only within the six-operator "
            "universe {0,3,5,10,12,15}, using the audited frozen-core LiH/STO-3G Hamiltonian "
            "at 1.595 Angstrom, one-repetition selected UCC, ascending candidate-ID "
            "operator order, statevector evaluation, and the declared optimization "
            "protocol. Repeated local optimization is not a mathematical proof of each "
            "continuous global minimum. The results do not establish necessity relative "
            "to the other ten "
            "operators, another operator order or ansatz, another geometry, or another "
            "molecular system."
        ),
    }
    diag.write_json(output_dir / "phase2b_exhaustive_conclusions.json", conclusions)
    return conclusions


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def write_markdown_report(
    summary: pd.DataFrame,
    necessity: pd.DataFrame,
    pareto: pd.DataFrame,
    conclusions: dict[str, Any],
    output_dir: Path,
) -> None:
    chemical = robust_chemical_rows(summary).sort_values(
        ["selected_count", "compiled_cx", "mean_error_ha"]
    )
    lines = [
        "# LiH Phase 2B Exhaustive-Support Report",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        f"Stable subsets: **{conclusions['num_subsets_stable']}/64**",
        "",
        f"Rescue-triggered subsets: **{conclusions['num_subsets_requiring_rescue']}**",
        "",
        f"Robust chemical-accuracy subsets: **{conclusions['num_robust_chemical_subsets']}**",
        "",
        "## Robust chemical-accuracy subsets",
        "",
        "| Mask | Candidate IDs | Count | Mean error (Ha) | Depth | CNOTs |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    if chemical.empty:
        lines.append("| NA | No stable chemical-accuracy subset available | NA | NA | NA | NA |")
    else:
        for _, row in chemical.iterrows():
            lines.append(
                f"| {int(row['subset_mask'])} | `{row['selected_ids_json']}` | "
                f"{int(row['selected_count'])} | {fmt(row['mean_error_ha'], 9)} | "
                f"{fmt(row['compiled_depth'], 0)} | {fmt(row['compiled_cx'], 0)} |"
            )

    lines.extend(
        [
            "",
            "## Operator necessity within the six-operator universe",
            "",
            "| Candidate | Chemical subsets with | Chemical subsets without | Required in every chemical subset |",
            "|---:|---:|---:|---|",
        ]
    )
    for _, row in necessity.iterrows():
        required = row["present_in_every_chemical_subset"]
        required_text = "NA" if pd.isna(required) else str(bool(required))
        lines.append(
            f"| {int(row['candidate_id'])} | "
            f"{int(row['chemical_subsets_with_operator'])} | "
            f"{int(row['chemical_subsets_without_operator'])} | {required_text} |"
        )

    lines.extend(
        [
            "",
            "## Error-versus-CNOT Pareto front",
            "",
            "| Mask | Candidate IDs | Error (Ha) | CNOTs |",
            "|---:|---|---:|---:|",
        ]
    )
    for _, row in pareto.iterrows():
        lines.append(
            f"| {int(row['subset_mask'])} | `{row['selected_ids_json']}` | "
            f"{fmt(row['mean_error_ha'], 9)} | {fmt(row['compiled_cx'], 0)} |"
        )

    lines.extend(["", "## Claim boundary", "", conclusions["claim_boundary"], ""])
    (output_dir / "phase2b_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively optimize all 64 subsets of the discovered six-operator "
            "LiH support with adaptive rescue restarts."
        )
    )
    parser.add_argument(
        "--subset-mask",
        action="append",
        type=int,
        choices=range(64),
        help="Run only this mask (0--63); repeat for multiple masks. Default: all 64.",
    )
    parser.add_argument("--bond-length", type=float, default=1.595)
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--base-restarts", type=int, default=2)
    parser.add_argument("--rescue-restarts", type=int, default=3)
    parser.add_argument("--disagreement-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--optimizer", choices=["SLSQP", "COBYLA", "L-BFGS-B"], default="SLSQP"
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--optimizer-tolerance", type=float, default=1e-9)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--parameter-seed", type=int, default=20260722)
    parser.add_argument("--expected-exact", type=float, default=EXPECTED_EXACT_TOTAL_HA)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=Path, default=Path("lih_phase2b_output"))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume matching checkpointed runs (default: true).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subsets_to_run = selected_subsets(args.subset_mask)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = diag.MolecularConfig(
        atom=f"Li 0 0 0; H 0 0 {args.bond_length}",
        basis=args.basis,
        charge=0,
        spin=0,
        unit="ANGSTROM",
        freeze_core=True,
        mapper="JordanWignerMapper",
        excitation_rank=2,
        preserve_spin=True,
    )
    print("Building and auditing the Phase-2B LiH Hamiltonian...", flush=True)
    system = diag.build_system(config)
    validate_candidate_pool(system)
    audit = diag.validate_reference(system, [args.expected_exact], args.audit_tolerance)
    diag.write_json(args.output_dir / "phase2b_reference_audit.json", audit)
    diag.write_candidate_pool(system, args.output_dir / "phase2b_candidate_pool.csv")
    if not audit["audit_passed"]:
        raise RuntimeError("The internal Hamiltonian audit failed; Phase 2B was aborted.")
    if abs(system.exact_total_ha - args.expected_exact) > args.audit_tolerance:
        raise RuntimeError(
            "The exact energy does not match the audited Phase-1/2A reference. "
            "Check bond length, basis, frozen-core configuration, and environment."
        )

    runs, protocol = run_experiment(
        system=system,
        subsets_to_run=subsets_to_run,
        output_dir=args.output_dir,
        optimizer=args.optimizer,
        maxiter=args.maxiter,
        optimizer_tolerance=args.optimizer_tolerance,
        initial_scale=args.initial_scale,
        parameter_seed=args.parameter_seed,
        base_restarts=args.base_restarts,
        rescue_restarts=args.rescue_restarts,
        disagreement_tolerance_ha=args.disagreement_tolerance,
        resume=args.resume,
    )
    summary = summarize_subsets(
        runs,
        args.output_dir,
        args.base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
    )
    analysis_complete = bool(
        summary["analysis_complete"].map(as_bool).all()
        and summary["analysis_stable"].map(as_bool).all()
    )
    necessity = make_operator_necessity(summary, analysis_complete, args.output_dir)
    pareto = pareto_front(summary)
    pareto.to_csv(args.output_dir / "phase2b_pareto_front.csv", index=False)
    make_pair_interactions(summary, args.output_dir)
    chemical = robust_chemical_rows(summary).sort_values(
        ["selected_count", "compiled_cx", "mean_error_ha"]
    )
    chemical.to_csv(args.output_dir / "phase2b_chemical_subsets.csv", index=False)
    conclusions = make_conclusions(
        system,
        protocol,
        summary,
        necessity,
        pareto,
        args.output_dir,
    )
    write_markdown_report(summary, necessity, pareto, conclusions, args.output_dir)

    print("\nPhase 2B status:")
    display_columns = [
        "subset_mask",
        "selected_ids_json",
        "selected_count",
        "mean_error_ha",
        "robust_chemical_accuracy",
        "compiled_depth",
        "compiled_cx",
        "rescue_triggered",
        "classification",
    ]
    completed_display = summary[summary["classification"] != "not_run"]
    print(completed_display.reindex(columns=display_columns).to_string(index=False))
    print("\nExhaustive conclusions:")
    print(json.dumps(diag.json_ready(conclusions), indent=2, sort_keys=True))
    print(f"\nSaved results in: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed Phase-2B evaluations remain checkpointed.",
            file=sys.stderr,
        )
