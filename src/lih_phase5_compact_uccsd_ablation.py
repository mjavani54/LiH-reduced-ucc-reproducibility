#!/usr/bin/env python3
"""Phase 5A: nested ablation of the compact LiH UCCSD support.

The audited Phase-4 UCCSD scan identified four active singles and independently
rediscovered the six doubles retained by Phase 3:

    singles: S0, S3, S4, S7
    doubles: D0, D3, D5, D10, D12, D15

This script tests that ten-operator support against the full 24-operator UCCSD
parent and five nested controls.  The default pilot uses the two stretched
geometries at which the doubles-only parent failed: 2.5 and 3.0 Angstrom.

Reference operator IDs are translated at every geometry with the sequential
maximum-overlap orbital tracking used in Phase 3.  Checkpoints are normalized
from ``bond_length_angstrom`` rather than trusting CSV-parsed geometry strings;
this fixes the Phase-4 resume-key failure in which ``1.200000`` became ``1.2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import lih_phase2b_exhaustive_support as p2b
    import lih_phase3_geometry_transfer as p3
    import lih_phase4_ansatz_diagnosis as p4
    import lih_reference_failure_diagnostics as diag
except ImportError as exc:
    raise SystemExit(
        "Place lih_phase5_compact_uccsd_ablation.py, "
        "lih_phase4_ansatz_diagnosis.py, lih_phase3_geometry_transfer.py, "
        "lih_phase2b_exhaustive_support.py, and "
        "lih_reference_failure_diagnostics.py in the same directory."
    ) from exc


REFERENCE_BOND_LENGTH = 1.595
EXPECTED_REFERENCE_EXACT_TOTAL_HA = -7.8821745057672805
EXPECTED_ACTIVE_SPATIAL_ORBITALS = 5
EXPECTED_ACTIVE_ELECTRONS = 2
EXPECTED_SINGLE_COUNT = 8
EXPECTED_DOUBLE_COUNT = 16

PILOT_TARGET_BOND_LENGTHS: tuple[float, ...] = (2.5, 3.0)
PILOT_TRACKING_BOND_LENGTHS: tuple[float, ...] = (1.595, 2.1, 2.5, 3.0)
FULL_BOND_LENGTHS: tuple[float, ...] = (
    1.2,
    1.4,
    1.595,
    1.8,
    2.1,
    2.5,
    3.0,
)

ACTIVE_SINGLE_LOCAL_IDS: tuple[int, ...] = (0, 3, 4, 7)
ACTIVE_DOUBLE_LOCAL_IDS: tuple[int, ...] = (0, 3, 5, 10, 12, 15)
ACTIVE_DOUBLE_GLOBAL_IDS: tuple[int, ...] = tuple(
    EXPECTED_SINGLE_COUNT + index for index in ACTIVE_DOUBLE_LOCAL_IDS
)
STRICT_FULL24_TOLERANCE_HA = 1e-6
TRACKING_METHOD = p3.TRACKING_METHOD
OPERATOR_ORDERING = "ascending_reference_global_uccsd_order_index"


@dataclass(frozen=True)
class SupportVariant:
    name: str
    reference_global_ids: tuple[int, ...]
    scientific_role: str
    is_full24: bool = False


VARIANTS: tuple[SupportVariant, ...] = (
    SupportVariant(
        "full24",
        tuple(range(EXPECTED_SINGLE_COUNT + EXPECTED_DOUBLE_COUNT)),
        "Positive control: all eight singles and all sixteen doubles.",
        is_full24=True,
    ),
    SupportVariant(
        "allS_D6",
        tuple(range(EXPECTED_SINGLE_COUNT)) + ACTIVE_DOUBLE_GLOBAL_IDS,
        "Tests removal of the ten inactive doubles while retaining every single.",
    ),
    SupportVariant(
        "S4_allD",
        ACTIVE_SINGLE_LOCAL_IDS
        + tuple(
            range(
                EXPECTED_SINGLE_COUNT,
                EXPECTED_SINGLE_COUNT + EXPECTED_DOUBLE_COUNT,
            )
        ),
        "Tests removal of the four inactive singles while retaining every double.",
    ),
    SupportVariant(
        "compact10",
        tuple(sorted(ACTIVE_SINGLE_LOCAL_IDS + ACTIVE_DOUBLE_GLOBAL_IDS)),
        "Main hypothesis: four active singles plus the discovered six doubles.",
    ),
    SupportVariant(
        "pairA_D6",
        tuple(sorted((0, 4) + ACTIVE_DOUBLE_GLOBAL_IDS)),
        "Compact support retaining only the S0/S4 spin-paired single channel.",
    ),
    SupportVariant(
        "pairB_D6",
        tuple(sorted((3, 7) + ACTIVE_DOUBLE_GLOBAL_IDS)),
        "Compact support retaining only the S3/S7 spin-paired single channel.",
    ),
    SupportVariant(
        "D6_only",
        ACTIVE_DOUBLE_GLOBAL_IDS,
        "Negative control: the transferred six-double support without singles.",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def geometry_key(bond_length: float) -> str:
    """Return the sole canonical geometry representation used by checkpoints."""
    return f"{float(bond_length):.6f}"


def normalized_values(values: Iterable[float]) -> list[float]:
    result = sorted({round(float(value), 9) for value in values})
    if not result or any(value <= 0 for value in result):
        raise ValueError("At least one positive bond length is required.")
    keys = [geometry_key(value) for value in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Bond lengths collide at six-decimal precision.")
    return result


def resolved_grids(
    profile: str,
    overrides: Sequence[float] | None,
) -> tuple[list[float], list[float]]:
    if overrides:
        targets = normalized_values(overrides)
        lower = min([REFERENCE_BOND_LENGTH, *targets])
        upper = max([REFERENCE_BOND_LENGTH, *targets])
        anchors = [
            value
            for value in FULL_BOND_LENGTHS
            if lower - 1e-9 <= value <= upper + 1e-9
        ]
        tracking = normalized_values([REFERENCE_BOND_LENGTH, *anchors, *targets])
    elif profile == "pilot":
        targets = list(PILOT_TARGET_BOND_LENGTHS)
        tracking = list(PILOT_TRACKING_BOND_LENGTHS)
    else:
        targets = list(FULL_BOND_LENGTHS)
        tracking = list(FULL_BOND_LENGTHS)
    if not any(abs(value - REFERENCE_BOND_LENGTH) <= 1e-9 for value in tracking):
        raise RuntimeError("The orbital-tracking grid lacks the reference geometry.")
    return targets, tracking


def selected_variants(names: Sequence[str] | None) -> list[SupportVariant]:
    if not names:
        return list(VARIANTS)
    return [VARIANT_BY_NAME[name] for name in dict.fromkeys(names)]


def build_config(bond_length: float, basis: str) -> diag.MolecularConfig:
    return p3.build_config(bond_length, basis)


def full_pool(
    system: diag.SystemData,
) -> tuple[list[diag.Excitation], list[diag.Excitation], list[diag.Excitation]]:
    singles, doubles = p4.excitation_pools(system)
    if len(singles) != EXPECTED_SINGLE_COUNT or len(doubles) != EXPECTED_DOUBLE_COUNT:
        raise RuntimeError("The UCCSD operator-pool dimensions changed.")
    combined = [*singles, *doubles]
    if len(set(combined)) != len(combined):
        raise RuntimeError("The UCCSD operator pool contains duplicate excitations.")
    return singles, doubles, combined


def operator_label(global_id: int) -> str:
    if global_id < EXPECTED_SINGLE_COUNT:
        return f"S{global_id}"
    return f"D{global_id - EXPECTED_SINGLE_COUNT}"


def operator_type(global_id: int) -> str:
    return "single" if global_id < EXPECTED_SINGLE_COUNT else "double"


def count_types(global_ids: Sequence[int]) -> tuple[int, int]:
    singles = sum(index < EXPECTED_SINGLE_COUNT for index in global_ids)
    return singles, len(global_ids) - singles


def write_operator_pool(
    reference_pool: Sequence[diag.Excitation],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for global_id, excitation in enumerate(reference_pool):
        row: dict[str, Any] = {
            "operator_label": operator_label(global_id),
            "excitation_type": operator_type(global_id),
            "type_local_index": (
                global_id
                if global_id < EXPECTED_SINGLE_COUNT
                else global_id - EXPECTED_SINGLE_COUNT
            ),
            "global_uccsd_order_index": global_id,
            "reference_excitation_json": json.dumps(
                diag.excitation_to_json(excitation)
            ),
            "phase4_empirically_active": global_id
            in set(ACTIVE_SINGLE_LOCAL_IDS + ACTIVE_DOUBLE_GLOBAL_IDS),
        }
        for variant in VARIANTS:
            row[f"included_in_{variant.name}"] = (
                global_id in variant.reference_global_ids
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "phase5_operator_pool.csv", index=False)
    return frame


def variant_degenerate_subspace_sensitive(
    variant: SupportVariant,
    reference_pool: Sequence[diag.Excitation],
    reference_groups: Sequence[tuple[int, ...]],
    num_spatial_orbitals: int,
) -> bool:
    used_virtual_spatial: set[int] = set()
    for global_id in variant.reference_global_ids:
        for spin_orbital in reference_pool[global_id][1]:
            used_virtual_spatial.add(int(spin_orbital) % num_spatial_orbitals)
    for group in reference_groups:
        group_set = set(group)
        overlap = used_virtual_spatial & group_set
        if len(group) > 1 and overlap and overlap != group_set:
            return True
    return False


def build_excitation_mappings(
    systems: dict[str, diag.SystemData],
    pools: dict[str, list[diag.Excitation]],
    snapshots: dict[str, p3.OrbitalSnapshot],
    tracking_bond_lengths: Sequence[float],
    reference_groups: Sequence[tuple[int, ...]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    reference_key = geometry_key(REFERENCE_BOND_LENGTH)
    reference_pool = pools[reference_key]
    if len(reference_pool) != EXPECTED_SINGLE_COUNT + EXPECTED_DOUBLE_COUNT:
        raise RuntimeError("The reference UCCSD pool does not contain 24 operators.")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for tracking_index, bond_length in enumerate(sorted(tracking_bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        snapshot = snapshots[key]
        if snapshot.reference_to_canonical is None:
            raise RuntimeError(f"No orbital mapping exists at R={bond_length}.")
        current_pool = pools[key]
        current_lookup = {
            excitation: global_id
            for global_id, excitation in enumerate(current_pool)
        }
        if len(current_lookup) != len(current_pool):
            raise RuntimeError(f"The current UCCSD pool is not unique at R={bond_length}.")

        for variant in VARIANTS:
            reference_ids = list(variant.reference_global_ids)
            mapped_excitations = [
                p3.translate_excitation(
                    reference_pool[global_id],
                    snapshot.reference_to_canonical,
                    system.problem.num_spatial_orbitals,
                )
                for global_id in reference_ids
            ]
            try:
                mapped_ids = [current_lookup[item] for item in mapped_excitations]
            except KeyError as exc:
                raise RuntimeError(
                    f"A mapped {variant.name} excitation at R={bond_length} "
                    "is absent from the current UCCSD pool."
                ) from exc
            if len(mapped_ids) != len(set(mapped_ids)):
                raise RuntimeError(
                    f"Orbital translation collapsed two {variant.name} operators "
                    f"at R={bond_length}."
                )
            sensitive = variant_degenerate_subspace_sensitive(
                variant,
                reference_pool,
                reference_groups,
                system.problem.num_spatial_orbitals,
            )
            record = {
                "tracking_index": tracking_index,
                "bond_length_angstrom": bond_length,
                "geometry_key": key,
                "variant": variant.name,
                "reference_global_ids": reference_ids,
                "mapped_current_global_ids": mapped_ids,
                "mapped_excitations": mapped_excitations,
                "reference_labels": [operator_label(item) for item in reference_ids],
                "mapped_current_labels": [operator_label(item) for item in mapped_ids],
                "degenerate_subspace_sensitive": sensitive,
                "orbital_mapping_identity": bool(
                    np.array_equal(
                        snapshot.reference_to_canonical,
                        np.arange(system.problem.num_spatial_orbitals),
                    )
                ),
            }
            records[(key, variant.name)] = record
            rows.append(
                {
                    "tracking_index": tracking_index,
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "variant": variant.name,
                    "reference_global_ids_json": json.dumps(reference_ids),
                    "reference_labels_json": json.dumps(record["reference_labels"]),
                    "mapped_current_global_ids_json": json.dumps(mapped_ids),
                    "mapped_current_labels_json": json.dumps(
                        record["mapped_current_labels"]
                    ),
                    "mapped_excitations_json": json.dumps(
                        [
                            diag.excitation_to_json(excitation)
                            for excitation in mapped_excitations
                        ]
                    ),
                    "operator_ordering": OPERATOR_ORDERING,
                    "degenerate_subspace_sensitive": sensitive,
                    "orbital_mapping_identity": record["orbital_mapping_identity"],
                }
            )
    return records, pd.DataFrame(rows)


def make_protocol_fingerprint(
    systems: dict[str, diag.SystemData],
    target_bond_lengths: Sequence[float],
    tracking_bond_lengths: Sequence[float],
    mappings: dict[tuple[str, str], dict[str, Any]],
    pool_hash: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    parameter_seed: int,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
    orbital_overlap_threshold: float,
    degeneracy_tolerance_ha: float,
) -> str:
    payload = {
        "analysis_phase": "Phase 5A compact UCCSD nested ablation",
        "target_bond_lengths_angstrom": list(target_bond_lengths),
        "tracking_bond_lengths_angstrom": list(tracking_bond_lengths),
        "configuration_fingerprints": {
            key: systems[key].fingerprint for key in sorted(systems)
        },
        "operator_pool_fingerprint": pool_hash,
        "variants": [
            {
                "name": variant.name,
                "reference_global_ids": list(variant.reference_global_ids),
            }
            for variant in VARIANTS
        ],
        "mapped_current_global_ids": {
            f"{key}|{variant.name}": mappings[(key, variant.name)][
                "mapped_current_global_ids"
            ]
            for key in sorted(systems)
            for variant in VARIANTS
        },
        "tracking_method": TRACKING_METHOD,
        "operator_ordering": OPERATOR_ORDERING,
        "orbital_overlap_threshold": orbital_overlap_threshold,
        "degeneracy_tolerance_ha": degeneracy_tolerance_ha,
        "ansatz_repetitions": 1,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        "parameter_seed": parameter_seed,
        "base_restarts": base_restarts,
        "rescue_restarts": rescue_restarts,
        "disagreement_tolerance_ha": disagreement_tolerance_ha,
        "initialization_policy": (
            "zero_then_previous_target_geometry_continuation_when_available_"
            "then_random"
        ),
        "checkpoint_geometry_key_policy": (
            "canonicalize_from_bond_length_angstrom_to_six_decimals"
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonicalize_checkpoint(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize resume keys and reject ambiguous duplicate attempts."""
    if frame.empty:
        return frame.copy()
    required = {
        "protocol_fingerprint",
        "bond_length_angstrom",
        "variant",
        "restart",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "The existing Phase-5 checkpoint is missing columns: "
            + ", ".join(missing)
        )
    result = frame.copy()
    bonds = pd.to_numeric(result["bond_length_angstrom"], errors="coerce")
    if bonds.isna().any():
        bad = result.index[bonds.isna()].tolist()
        raise RuntimeError(
            f"The Phase-5 checkpoint has nonnumeric bond lengths in rows {bad}."
        )
    result["bond_length_angstrom"] = bonds.astype(float)
    result["geometry_key"] = bonds.map(geometry_key)
    restarts = pd.to_numeric(result["restart"], errors="coerce")
    if restarts.isna().any() or not np.allclose(restarts, np.round(restarts)):
        raise RuntimeError("The Phase-5 checkpoint contains invalid restart IDs.")
    result["restart"] = restarts.astype(int)
    duplicate_columns = [
        "protocol_fingerprint",
        "geometry_key",
        "variant",
        "restart",
    ]
    duplicates = result.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        values = (
            result.loc[duplicates, duplicate_columns]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "The Phase-5 checkpoint contains duplicate canonical run keys. "
            "Move the checkpoint aside or choose a clean output directory. "
            f"Duplicates: {values[:5]}"
        )
    return result


def best_previous_target_parameters(
    rows: Sequence[dict[str, Any]],
    protocol: str,
    variant_name: str,
    current_target_index: int,
    expected_parameters: int,
) -> tuple[np.ndarray, float] | None:
    if not rows or current_target_index <= 0:
        return None
    frame = canonicalize_checkpoint(pd.DataFrame(rows))
    candidates = frame[
        (frame["protocol_fingerprint"].astype(str) == protocol)
        & (frame["variant"].astype(str) == variant_name)
        & (
            pd.to_numeric(frame["geometry_index"], errors="coerce")
            == current_target_index - 1
        )
    ].copy()
    candidates["_energy"] = pd.to_numeric(
        candidates["vqe_total_energy_ha"], errors="coerce"
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


def common_run_fields(
    system: diag.SystemData,
    geometry_index: int,
    bond_length: float,
    variant: SupportVariant,
    mapping: dict[str, Any],
    protocol: str,
    pool_hash: str,
    optimizer: str,
    maxiter: int,
    optimizer_tolerance: float,
    initial_scale: float,
    resources: dict[str, int | None],
    num_parameters: int,
) -> dict[str, Any]:
    single_count, double_count = count_types(variant.reference_global_ids)
    return {
        "protocol_fingerprint": protocol,
        "configuration_fingerprint": system.fingerprint,
        "operator_pool_fingerprint": pool_hash,
        "geometry_index": geometry_index,
        "bond_length_angstrom": bond_length,
        "geometry_key": geometry_key(bond_length),
        "variant": variant.name,
        "scientific_role": variant.scientific_role,
        "reference_global_ids_json": json.dumps(
            list(variant.reference_global_ids)
        ),
        "reference_labels_json": json.dumps(mapping["reference_labels"]),
        "mapped_current_global_ids_json": json.dumps(
            mapping["mapped_current_global_ids"]
        ),
        "mapped_current_labels_json": json.dumps(
            mapping["mapped_current_labels"]
        ),
        "mapped_excitations_json": json.dumps(
            [
                diag.excitation_to_json(item)
                for item in mapping["mapped_excitations"]
            ]
        ),
        "operator_ordering": OPERATOR_ORDERING,
        "degenerate_subspace_sensitive": mapping[
            "degenerate_subspace_sensitive"
        ],
        "num_single_excitations": single_count,
        "num_double_excitations": double_count,
        "num_unique_excitations": len(mapping["mapped_excitations"]),
        "num_parameters": num_parameters,
        "exact_total_energy_ha": system.exact_total_ha,
        "hf_total_energy_ha": system.hf_total_driver_ha,
        "optimizer": optimizer,
        "maxiter": maxiter,
        "optimizer_tolerance": optimizer_tolerance,
        "initial_scale": initial_scale,
        **resources,
    }


def run_experiment(
    systems: dict[str, diag.SystemData],
    mappings: dict[tuple[str, str], dict[str, Any]],
    target_bond_lengths: Sequence[float],
    variants_to_run: Sequence[SupportVariant],
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
    runs_path = output_dir / "phase5_restart_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if resume and runs_path.exists():
        previous = canonicalize_checkpoint(pd.read_csv(runs_path))
        rows = previous.to_dict(orient="records")
        matching = previous[
            previous["protocol_fingerprint"].astype(str) == protocol
        ]
        completed = {
            (
                geometry_key(row["bond_length_angstrom"]),
                str(row["variant"]),
                int(row["restart"]),
            )
            for row in matching.to_dict(orient="records")
        }
        print(f"Resuming with {len(completed)} completed Phase-5 attempts.")

    def current_runs(key: str, variant_name: str) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = canonicalize_checkpoint(pd.DataFrame(rows))
        return frame[
            (frame["protocol_fingerprint"].astype(str) == protocol)
            & (frame["geometry_key"].astype(str) == key)
            & (frame["variant"].astype(str) == variant_name)
        ].copy()

    for geometry_index, bond_length in enumerate(sorted(target_bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        for variant in variants_to_run:
            variant_index = next(
                index
                for index, declared in enumerate(VARIANTS)
                if declared.name == variant.name
            )
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

            mapping = mappings[(key, variant.name)]
            excitations = mapping["mapped_excitations"]
            ansatz = p4.build_ansatz(system, excitations, reps=1)
            if ansatz.num_parameters != len(excitations):
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
                mapping,
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
                    previous = best_previous_target_parameters(
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
                    "continuation_source_bond_length_angstrom": continuation_source,
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

    all_runs = canonicalize_checkpoint(pd.DataFrame(rows))
    return all_runs[
        all_runs["protocol_fingerprint"].astype(str) == protocol
    ].copy()


def summarize_runs(
    runs: pd.DataFrame,
    systems: dict[str, diag.SystemData],
    mappings: dict[tuple[str, str], dict[str, Any]],
    target_bond_lengths: Sequence[float],
    variants_to_run: Sequence[SupportVariant],
    output_dir: Path,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for geometry_index, bond_length in enumerate(sorted(target_bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        for variant in variants_to_run:
            group = runs[
                (runs["geometry_key"].astype(str) == key)
                & (runs["variant"].astype(str) == variant.name)
            ].sort_values("restart")
            if group.empty:
                raise RuntimeError(
                    f"No Phase-5 runs exist for R={bond_length}, {variant.name}."
                )
            rescue_triggered = bool(
                (group["restart_stage"].astype(str) == "rescue").any()
            )
            expected_attempts = base_restarts + (
                rescue_restarts if rescue_triggered else 0
            )
            attempts_complete = len(group) == expected_attempts
            stable, warnings = p2b.assess_analysis_runs(
                group,
                expected_attempts,
                disagreement_tolerance_ha,
            )
            errors = p2b.finite_series(group, "absolute_error_ha")
            energies = p2b.finite_series(group, "vqe_total_energy_ha")
            valid_mask = errors.notna() & energies.notna()
            valid_errors = errors[valid_mask]
            valid_energies = energies[valid_mask]
            robust_chemical: bool | float = np.nan
            if stable and not valid_errors.empty:
                robust_chemical = bool(
                    (valid_errors <= diag.CHEMICAL_ACCURACY_HA).all()
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
            mapping = mappings[(key, variant.name)]
            single_count, double_count = count_types(
                variant.reference_global_ids
            )
            reason_values = group["rescue_trigger_reasons"].dropna().astype(str)
            rows.append(
                {
                    "geometry_index": geometry_index,
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "configuration_fingerprint": system.fingerprint,
                    "variant": variant.name,
                    "scientific_role": variant.scientific_role,
                    "reference_global_ids_json": json.dumps(
                        list(variant.reference_global_ids)
                    ),
                    "reference_labels_json": json.dumps(
                        mapping["reference_labels"]
                    ),
                    "mapped_current_global_ids_json": json.dumps(
                        mapping["mapped_current_global_ids"]
                    ),
                    "num_single_excitations": single_count,
                    "num_double_excitations": double_count,
                    "num_unique_excitations": len(
                        variant.reference_global_ids
                    ),
                    "num_parameters": int(first["num_parameters"]),
                    "degenerate_subspace_sensitive": mapping[
                        "degenerate_subspace_sensitive"
                    ],
                    "completed_attempts": int(len(group)),
                    "expected_attempts": expected_attempts,
                    "rescue_triggered": rescue_triggered,
                    "rescue_trigger_reasons": (
                        reason_values.iloc[0] if not reason_values.empty else None
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
                        float(valid_energies.min())
                        if not valid_energies.empty
                        else np.nan
                    ),
                    "mean_total_energy_ha": (
                        float(valid_energies.mean())
                        if not valid_energies.empty
                        else np.nan
                    ),
                    "best_error_exact_ha": (
                        float(valid_errors.min())
                        if not valid_errors.empty
                        else np.nan
                    ),
                    "mean_error_exact_ha": (
                        float(valid_errors.mean())
                        if not valid_errors.empty
                        else np.nan
                    ),
                    "worst_error_exact_ha": (
                        float(valid_errors.max())
                        if not valid_errors.empty
                        else np.nan
                    ),
                    "energy_spread_ha": (
                        float(valid_energies.max() - valid_energies.min())
                        if not valid_energies.empty
                        else np.nan
                    ),
                    "chemical_success_rate": (
                        float(
                            (
                                valid_errors
                                <= diag.CHEMICAL_ACCURACY_HA
                            ).mean()
                        )
                        if not valid_errors.empty
                        else np.nan
                    ),
                    "robust_chemical_accuracy": robust_chemical,
                    "optimizer_success_rate": float(
                        group["optimizer_success"].map(p2b.as_bool).mean()
                    ),
                    "median_iterations": float(
                        pd.to_numeric(
                            group["num_iterations"], errors="coerce"
                        ).median()
                    ),
                    "median_elapsed_seconds": float(
                        pd.to_numeric(
                            group["elapsed_seconds"], errors="coerce"
                        ).median()
                    ),
                    "total_elapsed_seconds_all_attempts": float(
                        pd.to_numeric(
                            group["elapsed_seconds"], errors="coerce"
                        )
                        .fillna(0)
                        .sum()
                    ),
                    "logical_depth": first.get("logical_depth"),
                    "compiled_depth": first.get("compiled_depth"),
                    "compiled_cx": first.get("compiled_cx"),
                    "any_variational_violation": bool(
                        group["variational_violation"].map(p2b.as_bool).any()
                    ),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["geometry_index", "variant"])
    summary = add_full24_comparisons(summary)
    summary.to_csv(
        output_dir / "phase5_variant_geometry_summary.csv",
        index=False,
    )
    return summary


def numeric(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def add_full24_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    defaults: dict[str, Any] = {
        "full24_mean_total_energy_ha": np.nan,
        "full24_mean_error_exact_ha": np.nan,
        "signed_energy_difference_vs_full24_ha": np.nan,
        "absolute_energy_difference_vs_full24_ha": np.nan,
        "matches_full24_within_1e-6_ha": pd.NA,
        "parameter_reduction_fraction_vs_full24": np.nan,
        "compiled_depth_reduction_fraction_vs_full24": np.nan,
        "compiled_cx_reduction_fraction_vs_full24": np.nan,
    }
    for column, default in defaults.items():
        result[column] = default

    for _, indices in result.groupby("geometry_key").groups.items():
        geometry = result.loc[indices]
        baseline_rows = geometry[geometry["variant"] == "full24"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        if not p2b.as_bool(baseline["analysis_stable"]):
            continue
        baseline_energy = numeric(baseline["mean_total_energy_ha"])
        baseline_error = numeric(baseline["mean_error_exact_ha"])
        if baseline_energy is None or baseline_error is None:
            continue
        for index in indices:
            row = result.loc[index]
            result.at[index, "full24_mean_total_energy_ha"] = baseline_energy
            result.at[index, "full24_mean_error_exact_ha"] = baseline_error
            if p2b.as_bool(row["analysis_stable"]):
                row_energy = numeric(row["mean_total_energy_ha"])
                if row_energy is not None:
                    difference = row_energy - baseline_energy
                    result.at[
                        index,
                        "signed_energy_difference_vs_full24_ha",
                    ] = difference
                    result.at[
                        index,
                        "absolute_energy_difference_vs_full24_ha",
                    ] = abs(difference)
                    result.at[
                        index,
                        "matches_full24_within_1e-6_ha",
                    ] = abs(difference) <= STRICT_FULL24_TOLERANCE_HA
            for resource, output_column in (
                ("num_parameters", "parameter_reduction_fraction_vs_full24"),
                (
                    "compiled_depth",
                    "compiled_depth_reduction_fraction_vs_full24",
                ),
                ("compiled_cx", "compiled_cx_reduction_fraction_vs_full24"),
            ):
                baseline_value = numeric(baseline[resource])
                row_value = numeric(row[resource])
                if (
                    baseline_value is not None
                    and row_value is not None
                    and baseline_value > 0
                ):
                    result.at[index, output_column] = (
                        1.0 - row_value / baseline_value
                    )
    return result


def strict_pass(row: pd.Series | None) -> bool | None:
    if row is None:
        return None
    if not p2b.as_bool(row["analysis_stable"]):
        return None
    match = row["matches_full24_within_1e-6_ha"]
    if pd.isna(match):
        return None
    return bool(
        p2b.as_bool(row["robust_chemical_accuracy"])
        and p2b.as_bool(match)
    )


def variant_row(
    geometry: pd.DataFrame,
    variant_name: str,
) -> pd.Series | None:
    selected = geometry[geometry["variant"] == variant_name]
    return None if selected.empty else selected.iloc[0]


def causal_tests_for_geometry(geometry: pd.DataFrame) -> dict[str, Any]:
    rows = {
        variant.name: variant_row(geometry, variant.name)
        for variant in VARIANTS
    }
    full = rows["full24"]
    compact = rows["compact10"]
    pair_a_only = rows["pairA_D6"]
    pair_b_only = rows["pairB_D6"]
    doubles_only = rows["D6_only"]

    full_control_chemical = (
        None
        if full is None or not p2b.as_bool(full["analysis_stable"])
        else p2b.as_bool(full["robust_chemical_accuracy"])
    )
    compact_pass = strict_pass(compact)
    pair_a_pass = strict_pass(pair_a_only)
    pair_b_pass = strict_pass(pair_b_only)
    d6_nonchemical = (
        None
        if doubles_only is None
        or not p2b.as_bool(doubles_only["analysis_stable"])
        else not p2b.as_bool(doubles_only["robust_chemical_accuracy"])
    )
    return {
        "full24_positive_control_chemical_accuracy": full_control_chemical,
        "inactive_doubles_removable_with_all_singles": strict_pass(
            rows["allS_D6"]
        ),
        "inactive_singles_removable_with_all_doubles": strict_pass(
            rows["S4_allD"]
        ),
        "compact10_strict_success": compact_pass,
        "pairA_D6_strict_success": pair_a_pass,
        "pairB_D6_strict_success": pair_b_pass,
        "pairA_S0_S4_required_within_compact10": (
            None
            if compact_pass is not True or pair_b_pass is None
            else not pair_b_pass
        ),
        "pairB_S3_S7_required_within_compact10": (
            None
            if compact_pass is not True or pair_a_pass is None
            else not pair_a_pass
        ),
        "D6_only_is_stable_nonchemical_negative_control": d6_nonchemical,
    }


def make_geometry_summary(
    variant_summary: pd.DataFrame,
    systems: dict[str, diag.SystemData],
    target_bond_lengths: Sequence[float],
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for geometry_index, bond_length in enumerate(sorted(target_bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        group = variant_summary[variant_summary["geometry_key"] == key]
        stable = group[group["analysis_stable"].map(p2b.as_bool)]
        chemical = stable[
            stable["robust_chemical_accuracy"].map(p2b.as_bool)
        ]
        best = (
            None
            if stable.empty
            else stable.sort_values("mean_error_exact_ha").iloc[0]
        )
        row: dict[str, Any] = {
            "geometry_index": geometry_index,
            "bond_length_angstrom": bond_length,
            "configuration_fingerprint": system.fingerprint,
            "exact_total_energy_ha": system.exact_total_ha,
            "hf_total_energy_ha": system.hf_total_driver_ha,
            "hf_error_ha": (
                system.hf_total_driver_ha - system.exact_total_ha
            ),
            "num_variants_stable": int(len(stable)),
            "num_variants_chemical": int(len(chemical)),
            "best_variant": None if best is None else str(best["variant"]),
            "best_mean_error_exact_ha": (
                np.nan if best is None else float(best["mean_error_exact_ha"])
            ),
        }
        tests = causal_tests_for_geometry(group)
        row.update(tests)
        for name in ("full24", "compact10", "D6_only"):
            selected = variant_row(group, name)
            row[f"{name}_mean_error_exact_ha"] = (
                np.nan
                if selected is None
                else selected.get("mean_error_exact_ha", np.nan)
            )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("geometry_index")
    result.to_csv(output_dir / "phase5_geometry_summary.csv", index=False)
    return result


def nullable_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return p2b.as_bool(value)


def make_conclusions(
    profile: str,
    protocol: str,
    pool_hash: str,
    target_bond_lengths: Sequence[float],
    tracking_bond_lengths: Sequence[float],
    variant_summary: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    orbital_summary: pd.DataFrame,
    variants_to_run: Sequence[SupportVariant],
    output_dir: Path,
) -> dict[str, Any]:
    requested_names = {variant.name for variant in variants_to_run}
    expected_names = set(VARIANT_BY_NAME)
    all_combinations_stable = bool(
        len(variant_summary) == len(target_bond_lengths) * len(variants_to_run)
        and variant_summary["analysis_complete"].map(p2b.as_bool).all()
        and variant_summary["analysis_stable"].map(p2b.as_bool).all()
    )
    all_variants_requested = requested_names == expected_names
    all_orbitals_reliable = bool(
        orbital_summary["tracking_reliable"].map(p2b.as_bool).all()
    )
    if all_combinations_stable and all_variants_requested and profile == "pilot":
        analysis_status = "pilot_complete"
    elif all_combinations_stable and all_variants_requested and profile == "full":
        analysis_status = "full_grid_complete"
    elif all_combinations_stable:
        analysis_status = "partial_ablation"
    else:
        analysis_status = "partial_or_unresolved"

    causal_by_geometry = {
        geometry_key(row["bond_length_angstrom"]): {
            key: nullable_bool(row[key])
            for key in (
                "full24_positive_control_chemical_accuracy",
                "inactive_doubles_removable_with_all_singles",
                "inactive_singles_removable_with_all_doubles",
                "compact10_strict_success",
                "pairA_D6_strict_success",
                "pairB_D6_strict_success",
                "pairA_S0_S4_required_within_compact10",
                "pairB_S3_S7_required_within_compact10",
                "D6_only_is_stable_nonchemical_negative_control",
            )
        }
        for _, row in geometry_summary.iterrows()
    }
    compact_values = [
        tests["compact10_strict_success"]
        for tests in causal_by_geometry.values()
    ]
    compact_all = (
        None
        if any(value is None for value in compact_values)
        else all(compact_values)
    )

    variant_results: dict[str, Any] = {}
    for variant in VARIANTS:
        group = variant_summary[
            variant_summary["variant"] == variant.name
        ].sort_values("geometry_index")
        stable = group[group["analysis_stable"].map(p2b.as_bool)]
        chemical = stable[
            stable["robust_chemical_accuracy"].map(p2b.as_bool)
        ]
        strict = stable[
            stable["matches_full24_within_1e-6_ha"]
            .map(p2b.as_bool)
            & stable["robust_chemical_accuracy"].map(p2b.as_bool)
        ]
        variant_results[variant.name] = {
            "reference_labels": [
                operator_label(index)
                for index in variant.reference_global_ids
            ],
            "num_operators": len(variant.reference_global_ids),
            "num_geometries_requested": int(len(group)),
            "num_geometries_stable": int(len(stable)),
            "num_geometries_chemical_accuracy": int(len(chemical)),
            "num_geometries_strict_full24_equivalent_and_chemical": int(
                len(strict)
            ),
            "worst_mean_error_exact_ha": numeric(
                pd.to_numeric(
                    stable["mean_error_exact_ha"], errors="coerce"
                ).max()
            ),
            "worst_absolute_energy_difference_vs_full24_ha": numeric(
                pd.to_numeric(
                    stable["absolute_energy_difference_vs_full24_ha"],
                    errors="coerce",
                ).max()
            ),
            "median_parameter_reduction_fraction_vs_full24": numeric(
                pd.to_numeric(
                    stable["parameter_reduction_fraction_vs_full24"],
                    errors="coerce",
                ).median()
            ),
            "median_compiled_cx_reduction_fraction_vs_full24": numeric(
                pd.to_numeric(
                    stable["compiled_cx_reduction_fraction_vs_full24"],
                    errors="coerce",
                ).median()
            ),
        }

    conclusions = {
        "analysis_status": analysis_status,
        "profile": profile,
        "protocol_fingerprint": protocol,
        "operator_pool_fingerprint": pool_hash,
        "target_bond_lengths_angstrom": list(target_bond_lengths),
        "tracking_bond_lengths_angstrom": list(tracking_bond_lengths),
        "tracking_method": TRACKING_METHOD,
        "all_orbital_mappings_reliable": all_orbitals_reliable,
        "variants_requested": [
            variant.name for variant in variants_to_run
        ],
        "num_geometry_variant_combinations_expected": (
            len(target_bond_lengths) * len(variants_to_run)
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
        "compact10_strict_success_at_every_target_geometry": compact_all,
        "causal_tests_by_geometry": causal_by_geometry,
        "variant_results": variant_results,
        "strict_full24_equivalence_tolerance_ha": STRICT_FULL24_TOLERANCE_HA,
        "thresholds_ha": {
            "chemical": diag.CHEMICAL_ACCURACY_HA,
            "good": diag.GOOD_ACCURACY_HA,
            "acceptable": diag.ACCEPTABLE_ACCURACY_HA,
        },
        "interpretation_rule": {
            "strict_support_success": (
                "Every analyzed restart is stable and chemically accurate, "
                "and the mean energy matches stable full24 within 1e-6 Ha."
            ),
            "pair_required_within_compact10": (
                "compact10 passes the strict test while the otherwise identical "
                "support with that spin-paired single channel removed does not."
            ),
            "negative_control_confirmed": (
                "D6_only is stable but fails chemical accuracy."
            ),
        },
        "checkpoint_key_policy": (
            "Every loaded geometry key is reconstructed from "
            "bond_length_angstrom at six-decimal precision; duplicate canonical "
            "run keys abort instead of being silently reused."
        ),
        "claim_boundary": (
            "This nested ablation tests the audited frozen-core LiH/STO-3G "
            "Hamiltonians with maximum-overlap orbital tracking, Jordan-Wigner "
            "mapping, one-repetition product-form selected UCC, statevector "
            "energies, and the declared local-optimization protocol. Agreement "
            "across restarts is strong numerical evidence but not a mathematical "
            "proof of global optimality or representability. It does not "
            "establish transfer to other basis sets, active spaces, molecules, "
            "ansatz families, or noisy hardware."
        ),
    }
    diag.write_json(
        output_dir / "phase5_causal_conclusions.json",
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
        "# LiH Phase 5A Compact-UCCSD Ablation Report",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        (
            "Compact10 strict success at every target geometry: "
            f"**{conclusions['compact10_strict_success_at_every_target_geometry']}**"
        ),
        "",
        (
            "Stable geometry–variant combinations: "
            f"**{conclusions['num_geometry_variant_combinations_stable']}/"
            f"{conclusions['num_geometry_variant_combinations_expected']}**"
        ),
        "",
        "## Geometry results",
        "",
        "| R (Å) | Full24 error (Ha) | Compact10 error (Ha) | "
        "|ΔE compact−full| (Ha) | Compact strict | D6-only error (Ha) |",
        "|---:|---:|---:|---:|:---:|---:|",
    ]
    for _, row in geometry_summary.iterrows():
        geometry = variant_summary[
            variant_summary["geometry_key"]
            == geometry_key(row["bond_length_angstrom"])
        ]
        compact = variant_row(geometry, "compact10")
        full = variant_row(geometry, "full24")
        d6 = variant_row(geometry, "D6_only")
        lines.append(
            "| "
            f"{float(row['bond_length_angstrom']):.3f} | "
            f"{fmt(None if full is None else full['mean_error_exact_ha'])} | "
            f"{fmt(None if compact is None else compact['mean_error_exact_ha'])} | "
            f"{fmt(None if compact is None else compact['absolute_energy_difference_vs_full24_ha'], 9)} | "
            f"{row['compact10_strict_success']} | "
            f"{fmt(None if d6 is None else d6['mean_error_exact_ha'])} |"
        )
    lines.extend(
        [
            "",
            "## Variant summary",
            "",
            "| Variant | Operators | Stable geometries | Chemical geometries | "
            "Strict geometries | Median CX reduction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        result = conclusions["variant_results"][variant.name]
        lines.append(
            f"| {variant.name} | {result['num_operators']} | "
            f"{result['num_geometries_stable']} | "
            f"{result['num_geometries_chemical_accuracy']} | "
            f"{result['num_geometries_strict_full24_equivalent_and_chemical']} | "
            f"{fmt(result['median_compiled_cx_reduction_fraction_vs_full24'], 3)} |"
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
    (output_dir / "phase5_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def audit_manifest(
    profile: str,
    systems: dict[str, diag.SystemData],
    target_bond_lengths: Sequence[float],
    tracking_bond_lengths: Sequence[float],
    orbital_summary: pd.DataFrame,
    mappings: dict[tuple[str, str], dict[str, Any]],
    pool_hash: str,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = {
        "audit_passed": bool(
            orbital_summary["tracking_reliable"].map(p2b.as_bool).all()
        ),
        "profile": profile,
        "target_bond_lengths_angstrom": list(target_bond_lengths),
        "tracking_bond_lengths_angstrom": list(tracking_bond_lengths),
        "configuration_fingerprints": {
            key: systems[key].fingerprint for key in sorted(systems)
        },
        "operator_pool_fingerprint": pool_hash,
        "reference_supports": {
            variant.name: [
                operator_label(index)
                for index in variant.reference_global_ids
            ]
            for variant in VARIANTS
        },
        "mapped_support_sizes": {
            f"{key}|{variant.name}": len(
                mappings[(key, variant.name)]["mapped_excitations"]
            )
            for key in sorted(systems)
            for variant in VARIANTS
        },
        "all_variants_degenerate_subspace_complete": not any(
            mappings[(key, variant.name)]["degenerate_subspace_sensitive"]
            for key in sorted(systems)
            for variant in VARIANTS
        ),
        "software_versions": diag.dependency_versions(),
    }
    diag.write_json(output_dir / "phase5_audit_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test the four-single plus six-double compact LiH UCCSD support "
            "with nested causal controls."
        )
    )
    parser.add_argument(
        "--profile",
        choices=["pilot", "full"],
        default="pilot",
        help=(
            "pilot: optimize R=2.5 and 3.0; full: optimize the seven Phase-3 "
            "geometries. Both profiles audit an anchored orbital-tracking grid."
        ),
    )
    parser.add_argument(
        "--bond-length",
        action="append",
        type=float,
        help=(
            "Override the profile target grid; repeat for multiple geometries. "
            "Intermediate orbital-tracking anchors are added automatically."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANT_BY_NAME),
        help="Run only this variant; repeat for multiple variants. Default: all seven.",
    )
    parser.add_argument("--basis", default="sto3g")
    parser.add_argument("--base-restarts", type=int, default=2)
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
    parser.add_argument("--parameter-seed", type=int, default=20260725)
    parser.add_argument("--audit-tolerance", type=float, default=1e-8)
    parser.add_argument("--orbital-overlap-threshold", type=float, default=0.75)
    parser.add_argument("--degeneracy-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--allow-low-orbital-overlap",
        action="store_true",
        help="Continue with flagged orbital mappings instead of aborting.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build and audit Hamiltonians, pools, and mappings, then stop before VQE.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: lih_phase5_pilot for pilot or lih_phase5_full for full."
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
    if args.base_restarts < 2:
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
    if not 0 < args.orbital_overlap_threshold <= 1:
        raise ValueError("The orbital-overlap threshold must be in (0, 1].")
    if args.degeneracy_tolerance <= 0:
        raise ValueError("The degeneracy tolerance must be positive.")

    targets, tracking_grid = resolved_grids(args.profile, args.bond_length)
    variants_to_run = selected_variants(args.variant)
    output_dir = args.output_dir or Path(
        "lih_phase5_pilot"
        if args.profile == "pilot"
        else "lih_phase5_full"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    systems: dict[str, diag.SystemData] = {}
    pools: dict[str, list[diag.Excitation]] = {}
    snapshots: dict[str, p3.OrbitalSnapshot] = {}
    audits: list[dict[str, Any]] = []
    print("Building and auditing Phase-5 Hamiltonians and orbitals...", flush=True)
    for bond_length in tracking_grid:
        key = geometry_key(bond_length)
        system = diag.build_system(build_config(bond_length, args.basis))
        _, _, combined = full_pool(system)
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
                "The equilibrium exact energy does not match Phases 1-4."
            )
        snapshot = p3.build_orbital_snapshot(
            bond_length,
            args.basis,
            EXPECTED_ACTIVE_ELECTRONS,
            EXPECTED_ACTIVE_SPATIAL_ORBITALS,
        )
        hf_difference = abs(
            snapshot.scf_total_energy_ha - system.hf_total_driver_ha
        )
        if hf_difference > args.audit_tolerance:
            raise RuntimeError(
                f"Direct PySCF and Qiskit HF energies differ by "
                f"{hf_difference:.3e} Ha at R={bond_length}."
            )
        systems[key] = system
        pools[key] = combined
        snapshots[key] = snapshot
        audits.append(audit)

    reference_groups, orbital_long, orbital_summary = p3.track_orbitals(
        snapshots,
        tracking_grid,
        args.orbital_overlap_threshold,
        args.degeneracy_tolerance,
    )
    orbital_long.to_csv(output_dir / "phase5_orbital_tracking.csv", index=False)
    orbital_summary.to_csv(
        output_dir / "phase5_orbital_mapping_summary.csv",
        index=False,
    )
    invalid_occupation = orbital_summary[
        (orbital_summary["occupied_reference_maps_to_canonical"] != 0)
        | ~orbital_summary["active_occupation_pattern_ok"].map(p2b.as_bool)
    ]
    if not invalid_occupation.empty:
        values = invalid_occupation["bond_length_angstrom"].astype(float).tolist()
        raise RuntimeError(
            "The occupied reference orbital is not consistently mapped at "
            f"bond lengths {values}; excitation transfer is undefined."
        )
    unreliable = orbital_summary[
        ~orbital_summary["tracking_reliable"].map(p2b.as_bool)
    ]
    if not unreliable.empty and not args.allow_low_orbital_overlap:
        values = unreliable["bond_length_angstrom"].astype(float).tolist()
        raise RuntimeError(
            "Orbital tracking failed the overlap audit at bond lengths "
            f"{values}. Inspect the audit or pass --allow-low-orbital-overlap."
        )

    mappings, mapping_frame = build_excitation_mappings(
        systems,
        pools,
        snapshots,
        tracking_grid,
        reference_groups,
    )
    mapping_frame.to_csv(
        output_dir / "phase5_excitation_mapping.csv",
        index=False,
    )
    reference_pool = pools[geometry_key(REFERENCE_BOND_LENGTH)]
    write_operator_pool(reference_pool, output_dir)
    reference_singles = reference_pool[:EXPECTED_SINGLE_COUNT]
    reference_doubles = reference_pool[EXPECTED_SINGLE_COUNT:]
    pool_hash = p4.pool_fingerprint(reference_singles, reference_doubles)
    diag.write_json(output_dir / "phase5_reference_audits.json", audits)
    manifest = audit_manifest(
        args.profile,
        systems,
        targets,
        tracking_grid,
        orbital_summary,
        mappings,
        pool_hash,
        output_dir,
    )
    if args.audit_only:
        print(
            json.dumps(
                {
                    "audit_passed": manifest["audit_passed"],
                    "target_bond_lengths_angstrom": targets,
                    "tracking_bond_lengths_angstrom": tracking_grid,
                    "variants": {
                        variant.name: [
                            operator_label(index)
                            for index in variant.reference_global_ids
                        ]
                        for variant in VARIANTS
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    protocol = make_protocol_fingerprint(
        systems,
        targets,
        tracking_grid,
        mappings,
        pool_hash,
        args.optimizer,
        args.maxiter,
        args.optimizer_tolerance,
        args.initial_scale,
        args.parameter_seed,
        args.base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
        args.orbital_overlap_threshold,
        args.degeneracy_tolerance,
    )
    runs = run_experiment(
        systems,
        mappings,
        targets,
        variants_to_run,
        output_dir,
        protocol,
        pool_hash,
        args.optimizer,
        args.maxiter,
        args.optimizer_tolerance,
        args.initial_scale,
        args.parameter_seed,
        args.base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
        args.resume,
    )
    variant_summary = summarize_runs(
        runs,
        systems,
        mappings,
        targets,
        variants_to_run,
        output_dir,
        args.base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
    )
    geometry_summary = make_geometry_summary(
        variant_summary,
        systems,
        targets,
        output_dir,
    )
    conclusions = make_conclusions(
        args.profile,
        protocol,
        pool_hash,
        targets,
        tracking_grid,
        variant_summary,
        geometry_summary,
        orbital_summary,
        variants_to_run,
        output_dir,
    )
    write_report(
        variant_summary,
        geometry_summary,
        conclusions,
        output_dir,
    )

    display_columns = [
        "bond_length_angstrom",
        "variant",
        "num_parameters",
        "mean_error_exact_ha",
        "robust_chemical_accuracy",
        "absolute_energy_difference_vs_full24_ha",
        "matches_full24_within_1e-6_ha",
        "compiled_depth",
        "compiled_cx",
    ]
    print("\nPhase 5A variant summary:")
    print(variant_summary[display_columns].to_string(index=False))
    print("\nPhase 5A causal conclusions:")
    print(json.dumps(conclusions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
