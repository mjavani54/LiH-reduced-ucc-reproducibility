#!/usr/bin/env python3
"""Phase 3: geometry transfer of the LiH excitation-support rule.

This program tests whether the operator supports discovered at the audited
LiH/STO-3G frozen-core geometry of 1.595 Angstrom transfer along a bond-length
scan. It compares five fixed supports with the full sixteen-double baseline:

* core3:       {3, 12, 15}
* minimal4:    {0, 3, 12, 15}
* compact5a:   {0, 3, 5, 12, 15}
* compact5b:   {3, 5, 10, 12, 15}
* discovered6: {0, 3, 5, 10, 12, 15}
* full16:      all sixteen opposite-spin doubles

Candidate IDs are not reused blindly. Canonical molecular orbitals are tracked
from the equilibrium reference in both bond-length directions using maximum
overlap. Degenerate orbital blocks are validated by their subspace singular
values. Reference excitations are then translated into the current canonical
orbital indices while their equilibrium operator order is preserved.

Every optimization is checkpointed. Two base restarts are used by default,
with three rescue restarts triggered only by optimizer failure, nonfinite
energy, variational violation, excessive restart disagreement, or disagreement
on chemical-accuracy classification.
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
from scipy.optimize import linear_sum_assignment

try:
    import lih_reference_failure_diagnostics as diag
    import lih_phase2b_exhaustive_support as p2b
except ImportError as exc:
    raise SystemExit(
        "Place lih_phase3_geometry_transfer.py, "
        "lih_phase2b_exhaustive_support.py, and "
        "lih_reference_failure_diagnostics.py in the same directory."
    ) from exc


REFERENCE_BOND_LENGTH = 1.595
EXPECTED_REFERENCE_EXACT_TOTAL_HA = -7.8821745057672805
DEFAULT_BOND_LENGTHS: tuple[float, ...] = (1.2, 1.4, 1.595, 1.8, 2.1, 2.5, 3.0)
EXPECTED_ACTIVE_SPATIAL_ORBITALS = 5
EXPECTED_ACTIVE_ELECTRONS = 2
TRACKING_METHOD = "sequential_maximum_overlap_anchored_at_reference"
OPERATOR_ORDERING = "ascending_reference_candidate_id"


@dataclass(frozen=True)
class Variant:
    name: str
    reference_ids: tuple[int, ...]
    scientific_role: str
    is_full16: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "core3",
        (3, 12, 15),
        "Negative control: the necessary core that narrowly misses chemical accuracy at equilibrium.",
    ),
    Variant(
        "minimal4",
        (0, 3, 12, 15),
        "Minimum-cardinality equilibrium chemical-accuracy support (Phase-2B mask 51).",
    ),
    Variant(
        "compact5a",
        (0, 3, 5, 12, 15),
        "CNOT-efficient five-operator equilibrium support (Phase-2B mask 55).",
    ),
    Variant(
        "compact5b",
        (3, 5, 10, 12, 15),
        "Five-operator support containing both numerically degenerate correction channels (mask 62).",
    ),
    Variant(
        "discovered6",
        (0, 3, 5, 10, 12, 15),
        "Complete empirically active support that matched full16 at equilibrium.",
    ),
    Variant(
        "full16",
        tuple(range(16)),
        "Full spin-preserving opposite-spin double-excitation baseline.",
        is_full16=True,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


@dataclass
class OrbitalSnapshot:
    bond_length: float
    molecule: Any
    active_coefficients: np.ndarray
    active_energies: np.ndarray
    active_occupations: np.ndarray
    scf_total_energy_ha: float
    num_frozen_core_orbitals: int
    reference_to_canonical: np.ndarray | None = None
    matched_individual_overlaps: np.ndarray | None = None
    group_quality_by_reference_orbital: np.ndarray | None = None
    tracking_reliable: bool = False
    occupied_reference_maps_to_canonical: int | None = None
    active_occupation_pattern_ok: bool = False


def geometry_key(bond_length: float) -> str:
    return f"{float(bond_length):.6f}"


def normalized_bond_lengths(values: Sequence[float] | None) -> list[float]:
    raw = list(DEFAULT_BOND_LENGTHS if not values else values)
    result = sorted({round(float(value), 9) for value in raw})
    if any(value <= 0 for value in result):
        raise ValueError("Every bond length must be positive.")
    if not any(abs(value - REFERENCE_BOND_LENGTH) <= 1e-9 for value in result):
        raise ValueError(
            f"The geometry grid must include the reference {REFERENCE_BOND_LENGTH} Angstrom."
        )
    keys = [geometry_key(value) for value in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Bond lengths must remain unique at six-decimal precision.")
    return result


def selected_variants(names: Sequence[str] | None) -> list[Variant]:
    if not names:
        return list(VARIANTS)
    return [VARIANT_BY_NAME[name] for name in dict.fromkeys(names)]


def expected_candidate_pool() -> list[diag.Excitation]:
    return [
        ((0, 5), (alpha_virtual, beta_virtual))
        for alpha_virtual in range(1, 5)
        for beta_virtual in range(6, 10)
    ]


def validate_candidate_pool(system: diag.SystemData) -> None:
    if list(system.candidate_excitations) != expected_candidate_pool():
        raise RuntimeError(
            "The generated excitation pool or ordering differs from the audited "
            "sixteen-candidate pool. Phase 3 cannot interpret candidate IDs safely."
        )
    if system.problem.num_spatial_orbitals != EXPECTED_ACTIVE_SPATIAL_ORBITALS:
        raise RuntimeError("Phase 3 requires exactly five active spatial orbitals.")
    if sum(system.problem.num_particles) != EXPECTED_ACTIVE_ELECTRONS:
        raise RuntimeError("Phase 3 requires exactly two active electrons.")


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


def build_orbital_snapshot(
    bond_length: float,
    basis: str,
    active_electrons: int,
    active_orbitals: int,
) -> OrbitalSnapshot:
    from pyscf import gto, scf

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
        raise RuntimeError(f"PySCF orbital-tracking SCF failed at R={bond_length} Angstrom.")

    frozen_core = (int(molecule.nelectron) - active_electrons) // 2
    if frozen_core < 0 or 2 * frozen_core + active_electrons != molecule.nelectron:
        raise RuntimeError("The inferred frozen-core electron count is inconsistent.")
    start = frozen_core
    stop = start + active_orbitals
    coefficients = np.asarray(mean_field.mo_coeff[:, start:stop], dtype=float)
    energies = np.asarray(mean_field.mo_energy[start:stop], dtype=float)
    occupations = np.asarray(mean_field.mo_occ[start:stop], dtype=float)
    if coefficients.shape[1] != active_orbitals:
        raise RuntimeError(
            f"Expected {active_orbitals} active orbitals but found {coefficients.shape[1]}."
        )
    return OrbitalSnapshot(
        bond_length=bond_length,
        molecule=molecule,
        active_coefficients=coefficients,
        active_energies=energies,
        active_occupations=occupations,
        scf_total_energy_ha=float(mean_field.e_tot),
        num_frozen_core_orbitals=frozen_core,
    )


def degeneracy_groups(energies: np.ndarray, tolerance_ha: float) -> list[tuple[int, ...]]:
    groups: list[list[int]] = [[0]]
    for index in range(1, len(energies)):
        if abs(float(energies[index] - energies[index - 1])) <= tolerance_ha:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [tuple(group) for group in groups]


def match_snapshot_to_previous(
    previous: OrbitalSnapshot,
    current: OrbitalSnapshot,
    reference_groups: Sequence[tuple[int, ...]],
    overlap_threshold: float,
) -> None:
    from pyscf import gto

    if previous.reference_to_canonical is None:
        raise RuntimeError("Previous orbital snapshot has no reference mapping.")
    previous_tracked = previous.active_coefficients[:, previous.reference_to_canonical]
    cross_overlap = gto.intor_cross(
        "int1e_ovlp",
        previous.molecule,
        current.molecule,
    )
    overlap_matrix = previous_tracked.T @ cross_overlap @ current.active_coefficients
    row_indices, column_indices = linear_sum_assignment(-np.abs(overlap_matrix))
    mapping = np.empty(len(row_indices), dtype=int)
    mapping[row_indices] = column_indices
    matched = np.abs(overlap_matrix[np.arange(len(mapping)), mapping])

    group_quality = np.empty(len(mapping), dtype=float)
    for group in reference_groups:
        rows = np.asarray(group, dtype=int)
        columns = mapping[rows]
        block = overlap_matrix[np.ix_(rows, columns)]
        singular_values = np.linalg.svd(block, compute_uv=False)
        quality = float(np.min(singular_values))
        group_quality[rows] = quality

    current.reference_to_canonical = mapping
    current.matched_individual_overlaps = matched
    current.group_quality_by_reference_orbital = group_quality
    current.occupied_reference_maps_to_canonical = int(mapping[0])
    occupation_ok = bool(
        np.isclose(current.active_occupations[mapping[0]], 2.0, atol=1e-8)
        and np.allclose(current.active_occupations[mapping[1:]], 0.0, atol=1e-8)
    )
    current.active_occupation_pattern_ok = occupation_ok
    current.tracking_reliable = bool(
        float(np.min(group_quality)) >= overlap_threshold
        and int(mapping[0]) == 0
        and occupation_ok
    )


def track_orbitals(
    snapshots: dict[str, OrbitalSnapshot],
    bond_lengths: Sequence[float],
    overlap_threshold: float,
    degeneracy_tolerance_ha: float,
) -> tuple[list[tuple[int, ...]], pd.DataFrame, pd.DataFrame]:
    reference_key = geometry_key(REFERENCE_BOND_LENGTH)
    reference = snapshots[reference_key]
    count = reference.active_coefficients.shape[1]
    reference.reference_to_canonical = np.arange(count, dtype=int)
    reference.matched_individual_overlaps = np.ones(count, dtype=float)
    reference.group_quality_by_reference_orbital = np.ones(count, dtype=float)
    reference.occupied_reference_maps_to_canonical = 0
    reference.active_occupation_pattern_ok = bool(
        np.isclose(reference.active_occupations[0], 2.0, atol=1e-8)
        and np.allclose(reference.active_occupations[1:], 0.0, atol=1e-8)
    )
    reference.tracking_reliable = reference.active_occupation_pattern_ok
    reference_groups = degeneracy_groups(reference.active_energies, degeneracy_tolerance_ha)

    lower = sorted(
        [value for value in bond_lengths if value < REFERENCE_BOND_LENGTH],
        reverse=True,
    )
    upper = sorted([value for value in bond_lengths if value > REFERENCE_BOND_LENGTH])
    for direction in (lower, upper):
        previous = reference
        for bond_length in direction:
            current = snapshots[geometry_key(bond_length)]
            match_snapshot_to_previous(
                previous,
                current,
                reference_groups,
                overlap_threshold,
            )
            previous = current

    group_by_orbital: dict[int, tuple[int, tuple[int, ...]]] = {}
    for group_id, group in enumerate(reference_groups):
        for orbital in group:
            group_by_orbital[orbital] = (group_id, group)

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for geometry_index, bond_length in enumerate(sorted(bond_lengths)):
        snapshot = snapshots[geometry_key(bond_length)]
        assert snapshot.reference_to_canonical is not None
        assert snapshot.matched_individual_overlaps is not None
        assert snapshot.group_quality_by_reference_orbital is not None
        mapping = snapshot.reference_to_canonical
        for reference_orbital in range(count):
            group_id, group = group_by_orbital[reference_orbital]
            canonical = int(mapping[reference_orbital])
            long_rows.append(
                {
                    "geometry_index": geometry_index,
                    "bond_length_angstrom": bond_length,
                    "reference_active_orbital": reference_orbital,
                    "current_canonical_active_orbital": canonical,
                    "reference_orbital_energy_ha": float(
                        reference.active_energies[reference_orbital]
                    ),
                    "current_canonical_orbital_energy_ha": float(
                        snapshot.active_energies[canonical]
                    ),
                    "current_canonical_occupation": float(
                        snapshot.active_occupations[canonical]
                    ),
                    "matched_individual_abs_overlap": float(
                        snapshot.matched_individual_overlaps[reference_orbital]
                    ),
                    "matched_degenerate_group_min_singular_value": float(
                        snapshot.group_quality_by_reference_orbital[reference_orbital]
                    ),
                    "reference_degeneracy_group": group_id,
                    "reference_degeneracy_group_json": json.dumps(list(group)),
                    "individual_identity_gauge_ambiguous": len(group) > 1,
                    "geometry_tracking_reliable": snapshot.tracking_reliable,
                }
            )
        summary_rows.append(
            {
                "geometry_index": geometry_index,
                "bond_length_angstrom": bond_length,
                "reference_to_canonical_mapping_json": json.dumps(mapping.tolist()),
                "minimum_individual_abs_overlap": float(
                    np.min(snapshot.matched_individual_overlaps)
                ),
                "minimum_degenerate_group_singular_value": float(
                    np.min(snapshot.group_quality_by_reference_orbital)
                ),
                "occupied_reference_maps_to_canonical": int(mapping[0]),
                "active_occupation_pattern_ok": snapshot.active_occupation_pattern_ok,
                "tracking_reliable": snapshot.tracking_reliable,
                "direct_pyscf_hf_total_energy_ha": snapshot.scf_total_energy_ha,
                "num_frozen_core_orbitals": snapshot.num_frozen_core_orbitals,
            }
        )
    return reference_groups, pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def map_spin_orbital(index: int, mapping: np.ndarray, num_spatial_orbitals: int) -> int:
    spin_block, spatial = divmod(int(index), num_spatial_orbitals)
    if spin_block not in {0, 1}:
        raise ValueError(f"Spin-orbital index {index} is outside the expected two blocks.")
    return int(mapping[spatial] + spin_block * num_spatial_orbitals)


def translate_excitation(
    excitation: diag.Excitation,
    mapping: np.ndarray,
    num_spatial_orbitals: int,
) -> diag.Excitation:
    occupied, virtual = excitation
    return (
        tuple(map_spin_orbital(index, mapping, num_spatial_orbitals) for index in occupied),
        tuple(map_spin_orbital(index, mapping, num_spatial_orbitals) for index in virtual),
    )


def variant_degenerate_subspace_sensitive(
    variant: Variant,
    reference_pool: Sequence[diag.Excitation],
    reference_groups: Sequence[tuple[int, ...]],
    num_spatial_orbitals: int,
) -> bool:
    if variant.is_full16:
        return False
    used_virtual_spatial: set[int] = set()
    for candidate_id in variant.reference_ids:
        for spin_orbital in reference_pool[candidate_id][1]:
            used_virtual_spatial.add(int(spin_orbital) % num_spatial_orbitals)
    for group in reference_groups:
        group_set = set(group)
        intersection = used_virtual_spatial & group_set
        if len(group) > 1 and intersection and intersection != group_set:
            return True
    return False


def build_excitation_mappings(
    systems: dict[str, diag.SystemData],
    snapshots: dict[str, OrbitalSnapshot],
    bond_lengths: Sequence[float],
    reference_groups: Sequence[tuple[int, ...]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    reference_system = systems[geometry_key(REFERENCE_BOND_LENGTH)]
    reference_pool = reference_system.candidate_excitations
    result: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for geometry_index, bond_length in enumerate(sorted(bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        snapshot = snapshots[key]
        assert snapshot.reference_to_canonical is not None
        candidate_lookup = {
            excitation: candidate_id
            for candidate_id, excitation in enumerate(system.candidate_excitations)
        }
        for variant in VARIANTS:
            if variant.is_full16:
                reference_ids = list(range(len(reference_pool)))
                mapped_excitations = list(system.candidate_excitations)
                mapped_ids = list(range(len(system.candidate_excitations)))
                ordering = "ascending_current_candidate_id"
            else:
                reference_ids = list(variant.reference_ids)
                mapped_excitations = [
                    translate_excitation(
                        reference_pool[candidate_id],
                        snapshot.reference_to_canonical,
                        system.problem.num_spatial_orbitals,
                    )
                    for candidate_id in reference_ids
                ]
                try:
                    mapped_ids = [candidate_lookup[item] for item in mapped_excitations]
                except KeyError as exc:
                    raise RuntimeError(
                        f"A mapped excitation for {variant.name} at R={bond_length} "
                        "is absent from the current candidate pool."
                    ) from exc
                ordering = OPERATOR_ORDERING
            sensitive = variant_degenerate_subspace_sensitive(
                variant,
                reference_pool,
                reference_groups,
                system.problem.num_spatial_orbitals,
            )
            record = {
                "geometry_index": geometry_index,
                "bond_length_angstrom": bond_length,
                "geometry_key": key,
                "variant": variant.name,
                "reference_ids": reference_ids,
                "reference_ids_json": json.dumps(reference_ids),
                "mapped_current_ids": mapped_ids,
                "mapped_current_ids_json": json.dumps(mapped_ids),
                "mapped_excitations": mapped_excitations,
                "mapped_excitations_json": json.dumps(
                    [diag.excitation_to_json(item) for item in mapped_excitations]
                ),
                "operator_ordering": ordering,
                "degenerate_subspace_sensitive": sensitive,
                "orbital_mapping_identity": bool(
                    np.array_equal(
                        snapshot.reference_to_canonical,
                        np.arange(system.problem.num_spatial_orbitals),
                    )
                ),
            }
            result[(key, variant.name)] = record
            rows.append(
                {
                    column: value
                    for column, value in record.items()
                    if column not in {"reference_ids", "mapped_current_ids", "mapped_excitations"}
                }
            )
    return result, pd.DataFrame(rows)


def protocol_fingerprint(
    systems: dict[str, diag.SystemData],
    bond_lengths: Sequence[float],
    excitation_mappings: dict[tuple[str, str], dict[str, Any]],
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
        "bond_lengths_angstrom": list(bond_lengths),
        "configuration_fingerprints": {
            key: systems[key].fingerprint for key in sorted(systems)
        },
        "variants": [
            {"name": variant.name, "reference_ids": list(variant.reference_ids)}
            for variant in VARIANTS
        ],
        "mapped_current_ids": {
            f"{key}|{variant.name}": excitation_mappings[(key, variant.name)][
                "mapped_current_ids"
            ]
            for key in sorted(systems)
            for variant in VARIANTS
        },
        "tracking_method": TRACKING_METHOD,
        "operator_ordering": OPERATOR_ORDERING,
        "orbital_overlap_threshold": orbital_overlap_threshold,
        "degeneracy_tolerance_ha": degeneracy_tolerance_ha,
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


def common_run_fields(
    system: diag.SystemData,
    mapping_record: dict[str, Any],
    variant: Variant,
    protocol: str,
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
        "geometry_index": mapping_record["geometry_index"],
        "bond_length_angstrom": mapping_record["bond_length_angstrom"],
        "geometry_key": mapping_record["geometry_key"],
        "variant": variant.name,
        "scientific_role": variant.scientific_role,
        "reference_ids_json": mapping_record["reference_ids_json"],
        "mapped_current_ids_json": mapping_record["mapped_current_ids_json"],
        "mapped_excitations_json": mapping_record["mapped_excitations_json"],
        "operator_ordering": mapping_record["operator_ordering"],
        "degenerate_subspace_sensitive": mapping_record[
            "degenerate_subspace_sensitive"
        ],
        "orbital_mapping_identity": mapping_record["orbital_mapping_identity"],
        "selected_count": len(mapping_record["mapped_excitations"]),
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
    bond_lengths: Sequence[float],
    variants_to_run: Sequence[Variant],
    excitation_mappings: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    protocol: str,
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

    runs_path = output_dir / "phase3_restart_runs.csv"
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, int]] = set()
    if resume and runs_path.exists():
        previous = pd.read_csv(runs_path)
        rows = previous.to_dict(orient="records")
        matching = previous[previous["protocol_fingerprint"].astype(str) == protocol]
        completed = {
            (str(row["geometry_key"]), str(row["variant"]), int(row["restart"]))
            for row in matching.to_dict(orient="records")
        }
        print(f"Resuming with {len(completed)} completed Phase-3 attempts.")

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
        for variant in variants_to_run:
            variant_index = list(VARIANT_BY_NAME).index(variant.name)
            existing = current_runs(key, variant.name)
            if not existing.empty:
                existing_base = existing[
                    pd.to_numeric(existing["restart"], errors="coerce") < base_restarts
                ].sort_values("restart")
                if len(existing_base) == base_restarts:
                    reasons = p2b.rescue_reasons(
                        existing_base,
                        base_restarts,
                        disagreement_tolerance_ha,
                    )
                    expected = base_restarts + (rescue_restarts if reasons else 0)
                    if all(
                        (key, variant.name, restart) in completed
                        for restart in range(expected)
                    ):
                        continue

            mapping_record = excitation_mappings[(key, variant.name)]
            ansatz = diag.build_selected_ansatz(
                system,
                mapping_record["mapped_excitations"],
            )
            resource_seed = parameter_seed + 10_000 * geometry_index + variant_index
            resources = diag.circuit_resource_metrics(ansatz, resource_seed)
            common = common_run_fields(
                system,
                mapping_record,
                variant,
                protocol,
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
                    f"R={bond_length:.3f} variant={variant.name} "
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
                        f"ERROR at R={bond_length}, {variant.name}, restart {restart}: {exc}",
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
                pd.to_numeric(base_frame["restart"], errors="coerce") < base_restarts
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
                for restart in range(base_restarts, base_restarts + rescue_restarts):
                    execute_restart(restart, "rescue", reasons)

    all_runs = pd.DataFrame(rows)
    return all_runs[all_runs["protocol_fingerprint"].astype(str) == protocol].copy()


def summarize_runs(
    runs: pd.DataFrame,
    systems: dict[str, diag.SystemData],
    bond_lengths: Sequence[float],
    excitation_mappings: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    base_restarts: int,
    rescue_restarts: int,
    disagreement_tolerance_ha: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for geometry_index, bond_length in enumerate(sorted(bond_lengths)):
        key = geometry_key(bond_length)
        system = systems[key]
        for variant in VARIANTS:
            mapping_record = excitation_mappings[(key, variant.name)]
            group = runs[
                (runs["geometry_key"].astype(str) == key)
                & (runs["variant"].astype(str) == variant.name)
            ].copy()
            group = group.sort_values("restart") if not group.empty else group
            if group.empty:
                rows.append(
                    {
                        "geometry_index": geometry_index,
                        "bond_length_angstrom": bond_length,
                        "geometry_key": key,
                        "configuration_fingerprint": system.fingerprint,
                        "variant": variant.name,
                        "scientific_role": variant.scientific_role,
                        "reference_ids_json": mapping_record["reference_ids_json"],
                        "mapped_current_ids_json": mapping_record[
                            "mapped_current_ids_json"
                        ],
                        "selected_count": len(mapping_record["mapped_excitations"]),
                        "degenerate_subspace_sensitive": mapping_record[
                            "degenerate_subspace_sensitive"
                        ],
                        "analysis_complete": False,
                        "analysis_stable": False,
                        "classification": "not_run",
                        "completed_attempts": 0,
                        "exact_total_energy_ha": system.exact_total_ha,
                        "hf_total_energy_ha": system.hf_total_driver_ha,
                        "hf_error_ha": system.hf_total_driver_ha
                        - system.exact_total_ha,
                    }
                )
                continue

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
            rescue_reason_values = group["rescue_trigger_reasons"].dropna().astype(str)
            rows.append(
                {
                    "geometry_index": geometry_index,
                    "bond_length_angstrom": bond_length,
                    "geometry_key": key,
                    "configuration_fingerprint": system.fingerprint,
                    "variant": variant.name,
                    "scientific_role": variant.scientific_role,
                    "reference_ids_json": mapping_record["reference_ids_json"],
                    "mapped_current_ids_json": mapping_record[
                        "mapped_current_ids_json"
                    ],
                    "mapped_excitations_json": mapping_record[
                        "mapped_excitations_json"
                    ],
                    "operator_ordering": mapping_record["operator_ordering"],
                    "selected_count": len(mapping_record["mapped_excitations"]),
                    "degenerate_subspace_sensitive": mapping_record[
                        "degenerate_subspace_sensitive"
                    ],
                    "orbital_mapping_identity": mapping_record[
                        "orbital_mapping_identity"
                    ],
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
                    "hf_error_ha": system.hf_total_driver_ha
                    - system.exact_total_ha,
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
                        float((errors <= diag.CHEMICAL_ACCURACY_HA).mean())
                        if not errors.empty
                        else np.nan
                    ),
                    "robust_chemical_accuracy": robust_chemical,
                    "optimizer_success_rate": (
                        float(analysis_runs["optimizer_success"].map(p2b.as_bool).mean())
                        if not analysis_runs.empty
                        else np.nan
                    ),
                    "median_iterations": (
                        float(
                            pd.to_numeric(
                                valid["num_iterations"], errors="coerce"
                            ).median()
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "median_elapsed_seconds": (
                        float(
                            pd.to_numeric(
                                valid["elapsed_seconds"], errors="coerce"
                            ).median()
                        )
                        if not valid.empty
                        else np.nan
                    ),
                    "total_elapsed_seconds_all_attempts": float(
                        pd.to_numeric(
                            group["elapsed_seconds"], errors="coerce"
                        ).fillna(0).sum()
                    ),
                    "logical_depth": first.get("logical_depth"),
                    "compiled_depth": first.get("compiled_depth"),
                    "compiled_cx": first.get("compiled_cx"),
                    "num_parameters": int(first["num_parameters"]),
                    "any_variational_violation": bool(
                        group["variational_violation"].map(p2b.as_bool).any()
                    ),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["geometry_index", "variant"])
    summary = add_full16_comparisons(summary)
    summary.to_csv(output_dir / "phase3_variant_geometry_summary.csv", index=False)
    return summary


def add_full16_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    numeric_comparison_columns = [
        "full16_mean_total_energy_ha",
        "full16_mean_error_exact_ha",
        "signed_energy_difference_vs_full16_ha",
        "absolute_energy_difference_vs_full16_ha",
        "compiled_depth_reduction_vs_full16",
        "compiled_cx_reduction_vs_full16",
        "num_parameters_reduction_vs_full16",
    ]
    for column in numeric_comparison_columns:
        result[column] = np.nan
    result["full16_robust_chemical_accuracy"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result["matches_full16_within_1e-6_ha"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result["selected_energy_below_full16_flag"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    result["transfer_classification"] = pd.Series(
        None, index=result.index, dtype="object"
    )

    for key, indices in result.groupby("geometry_key").groups.items():
        geometry = result.loc[indices]
        full_rows = geometry[geometry["variant"] == "full16"]
        if full_rows.empty or not p2b.as_bool(full_rows.iloc[0]["analysis_stable"]):
            result.loc[indices, "transfer_classification"] = "full16_unresolved"
            continue
        full = full_rows.iloc[0]
        full_energy = float(full["mean_total_energy_ha"])
        full_error = float(full["mean_error_exact_ha"])
        full_chemical = p2b.as_bool(full["robust_chemical_accuracy"])
        for index in indices:
            row = result.loc[index]
            result.at[index, "full16_mean_total_energy_ha"] = full_energy
            result.at[index, "full16_mean_error_exact_ha"] = full_error
            result.at[index, "full16_robust_chemical_accuracy"] = full_chemical
            if not p2b.as_bool(row["analysis_stable"]):
                result.at[index, "transfer_classification"] = "variant_unresolved"
                continue
            difference = float(row["mean_total_energy_ha"]) - full_energy
            absolute_difference = abs(difference)
            strict_match = absolute_difference <= 1e-6
            result.at[index, "signed_energy_difference_vs_full16_ha"] = difference
            result.at[index, "absolute_energy_difference_vs_full16_ha"] = absolute_difference
            result.at[index, "matches_full16_within_1e-6_ha"] = strict_match
            result.at[index, "selected_energy_below_full16_flag"] = bool(
                row["variant"] != "full16" and difference < -1e-8
            )
            for resource in ("compiled_depth", "compiled_cx", "num_parameters"):
                denominator = full.get(resource)
                value = row.get(resource)
                output_column = f"{resource}_reduction_vs_full16"
                if pd.notna(denominator) and float(denominator) != 0 and pd.notna(value):
                    result.at[index, output_column] = (
                        float(denominator) - float(value)
                    ) / float(denominator)

            if row["variant"] == "full16":
                classification = (
                    "full16_chemical_accuracy"
                    if full_chemical
                    else "full16_parent_ansatz_failure"
                )
            else:
                variant_chemical = p2b.as_bool(row["robust_chemical_accuracy"])
                if variant_chemical and strict_match:
                    classification = "strict_transfer_success"
                elif variant_chemical:
                    classification = "chemical_without_strict_full16_equivalence"
                elif full_chemical:
                    classification = "support_transfer_failure"
                elif strict_match:
                    classification = "parent_ansatz_class_failure"
                else:
                    classification = "combined_parent_and_compression_failure"
            result.at[index, "transfer_classification"] = classification
    return result


def make_geometry_summary(
    variant_summary: pd.DataFrame,
    orbital_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    orbital_by_key = {
        geometry_key(row["bond_length_angstrom"]): row
        for _, row in orbital_summary.iterrows()
    }
    for key, group in variant_summary.groupby("geometry_key", sort=False):
        full = group[group["variant"] == "full16"].iloc[0]
        selected = group[group["variant"] != "full16"]
        orbital = orbital_by_key[str(key)]
        stable_selected = selected[selected["analysis_stable"].map(p2b.as_bool)]
        best_selected = (
            stable_selected.sort_values("mean_error_exact_ha").iloc[0]
            if not stable_selected.empty
            else None
        )
        rows.append(
            {
                "geometry_index": int(group["geometry_index"].iloc[0]),
                "bond_length_angstrom": float(group["bond_length_angstrom"].iloc[0]),
                "configuration_fingerprint": group[
                    "configuration_fingerprint"
                ].iloc[0],
                "exact_total_energy_ha": float(group["exact_total_energy_ha"].iloc[0]),
                "hf_total_energy_ha": float(group["hf_total_energy_ha"].iloc[0]),
                "hf_error_ha": float(group["hf_error_ha"].iloc[0]),
                "full16_mean_error_exact_ha": full.get("mean_error_exact_ha"),
                "full16_robust_chemical_accuracy": full.get(
                    "robust_chemical_accuracy"
                ),
                "num_selected_variants_chemical": int(
                    stable_selected["robust_chemical_accuracy"].map(p2b.as_bool).sum()
                ),
                "num_selected_variants_strict_full16_match": int(
                    stable_selected["matches_full16_within_1e-6_ha"].map(
                        p2b.as_bool
                    ).sum()
                ),
                "best_selected_variant": (
                    None if best_selected is None else best_selected["variant"]
                ),
                "best_selected_error_exact_ha": (
                    np.nan
                    if best_selected is None
                    else float(best_selected["mean_error_exact_ha"])
                ),
                "minimum_individual_abs_overlap": orbital[
                    "minimum_individual_abs_overlap"
                ],
                "minimum_degenerate_group_singular_value": orbital[
                    "minimum_degenerate_group_singular_value"
                ],
                "orbital_tracking_reliable": orbital["tracking_reliable"],
                "reference_to_canonical_mapping_json": orbital[
                    "reference_to_canonical_mapping_json"
                ],
            }
        )
    result = pd.DataFrame(rows).sort_values("geometry_index")
    result.to_csv(output_dir / "phase3_geometry_summary.csv", index=False)
    return result


def nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def make_conclusions(
    protocol: str,
    variant_summary: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    orbital_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    actual_bond_lengths = geometry_summary["bond_length_angstrom"].astype(float).tolist()
    expected_rows = len(actual_bond_lengths) * len(VARIANTS)
    all_requested_geometries = set(
        geometry_key(value) for value in actual_bond_lengths
    ) == set(variant_summary["geometry_key"].astype(str))
    optimization_complete = bool(
        len(variant_summary) == expected_rows
        and variant_summary["analysis_complete"].map(p2b.as_bool).all()
        and variant_summary["analysis_stable"].map(p2b.as_bool).all()
    )
    orbital_reliable = bool(orbital_summary["tracking_reliable"].map(p2b.as_bool).all())
    analysis_complete = all_requested_geometries and optimization_complete and orbital_reliable

    variant_results: dict[str, Any] = {}
    for variant in VARIANTS:
        group = variant_summary[variant_summary["variant"] == variant.name].sort_values(
            "geometry_index"
        )
        stable = group[group["analysis_stable"].map(p2b.as_bool)]
        chemical_mask = stable["robust_chemical_accuracy"].map(p2b.as_bool).astype(bool)
        strict_mask = stable["matches_full16_within_1e-6_ha"].map(
            p2b.as_bool
        ).astype(bool)
        chemical_geometries = (
            stable.loc[chemical_mask, "bond_length_angstrom"].astype(float).tolist()
        )
        strict_geometries = (
            stable.loc[strict_mask, "bond_length_angstrom"].astype(float).tolist()
        )
        variant_results[variant.name] = {
            "reference_ids": list(variant.reference_ids),
            "degenerate_subspace_sensitive": bool(
                group["degenerate_subspace_sensitive"].map(p2b.as_bool).any()
            ),
            "num_geometries_stable": int(len(stable)),
            "num_geometries_chemical_accuracy": len(chemical_geometries),
            "chemical_accuracy_bond_lengths_angstrom": chemical_geometries,
            "num_geometries_matching_full16_within_1e-6_ha": len(strict_geometries),
            "full16_match_bond_lengths_angstrom": strict_geometries,
            "all_geometries_chemical_accuracy": (
                bool(len(chemical_geometries) == len(actual_bond_lengths))
                if analysis_complete
                else None
            ),
            "all_geometries_match_full16_within_1e-6_ha": (
                bool(len(strict_geometries) == len(actual_bond_lengths))
                if analysis_complete
                else None
            ),
            "worst_mean_error_exact_ha": nullable_float(
                pd.to_numeric(stable["mean_error_exact_ha"], errors="coerce").max()
            ),
            "worst_absolute_energy_difference_vs_full16_ha": nullable_float(
                pd.to_numeric(
                    stable["absolute_energy_difference_vs_full16_ha"],
                    errors="coerce",
                ).max()
            ),
            "classification_counts": {
                str(key): int(value)
                for key, value in stable["transfer_classification"]
                .value_counts()
                .to_dict()
                .items()
            },
            "num_geometries_selected_energy_below_full16": int(
                stable["selected_energy_below_full16_flag"].map(p2b.as_bool).sum()
            ),
        }

    conclusions = {
        "analysis_status": "complete" if analysis_complete else "partial_or_unreliable",
        "protocol_fingerprint": protocol,
        "reference_bond_length_angstrom": REFERENCE_BOND_LENGTH,
        "bond_lengths_angstrom": geometry_summary["bond_length_angstrom"]
        .astype(float)
        .tolist(),
        "num_geometries": int(len(geometry_summary)),
        "num_geometry_variant_combinations_expected": expected_rows,
        "num_geometry_variant_combinations_stable": int(
            variant_summary["analysis_stable"].map(p2b.as_bool).sum()
        ),
        "num_combinations_requiring_rescue": int(
            variant_summary["rescue_triggered"].map(p2b.as_bool).sum()
        ),
        "all_orbital_mappings_reliable": orbital_reliable,
        "tracking_method": TRACKING_METHOD,
        "operator_ordering": OPERATOR_ORDERING,
        "variant_transfer_results": variant_results,
        "interpretation_rule": {
            "strict_transfer_success": (
                "The fixed support reaches chemical accuracy and matches full16 "
                "within 1e-6 Ha."
            ),
            "chemical_without_strict_full16_equivalence": (
                "The fixed support reaches chemical accuracy but does not reproduce "
                "full16 within 1e-6 Ha."
            ),
            "support_transfer_failure": (
                "The fixed support fails chemical accuracy while full16 succeeds."
            ),
            "parent_ansatz_class_failure": (
                "The fixed support matches full16, but both fail chemical accuracy."
            ),
            "combined_parent_and_compression_failure": (
                "Both the doubles-only parent ansatz and the compressed support are "
                "insufficient at this geometry."
            ),
        },
        "claim_boundary": (
            "This experiment tests fixed equilibrium supports only for frozen-core "
            "LiH/STO-3G over the declared bond-length grid, with one-repetition "
            "selected UCC, Jordan-Wigner mapping, statevector energies, maximum-overlap "
            "orbital tracking, and the declared local-optimization protocol. It does "
            "not establish transfer to other basis sets, active spaces, molecules, "
            "ansatz families, noisy hardware, or independently rotated gauges inside "
            "a degenerate orbital subspace."
        ),
    }
    diag.write_json(output_dir / "phase3_transfer_conclusions.json", conclusions)
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
        "# LiH Phase 3 Geometry-Transfer Report",
        "",
        f"Analysis status: **{conclusions['analysis_status']}**",
        "",
        f"Stable geometry-variant combinations: **{conclusions['num_geometry_variant_combinations_stable']}/{conclusions['num_geometry_variant_combinations_expected']}**",
        "",
        f"Rescue-triggered combinations: **{conclusions['num_combinations_requiring_rescue']}**",
        "",
        f"All orbital mappings reliable: **{conclusions['all_orbital_mappings_reliable']}**",
        "",
        "## Geometry-level diagnosis",
        "",
        "| R (A) | HF error | full16 error | full16 chemical | Best selected variant | Best selected error | Orbital overlap quality |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    for _, row in geometry_summary.iterrows():
        lines.append(
            f"| {fmt(row['bond_length_angstrom'], 3)} | {fmt(row['hf_error_ha'], 6)} | "
            f"{fmt(row['full16_mean_error_exact_ha'], 6)} | "
            f"{row['full16_robust_chemical_accuracy']} | {row['best_selected_variant']} | "
            f"{fmt(row['best_selected_error_exact_ha'], 6)} | "
            f"{fmt(row['minimum_degenerate_group_singular_value'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Variant transfer results",
            "",
            "| Variant | R (A) | Error vs exact | Difference vs full16 | Chemical | Strict full16 match | Classification |",
            "|---|---:|---:|---:|---|---|---|",
        ]
    )
    for _, row in variant_summary.iterrows():
        lines.append(
            f"| {row['variant']} | {fmt(row['bond_length_angstrom'], 3)} | "
            f"{fmt(row.get('mean_error_exact_ha'), 6)} | "
            f"{fmt(row.get('absolute_energy_difference_vs_full16_ha'), 6)} | "
            f"{row.get('robust_chemical_accuracy')} | "
            f"{row.get('matches_full16_within_1e-6_ha')} | "
            f"{row.get('transfer_classification')} |"
        )

    lines.extend(["", "## Claim boundary", "", conclusions["claim_boundary"], ""])
    (output_dir / "phase3_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test fixed equilibrium LiH operator supports across a bond-length grid "
            "with audited orbital-identity tracking."
        )
    )
    parser.add_argument(
        "--bond-length",
        action="append",
        type=float,
        help=(
            "Bond length in Angstrom; repeat for a custom grid. The reference 1.595 "
            "must be included. Default: 1.2, 1.4, 1.595, 1.8, 2.1, 2.5, 3.0."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANT_BY_NAME),
        help="Run only this variant; repeat for multiple variants. Default: all six.",
    )
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
        help="Build and audit all Hamiltonians and orbital mappings, then stop before VQE.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("lih_phase3_output"))
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
    if args.disagreement_tolerance <= 0 or args.optimizer_tolerance <= 0:
        raise ValueError("Optimization tolerances must be positive.")
    if not 0 < args.orbital_overlap_threshold <= 1:
        raise ValueError("The orbital-overlap threshold must be in (0, 1].")
    if args.degeneracy_tolerance <= 0:
        raise ValueError("The degeneracy tolerance must be positive.")
    bond_lengths = normalized_bond_lengths(args.bond_length)
    variants_to_run = selected_variants(args.variant)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    systems: dict[str, diag.SystemData] = {}
    audits: list[dict[str, Any]] = []
    snapshots: dict[str, OrbitalSnapshot] = {}
    print("Building and auditing all Phase-3 Hamiltonians...", flush=True)
    for bond_length in bond_lengths:
        key = geometry_key(bond_length)
        config = build_config(bond_length, args.basis)
        system = diag.build_system(config)
        validate_candidate_pool(system)
        reported = (
            [EXPECTED_REFERENCE_EXACT_TOTAL_HA]
            if abs(bond_length - REFERENCE_BOND_LENGTH) <= 1e-9
            else []
        )
        audit = diag.validate_reference(system, reported, args.audit_tolerance)
        if not audit["audit_passed"]:
            raise RuntimeError(f"The Hamiltonian audit failed at R={bond_length}.")
        if reported and abs(system.exact_total_ha - reported[0]) > args.audit_tolerance:
            raise RuntimeError("The equilibrium exact energy does not match Phase 1/2.")
        systems[key] = system
        audits.append(audit)
        snapshots[key] = build_orbital_snapshot(
            bond_length,
            args.basis,
            EXPECTED_ACTIVE_ELECTRONS,
            EXPECTED_ACTIVE_SPATIAL_ORBITALS,
        )
        hf_difference = abs(
            snapshots[key].scf_total_energy_ha - system.hf_total_driver_ha
        )
        if hf_difference > args.audit_tolerance:
            raise RuntimeError(
                f"Direct PySCF and Qiskit HF energies differ by {hf_difference:.3e} Ha "
                f"at R={bond_length}; orbital identity cannot be trusted."
            )

    reference_groups, orbital_long, orbital_summary = track_orbitals(
        snapshots,
        bond_lengths,
        args.orbital_overlap_threshold,
        args.degeneracy_tolerance,
    )
    orbital_long.to_csv(args.output_dir / "phase3_orbital_tracking.csv", index=False)
    orbital_summary.to_csv(
        args.output_dir / "phase3_orbital_mapping_summary.csv", index=False
    )
    unreliable = orbital_summary[
        ~orbital_summary["tracking_reliable"].map(p2b.as_bool)
    ]
    invalid_occupation = orbital_summary[
        (orbital_summary["occupied_reference_maps_to_canonical"] != 0)
        | ~orbital_summary["active_occupation_pattern_ok"].map(p2b.as_bool)
    ]
    if not invalid_occupation.empty:
        values = invalid_occupation["bond_length_angstrom"].astype(float).tolist()
        raise RuntimeError(
            "The tracked reference occupied orbital does not remain the canonical "
            f"active occupied orbital at bond lengths {values}. Fixed excitation "
            "transfer from the equilibrium Hartree--Fock reference is undefined there."
        )
    if not unreliable.empty and not args.allow_low_orbital_overlap:
        values = unreliable["bond_length_angstrom"].astype(float).tolist()
        raise RuntimeError(
            "Orbital tracking was unreliable at bond lengths "
            f"{values}. Inspect phase3_orbital_tracking.csv. To continue only after "
            "review, rerun with --allow-low-orbital-overlap."
        )

    excitation_mappings, excitation_mapping_table = build_excitation_mappings(
        systems,
        snapshots,
        bond_lengths,
        reference_groups,
    )
    excitation_mapping_table.to_csv(
        args.output_dir / "phase3_excitation_mapping.csv", index=False
    )
    audit_bundle = {
        "analysis_phase": "Phase 3 geometry transfer",
        "reference_bond_length_angstrom": REFERENCE_BOND_LENGTH,
        "bond_lengths_angstrom": bond_lengths,
        "tracking_method": TRACKING_METHOD,
        "orbital_overlap_threshold": args.orbital_overlap_threshold,
        "degeneracy_tolerance_ha": args.degeneracy_tolerance,
        "reference_degeneracy_groups": [list(group) for group in reference_groups],
        "all_orbital_mappings_reliable": bool(
            orbital_summary["tracking_reliable"].map(p2b.as_bool).all()
        ),
        "geometry_audits": audits,
    }
    diag.write_json(args.output_dir / "phase3_geometry_audits.json", audit_bundle)

    if args.audit_only:
        print("\nPhase 3 orbital-mapping audit:")
        display_columns = [
            "bond_length_angstrom",
            "reference_to_canonical_mapping_json",
            "minimum_individual_abs_overlap",
            "minimum_degenerate_group_singular_value",
            "occupied_reference_maps_to_canonical",
            "active_occupation_pattern_ok",
            "tracking_reliable",
        ]
        print(orbital_summary.reindex(columns=display_columns).to_string(index=False))
        print(f"\nAudit-only outputs saved in: {args.output_dir.resolve()}")
        return 0

    protocol = protocol_fingerprint(
        systems,
        bond_lengths,
        excitation_mappings,
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
        bond_lengths,
        variants_to_run,
        excitation_mappings,
        args.output_dir,
        protocol,
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
        bond_lengths,
        excitation_mappings,
        args.output_dir,
        args.base_restarts,
        args.rescue_restarts,
        args.disagreement_tolerance,
    )
    geometry_summary = make_geometry_summary(
        variant_summary,
        orbital_summary,
        args.output_dir,
    )
    conclusions = make_conclusions(
        protocol,
        variant_summary,
        geometry_summary,
        orbital_summary,
        args.output_dir,
    )
    write_report(variant_summary, geometry_summary, conclusions, args.output_dir)

    print("\nPhase 3 geometry summary:")
    columns = [
        "bond_length_angstrom",
        "hf_error_ha",
        "full16_mean_error_exact_ha",
        "full16_robust_chemical_accuracy",
        "num_selected_variants_chemical",
        "best_selected_variant",
        "best_selected_error_exact_ha",
        "orbital_tracking_reliable",
    ]
    print(geometry_summary.reindex(columns=columns).to_string(index=False))
    print("\nPhase 3 conclusions:")
    print(json.dumps(diag.json_ready(conclusions), indent=2, sort_keys=True))
    print(f"\nSaved results in: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed Phase-3 attempts remain checkpointed.",
            file=sys.stderr,
        )
