#!/usr/bin/env python3
"""Verify repository integrity and the principal deposited numerical claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE7C_PROTOCOL = "23e67a5e7699394b0da8b00100edf65822400169421eee30b017be31dbf4d33f"
PHASE7C_SUPPORT = "86bec8f309b3945e5bbb0e7bb4776a3411ed692f52cc49798be4de74576e0026"
PHASE7D_PROTOCOL = "454022626abb6f3096e3fda104e2d34c05050eee9a63176c50a62423e8b3c6aa"
PHASE7D_SUPPORT = "0049b07d140790c3d87336bccf293a344fa527001526b014fd074e6c8a2ac082"
STRICT_HA = 1.0e-6
CHEMICAL_HA = 1.6e-3
SPIN_TOLERANCE = 1.0e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(relative: str) -> dict:
    path = ROOT / relative
    require(path.is_file(), f"Missing required JSON: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    require(path.is_file(), f"Missing required CSV: {relative}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_manifest() -> int:
    path = ROOT / "MANIFEST.sha256"
    require(path.is_file(), "MANIFEST.sha256 is missing")
    checked = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        try:
            expected, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise AssertionError(f"Malformed manifest line {line_number}") from exc
        file_path = ROOT / relative
        require(file_path.is_file(), f"Manifest file is missing: {relative}")
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        require(actual == expected, f"Checksum mismatch: {relative}")
        checked += 1
    require(checked > 0, "Manifest contains no file records")
    return checked


def verify_scientific_records() -> None:
    phase6 = load_json("results/phase6/phase6_final_conclusions.json")
    require(
        phase6.get("analysis_status") == "complete_with_reproducible_best_basin_rescues",
        "Unexpected Phase-6 status",
    )

    phase7b = load_json("results/phase7b/phase7b_transfer_conclusions.json")
    require(phase7b.get("execution_complete") is True, "Phase 7B is incomplete")
    require(
        phase7b.get("analysis_status") == "complete_strict_support_transfer",
        "Unexpected Phase-7B status",
    )

    phase7c = load_json("results/phase7c/phase7c_augmentation_conclusions.json")
    require(phase7c.get("execution_complete") is True, "Phase 7C is incomplete")
    require(phase7c.get("phase7c_protocol_fingerprint") == PHASE7C_PROTOCOL, "Phase-7C protocol fingerprint mismatch")
    require(phase7c.get("phase7c_support_fingerprint") == PHASE7C_SUPPORT, "Phase-7C support fingerprint mismatch")
    require(phase7c.get("phase7d_confirmation_total_operator_count") == 47, "Phase-7C candidate count is not 47")

    phase7d = load_json("results/phase7d/phase7d_confirmation_conclusions.json")
    require(phase7d.get("execution_complete") is True, "Phase 7D is incomplete")
    require(phase7d.get("phase7d_protocol_fingerprint") == PHASE7D_PROTOCOL, "Phase-7D protocol fingerprint mismatch")
    require(phase7d.get("phase7d_support_fingerprint") == PHASE7D_SUPPORT, "Phase-7D support fingerprint mismatch")
    require(
        phase7d.get("primary_confirmation_classification")
        == "strict_spin_complete_expanded_space_confirmation",
        "Unexpected Phase-7D classification",
    )

    candidate = phase7d["confirmed_candidate"]
    require(candidate["operator_count"] == 47, "Confirmed candidate count is not 47")
    require(candidate["worst_geometry_error_ha"] <= STRICT_HA, "Candidate misses strict energy tolerance")
    require(candidate["maximum_spin_square"] <= SPIN_TOLERANCE, "Candidate misses spin tolerance")
    require(candidate["full99_operator_reduction_count"] == 52, "Unexpected operator reduction")

    boundary = phase7d["chemical_boundary_control"]
    require(boundary["operator_count"] == 46, "Boundary control count is not 46")
    require(boundary["worst_geometry_error_ha"] <= CHEMICAL_HA, "Boundary control misses chemical accuracy")
    require(boundary["structurally_spin_incomplete"] is True, "Boundary control should be spin incomplete")
    require(boundary["maximum_spin_square"] > SPIN_TOLERANCE, "Boundary spin failure is absent")

    spotcheck = load_json("results/phase7d/phase7d_full_statevector_spotcheck.json")
    require(spotcheck.get("spotcheck_passed") is True, "Full-statevector spot-check failed")
    require(spotcheck.get("maximum_instruction_qubits", 99) <= 2, "Spot-check used a wide dense instruction")
    require(spotcheck.get("avoided_dense_operator_bytes_complex128") == 16 * 1024**4, "Unexpected avoided dense allocation")
    require(spotcheck.get("energy_residual_ha", math.inf) <= 1.0e-8, "Spot-check energy residual is too large")

    rows = load_csv("results/phase7d/phase7d_variant_geometry_summary.csv")
    require(len(rows) == 12, "Phase-7D summary must contain 12 rows")
    variants = {row["variant"] for row in rows}
    require(
        variants == {"compact10", "chemical_boundary46", "spin_complete_candidate47", "full99"},
        "Unexpected Phase-7D variants",
    )
    geometries = {round(float(row["bond_length_angstrom"]), 3) for row in rows}
    require(geometries == {1.595, 2.5, 3.0}, "Unexpected Phase-7D geometries")

    for number, stem in {
        1: "leave_one_out_ablation",
        2: "cross_basis_transfer",
        3: "expanded_space_augmentation",
        4: "frozen_confirmation",
    }.items():
        png = ROOT / "figures" / f"figure{number}_{stem}.png"
        pdf = ROOT / "figures" / f"figure{number}_{stem}.pdf"
        require(png.is_file() and png.stat().st_size > 1_000, f"Invalid PNG: {png.name}")
        require(pdf.is_file() and pdf.stat().st_size > 1_000, f"Invalid PDF: {pdf.name}")
        require(pdf.read_bytes().startswith(b"%PDF-"), f"Malformed PDF header: {pdf.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip checksums after intentionally regenerating figure files.",
    )
    args = parser.parse_args()
    count = 0 if args.skip_manifest else verify_manifest()
    verify_scientific_records()
    if args.skip_manifest:
        print("PASS: deposited scientific records (manifest intentionally skipped)")
    else:
        print(f"PASS: {count} checksums and all deposited scientific records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
