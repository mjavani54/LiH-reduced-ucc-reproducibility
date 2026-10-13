#!/usr/bin/env python3
"""Verify checksums and principal claims of the post-review extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.sha256"
CHEMICAL_HA = 1.6e-3


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def package_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
    )


def write_manifest() -> None:
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in package_files()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest() -> None:
    if not MANIFEST.exists():
        raise AssertionError("MANIFEST.sha256 is missing")
    declared: dict[str, str] = {}
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        checksum, relative = raw.split("  ", 1)
        declared[relative] = checksum
    actual = {path.relative_to(ROOT).as_posix(): digest(path) for path in package_files()}
    if set(declared) != set(actual):
        missing = sorted(set(actual) - set(declared))
        extra = sorted(set(declared) - set(actual))
        raise AssertionError(f"Manifest membership mismatch; missing={missing}, extra={extra}")
    failures = [name for name, checksum in actual.items() if declared[name] != checksum]
    if failures:
        raise AssertionError(f"Checksum mismatch: {failures}")


def verify_scientific_invariants() -> None:
    lih = read_csv("main_and_holdout_support_results.csv")
    assert len(lih) == 28, f"Expected 28 LiH rows, found {len(lih)}"
    geometries = {float(row["bond_length_angstrom"]) for row in lih}
    assert geometries == {1.4, 1.595, 1.8, 2.1, 2.5, 2.75, 3.0}
    held_out = {
        float(row["bond_length_angstrom"])
        for row in lih
        if row["geometry_role"] == "held_out"
    }
    assert held_out == {1.4, 1.8, 2.1, 2.75}
    partner = [row for row in lih if row["variant"] == "spin_complete_candidate47"]
    assert len(partner) == 7
    assert max(float(row["absolute_error_ha"]) for row in partner) < 1e-12
    assert max(float(row["s2"]) for row in partner) < 1e-10
    assert max(float(row["gradient_infinity_norm"]) for row in partner) < 1e-7

    exhaustive = read_csv("boundary46_plus_one_exhaustive_summary.csv")
    assert len(exhaustive) == 53
    strict = [row for row in exhaustive if truth(row["all_geometry_strict"])]
    assert len(strict) == 1 and int(strict[0]["added_global_id"]) == 17

    random_controls = read_csv("partner_closed_random_control_summary.csv")
    assert len(random_controls) == 100
    assert not any(truth(row["all_geometry_chemical"]) for row in random_controls)

    with (ROOT / "results" / "adapt_vqe_final_supports.json").open(encoding="utf-8") as handle:
        adapt = json.load(handle)
    assert len(adapt) == 3
    assert {int(row["operator_count"]) for row in adapt} == {47}
    selected_sets = {tuple(sorted(int(value) for value in row["selected_ids"])) for row in adapt}
    assert len(selected_sets) == 1

    ranking = read_csv("ranking_sensitivity.csv")
    assert len(ranking) == 9
    assert all(truth(row["top37_set_identical_to_baseline"]) for row in ranking)

    circuits = {row["support"]: row for row in read_csv("matched_circuit_resources.csv")}
    assert int(circuits["partner47"]["compiled_cx"]) < int(circuits["full99"]["compiled_cx"])
    assert int(circuits["partner47"]["compiled_depth"]) < int(circuits["full99"]["compiled_depth"])

    h4 = read_csv("h4_transfer_results.csv")
    assert len(h4) == 12
    for spacing in (1.6, 2.0):
        rows = [row for row in h4 if float(row["spacing_angstrom"]) == spacing]
        for support in ("reference_ADAPT_prefix_spin_closed", "full_factorized_UCCSD"):
            row = next(item for item in rows if item["support_name"] == support)
            assert float(row["absolute_error_ha"]) > CHEMICAL_HA
    assert max(float(row["gradient_infinity_norm"]) for row in h4) < 1e-7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        print(f"Wrote {MANIFEST}")
        return
    verify_manifest()
    verify_scientific_invariants()
    print("Post-review extension verification passed.")


if __name__ == "__main__":
    main()

