# LiH manuscript revision repairs

This is an overlay for a new branch of the archived v0.9.0 project, not a complete
repository or a tested numerical rerun. No changes have been pushed to GitHub.
Check the live repository against the archived release before integrating it.

## Included changes

- `src/lih_reference_failure_diagnostics.py`: recovered complete source replacing
  the release's reconstructed helper. See `docs/SOURCE_RECOVERY_ADDENDUM.md`.
- `analysis/make_figures.py`: Figure 2 reads group singular-value quality and plots
  the compact10 all-geometry maximum as a horizontal line without data markers.
- `figures/figure2_cross_basis_transfer.png` and `.pdf`: corrected figure exports.
- `scripts/audit_submission.py`: portable, standard-library audit of energy-table
  consistency, confirmation thresholds, and archived support closure.
- `docs/random_support_spin_closure.csv`: twenty reconstructed support identities
  and their missing partners.
- `docs/submission_archive_audit.json`: revision findings and source hashes.

## Integration and checks

1. Preserve the original release ZIP and manifest. Create a new development branch
   in the actual repository. Inspect differences before replacing files, especially
   if the live repository contains work newer than the archived package.
2. Before applying this overlay, run the original verifier in the original tree:

   ```bash
   python scripts/verify_package.py
   ```

3. Copy the overlay's files into the corresponding paths on the new branch. The
   original manifest should now report changed files; that is expected. Do not
   describe this modified tree as byte-identical to the original release.
4. Run the new archive audit and regenerate figures in the project environment:

   ```bash
   python scripts/audit_submission.py
   bash scripts/reproduce_figures.sh
   ```

   The audit can also examine another checkout without modifying it:

   ```bash
   python scripts/audit_submission.py --root /path/to/project
   ```

5. In a clean, pinned computational environment, execute the supplied numerical
   pipeline, which writes new results under work:

   ```bash
   bash scripts/capture_environment.sh
   bash scripts/run_phase7_pipeline.sh
   ```

   This long numerical pipeline has NOT been executed as part of this revision.
   Preserve logs and export exact dependency versions. Capture thread settings and
   numerical-library details as well. The original Phase 2B–6 raw outputs are not
   all present; regenerating them is separate from the supplied Phase 7 pipeline.
6. Compare energies, spin values, supports, orbital maps, and fingerprints at their
   declared tolerances. Never overwrite the historical records under their old
   fingerprints. A changed numerical ranking defines a new experiment.
7. After successful integration and validation, copy the relevant environment
   records into a tracked release directory (the current manifest generator
   excludes `environment_capture` and `work`). Update README and source-provenance
   status, generate a NEW manifest, and verify that new release:

   ```bash
   python scripts/generate_manifest.py
   python scripts/verify_package.py
   ```

8. Publish an immutable release/commit identifier and update the manuscript's
   availability statement. A DOI is useful archival practice but is not claimed
   to have been assigned by this repair package.

## Expected archive-audit results

Twelve geometry/variant summaries; twenty-four restart records; candidate47 closed
under spin partners; boundary46 missing only E17; 0/20 random controls partner
closed; largest final gradient infinity norm 2.3772601581817376e-05. This check is
record verification, not evidence that the recovered source has been rerun.
