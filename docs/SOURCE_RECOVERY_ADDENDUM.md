# Source recovery addendum

This addendum accompanies the manuscript revision. It applies to the archived
v0.9.0 release-candidate package; the live GitHub commit has not been verified.

The earlier provenance document recorded a truncated 16,384-byte copy of
`lih_reference_failure_diagnostics.py` and a reconstructed replacement. A complete
43,483-byte source copy has now been recovered from the available project files.

| Identity | SHA-256 |
| --- | --- |
| First 16,384 bytes of recovered source | `66c4a21fee479badd838bab40ffc8c862a2813d2b6ab13d168b2c9b5d447ffd5` |
| Complete recovered source | `13af1c3cf44e16da229f9da88c8087e9306ea884254b319ae2767b6360c82997` |

The prefix digest exactly matches the documented truncated artifact. The full
file parses successfully. This resolves recovery of a complete source candidate,
but does not establish execution equivalence with the historical results. The
recovered copy is supplied at `src/lih_reference_failure_diagnostics.py`.

Historical software versions remain unavailable. The existing environment.yml
must continue to be described as a proposed target environment. A clean rerun,
captured versions, and comparison with the deposited numerical records are still
needed. Preserve the old release and its manifest; use a new release for revisions.

The figure repair changes presentation only: Figure 2 uses the actual orbital
group-quality metric and presents the reported compact10 maximum as a horizontal
bound, without representing it as three independent energy observations.

The structural audit reconstructs the archived composition-matched random
supports. All twenty have ten singles and 37 doubles; none is closed under the
spin-partner map. These are newly checked combinatorial properties of existing
support identities, not new VQE measurements.
