# LiH Phase 7C Expanded-Space Augmentation Discovery

Analysis status: **complete_candidate_frozen_for_phase7d_confirmation**

Minimal chemical external k: **36**

Guarded external k: **37**

Phase-7D confirmation external k: **37**

## Frozen decision rule

The primary candidate is the first stable nested support below 0.0016 Ha at all three geometries. The confirmation candidate is the first stable support below 0.0008 Ha, with fallback to the primary candidate.

## Deterministic path

| External k | Total operators | Stable at all geometries | Worst error (Ha) | Chemical | Guarded |
|---:|---:|:---:|---:|:---:|:---:|
| 0 | 10 | True | 0.0084590541 | False | False |
| 1 | 11 | True | 0.0083147978 | False | False |
| 2 | 12 | True | 0.0082571065 | False | False |
| 3 | 13 | True | 0.0082016815 | False | False |
| 4 | 14 | True | 0.0077611054 | False | False |
| 5 | 15 | True | 0.0073047227 | False | False |
| 6 | 16 | True | 0.0068748883 | False | False |
| 7 | 17 | True | 0.0068606934 | False | False |
| 8 | 18 | True | 0.0068466936 | False | False |
| 9 | 19 | True | 0.0068402086 | False | False |
| 10 | 20 | True | 0.0068337976 | False | False |
| 11 | 21 | True | 0.0067174662 | False | False |
| 12 | 22 | True | 0.0066021662 | False | False |
| 13 | 23 | True | 0.0065963796 | False | False |
| 14 | 24 | True | 0.0065906419 | False | False |
| 15 | 25 | True | 0.0065863924 | False | False |
| 16 | 26 | True | 0.0065821985 | False | False |
| 17 | 27 | True | 0.0065778493 | False | False |
| 18 | 28 | True | 0.0065735557 | False | False |
| 19 | 29 | True | 0.0055570647 | False | False |
| 20 | 30 | True | 0.0046668858 | False | False |
| 21 | 31 | True | 0.0046629009 | False | False |
| 22 | 32 | True | 0.0046590521 | False | False |
| 23 | 33 | True | 0.0046535296 | False | False |
| 24 | 34 | True | 0.0046480373 | False | False |
| 25 | 35 | True | 0.0046464853 | False | False |
| 26 | 36 | True | 0.0046449646 | False | False |
| 27 | 37 | True | 0.0046449050 | False | False |
| 28 | 38 | True | 0.0046424291 | False | False |
| 29 | 39 | True | 0.0046399728 | False | False |
| 30 | 40 | True | 0.0046374619 | False | False |
| 31 | 41 | True | 0.0046349703 | False | False |
| 32 | 42 | True | 0.0042775500 | False | False |
| 33 | 43 | True | 0.0039512726 | False | False |
| 34 | 44 | True | 0.0037089366 | False | False |
| 35 | 45 | True | 0.0034854583 | False | False |
| 36 | 46 | True | 0.0014633948 | True | False |
| 37 | 47 | True | 0.0000000004 | True | True |

## Audits and controls

Exact Hamiltonian audit passed: **True**.

compact10 reproduces the internal exact result: **True**.

Full99 worst exact error: **1.2184138142856682e-09 Ha**.

## Claim boundary

Phase 7C is a determinant-simulator discovery result for frozen LiH/6-31G 2e,10o Hamiltonians. The selected support is a hypothesis, not a confirmed production-Qiskit VQE result. Phase 7D must be preregistered and frozen before confirmation.
