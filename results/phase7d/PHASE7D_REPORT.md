# LiH Phase 7D Frozen Qiskit Confirmation

Analysis status: **complete_strict_47_operator_spin_complete_confirmation**

Primary classification: **strict_spin_complete_expanded_space_confirmation**

## Variational confirmation

| R (Å) | Variant | Operators | Error vs exact (Ha) | S² | Stable | Strict | Chemical |
|---:|---|---:|---:|---:|:---:|:---:|:---:|
| 1.595 | chemical_boundary46 | 46 | 1.7401279320e-04 | 5.143e-04 | True | False | True |
| 1.595 | compact10 | 10 | 2.1085664863e-03 | 1.119e-11 | True | False | False |
| 1.595 | full99 | 99 | 9.5403684952e-11 | 8.030e-12 | True | True | True |
| 1.595 | spin_complete_candidate47 | 47 | 9.5400132238e-11 | 8.030e-12 | True | True | True |
| 2.500 | chemical_boundary46 | 46 | 5.8001918865e-04 | 2.623e-03 | True | False | True |
| 2.500 | compact10 | 10 | 4.3020978886e-03 | 6.344e-11 | True | False | False |
| 2.500 | full99 | 99 | 1.0775291770e-10 | 4.030e-09 | True | True | True |
| 2.500 | spin_complete_candidate47 | 47 | 5.9887206305e-11 | 5.503e-10 | True | True | True |
| 3.000 | chemical_boundary46 | 46 | 1.4633948039e-03 | 2.337e-02 | True | False | True |
| 3.000 | compact10 | 10 | 8.4590540937e-03 | 2.129e-12 | True | False | False |
| 3.000 | full99 | 99 | 1.2421619289e-10 | 1.673e-09 | True | True | True |
| 3.000 | spin_complete_candidate47 | 47 | 2.8108715355e-10 | 2.588e-09 | True | True | True |

## Mechanism

The Phase-7C k=36 to k=37 discontinuity is a spin-complement completion effect. E8 is the alpha excitation 0->9; E17 is its beta counterpart 10->19. The frozen 47-operator support is fully closed under alpha/beta spin complementation, whereas boundary46 contains E8 without E17.

## Claim boundary

The result confirms a noiseless 47-operator spin-complete UCC support for the three frozen LiH/6-31G 2e,10o Hamiltonians. It does not establish transfer to other molecules, bases, active spaces, geometries outside the tested set, or noisy hardware.
