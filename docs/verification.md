# Release verification

The release was checked on Windows with Python 3.11.9, the locked numerical dependencies, and an NVIDIA RTX 4050 for neural training. The installed wheel was also tested against an extracted copy of the source ZIP in an isolated environment.

## SAPP results

The main receiver and native-delay experiments were refitted from observations, with 5,400 queries in each experiment. The following errors are in metres.

| Experiment | Mean | Median | P90 | P95 | RMSE | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| NIST receiver peaks | 1.121978 | 0.756899 | 2.666059 | 3.400231 | 1.551542 | 5.708940 |
| NIST native delays | 1.202642 | 0.727732 | 3.046940 | 3.869482 | 1.728372 | 7.082519 |
| Measured UWB, six anchors | 0.278882 | 0.247206 | 0.484872 | 0.587756 | 0.369569 | 10.343606 |

Every main NIST posterior-mode prediction matches the reference. The largest individual error difference for the primary geometric-median estimates is 0.000027 m. All 24 measured survey-size, anchor-count, and model combinations match their reference predictions exactly.

The multimodal example in the paper is preserved: room L, AP LA, layout L05, query index 81. Its mode error is 9.048458 m and its geometric-median error is 2.839473 m. Both positions match the reference. The tests also retain the separate sensitive mode estimate from AP LB, layout L02, query index 44.

## Baselines and experiment checks

Matching baselines, all three VT configurations, the source and residual controls, independently selected residual settings, survey-strip fits, sensitivity experiments, and source-discovery seeds were rerun. Verification checks individual errors and mean, median, P90, P95, RMSE, maximum, and the fraction above two metres.

The neural comparison includes three independent fits per setting. L-SwiGLU and the PDP Transformer reproduce the reference predictions exactly. P-NN retraining changes the pooled mean by -0.000807 m on NIST and -0.005955 m on UWB. Its corresponding maximum errors change by +0.107479 m and -0.114237 m. The verification report records these training differences separately.

The CNN common-survey and both additional-budget experiments reproduce their reference predictions exactly at both reported training durations. The complete [machine-readable report](verification.json) contains 157 comparisons: 155 pass the numerical tolerance and two record the P-NN retraining differences. No result group is missing.

## Data, figures, and package

- All 1,971 distributed input files pass their SHA-256 checks, including after ZIP extraction.
- All 140 NIST receiver files and 60 native-delay files regenerate exactly from the bundled channel records.
- All twelve figures regenerate from model refits. The manuscript and supplement compile to 11 and 14 pages, with no unresolved references or overfull boxes.
- The extracted source bundle passes the numerical, encoding, baseline, and training-resume tests. A further test verifies that nonfinite predictions fail verification.
- The core wheel runs its bundled demo with NumPy and SciPy. Ruff checks and wheel construction pass.

See [the reproduction guide](reproduction.md) for commands and the distinction between refitting published settings and repeating the additional validation searches.
