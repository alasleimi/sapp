# Reproduction

## Environments and outputs

Run commands from the repository root, or supply `--root /path/to/sapp`. Use `uv sync --locked --extra paper` for CPU or `uv sync --locked --extra cuda` for NVIDIA CUDA. The lockfile records both supported environments. The default core installation runs SAPP without PyTorch.

Each run has its own output directory. The default is `runs/reproduction`; `--output runs/my-run` selects another. Completed model and prediction files are reused when a stage resumes. Configuration changes require a new output directory. `run.json` records stage completion and package versions. Intermediate and completed outputs stay outside the input and reference directories.

## Stages

```sh
uv run --locked --extra paper sapp reproduce --list-stages
uv run --locked --extra paper sapp reproduce --stages primary native
```

| Stage | Computation | Main dependencies |
|---|---|---|
| `prepare` | Rebuild receiver peaks and strongest native delays from NIST channel records | Bundled native channels |
| `primary` | Refit six SAPP maps and evaluate 5,400 receiver queries with matching baselines | Receiver observations |
| `native` | Refit and evaluate the exact-delay experiment | Native delay observations |
| `surfaces` | Refit maps with contiguous survey strips withheld | Receiver survey |
| `optimized` | Independently tuned residual and alternative source estimator | Receiver survey and selected settings |
| `vt` | Published, calibrated, and tolerance-controlled VT interpolation (Zayets & Steinbach, 2018) | Receiver observations |
| `vt_diagnostics` | Held-out VT delay predictions and smooth-source positive control | Receiver survey |
| `components` | Matched SAPP component and decoder controls | Primary maps, fitted if absent |
| `sensitivity` | Survey density, bandwidth, SNR, and clock-offset experiments | Receiver conditions |
| `seeds` | Three additional source-discovery seeds | Receiver observations |
| `counts` | 24-peak count-term comparison | Receiver observations |
| `cnn` | Common-survey validation curves and three final CNN fits per room | CNN survey observations |
| `cnn_budget` | Additional CNN layout and AP training regimes | Additional CNN observations |
| `networks` | P-NN, L-SwiGLU, and PDP Transformer; three fits per setting | Power profiles |
| `uwb` | Measured SAPP and residual estimates across survey sizes and anchor subsets | Local UWB train/test observations |
| `uwb_matching` | Measured MPUrge-MAP and Chamfer comparisons | Local UWB observations |
| `figures` | Draw all twelve figures | Relevant fitted maps and predictions |
| `tables` | Calculate result tables and paired intervals | Relevant predictions and diagnostics |

The default `paper` profile runs all stages except `prepare`. The `primary`, `native`, and `uwb` profiles provide smaller complete evaluation groups. Use the same profile and tuning choice when resuming a run.

`--tune` repeats the extra spatial-validation searches for the residual and recent neural models and the UWB intensity models. Default refits use the recorded selections in `configs/`. Chamfer validation and the CNN learning curves are part of their regular stages. Test coordinates are used to evaluate final predictions. The raw training and test partitions are separate files.

## Figures and manuscript

| File | Contents | Required stage outputs |
|---|---|---|
| `method.pdf` | Observation, fitted model, query score | `primary` |
| `rooms.pdf` | Room geometry and survey layout | Experiment design |
| `cdf.pdf` | Main error distributions | `primary`, `cnn`, `networks`, `vt` |
| `ablations.pdf` | Geometric and residual component effects | `primary`, `components` |
| `surface_validation.pdf` | Delay prediction in a withheld survey strip | `surfaces` |
| `sensitivity.pdf` | Survey and receiver changes | `primary`, `sensitivity` |
| `failure_diagnostic.pdf` | Multimodal likelihood and observed peaks | `primary` |
| `uwb_cdf.pdf` | Joint measured error distributions | `uwb`, `uwb_matching`, `networks` |
| `cnn_validation.pdf` | CNN training and validation curves | `cnn` |
| `recent_cdf.pdf` | Recent neural comparators | `primary`, `uwb`, `networks` |
| `uwb_survey.pdf` | Measured survey and test geometry | UWB data |
| `uwb_operating.pdf` | Survey-size and anchor-count comparisons | `uwb` with the `paper` profile |

`sapp figures --only failure_diagnostic` redraws a selected figure. `--from reference` reads the archived predictions and maps; `--from run` reads the chosen output directory. Plotting contains no training and does not substitute archived predictions into fresh experiment outputs.

The `paper` directory contains the manuscript and supplement sources. The PDF build uses the figures and tables generated in the selected output directory. Numerical values written in prose are maintained in the manuscript source; update them when changing the experiment configuration.

## Checks

```sh
uv run --locked --extra paper sapp verify --data-only
uv run --locked --extra paper sapp verify --profile primary
uv run --locked --extra paper python -m pytest
```

Input integrity is checked against `data/manifest.json`. Numerical verification writes `verification.json`, reports missing results, and compares individual errors and error-distribution summaries with the released predictions. The accelerated analytical scorer is checked against the double-precision score during evaluation. Near-tied mode candidates are rescored in double precision.

Neural training records each seed, selected configuration, history, weights, and predictions. CNN fits save optimizer and shuffle checkpoints every 25 epochs and resume from those checkpoints. CPU and CUDA kernels can produce different optimization trajectories. Verification reports these as `RETRAINED`, with the reference metrics and their differences. The figures average independent neural fits as specified in the paper.

## Starting point for channel reproduction

The native NIST channel records, room geometry, materials, and simulator configuration are included. `prepare` reruns the receiver and delay extraction from those records. It does not launch MATLAB to regenerate stochastic ray-tracer channels. The simulator revision and scenario files are described in [data provenance](data-and-api.md#data-provenance).

The UWB release inputs contain the extracted observed peak sets and power profiles from the already downloaded Fraunhofer measurements. Reproduction consumes those local files. No command downloads or replaces datasets.
