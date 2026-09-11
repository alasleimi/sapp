# Baselines

All estimators used in the manuscript have implementations in `src/sapp/baselines/`. Experiment adapters apply the observation format, survey budget, and selection procedure specified in the manuscript. No external project checkout is imported.

Import the method by name: `from sapp.baselines import mpurge_map, mca`. Use `mpurge_map.localize(query_sets, survey_sets, survey_xy)` for a batch of MPUrge-MAP estimates, or `mca.localize(query_set, survey_sets, survey_xy, epsilon_m=0.5)` for one MCA estimate. The range arrays contain finite values in metres, and survey coordinates have shape `(N, 2)`. Both modules require NumPy. `_mpurge_map_numba.py` provides optional compiled acceleration.

| Comparator | Implementation | Reference and role |
|---|---|---|
| MCA | `mca.py` | [Multipath component analysis, 2017](https://doi.org/10.1109/IPIN.2017.8115943); direct delay-set matching |
| MPUrge-MAP | `mpurge_map.py`, `_mpurge_map_numba.py` | [MPUrge-MAP, 2026](https://doi.org/10.1109/IWCMC69287.2026.11580087); coverage-penalized matching and coordinate estimation |
| VT interpolation (Zayets & Steinbach, 2018) | `virtual_transmitters.py` | [Zayets and Steinbach, 2018](https://doi.org/10.1109/ICC.2018.8422206); match clusters, fit virtual transmitters, synthesize fingerprints |
| Chamfer weighted kNN | `chamfer.py` | Symmetric nearest-peak distance and inverse-distance coordinate weighting; spatial validation selects neighbors and weight exponent |
| CNN regressor | `cnn.py`, `_cnn_cuda.py` | [CNN multipath positioning, 2026](https://doi.org/10.1109/IWCMC69287.2026.11580049); the paper's encoded delay and coordinate input, with three independent fits |
| P-NN | `networks.py` | [Oh et al., 2024](https://doi.org/10.1109/JSAC.2024.3413977); power-profile network |
| L-SwiGLU | `networks.py` | [2025 preprint](https://arxiv.org/abs/2501.07774); sensor-set processing with SwiGLU blocks |
| PDP Transformer | `networks.py` | [Cao et al., 2026](https://doi.org/10.3390/s26051486); joint power-delay-profile input |

The main VT result uses the development-selected interpolation settings. The supplement also evaluates the published settings and a shared matching tolerance. These are three configurations of one implementation. `source_fitting.py` supplies the alternative source estimator used in the matched source-fitting experiment.

The SAPP component controls reuse its likelihood and decoder: residual only, sources only, constant visibility, conditional peak count, and integrated intensity. `experiments/optimized.py` additionally selects the residual model's smoothing and temperature independently.

NIST comparisons evaluate each AP separately. Measured joint UWB comparisons combine all six anchors. The neural UWB inputs retain that six-anchor dimension, including for the PDP Transformer. The additional CNN survey regimes live in `experiments/cnn_budget.py`; they use the same regressor as the common-survey experiment.

`paper/references.bib` contains the full bibliography. Comparator reconstructions and data transformations follow the manuscript's experimental setting; the corresponding original publications describe their original data and budgets.
