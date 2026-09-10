# Data and API

## SAPP inputs

`fit(reference_xy, reference_peaks, parameters)` accepts an `(N, 2)` array of survey coordinates and a sequence of `N` one-dimensional arrays containing observed ranges. Coordinates and ranges are in metres. Each set can have a different length. Peak order carries no meaning for SAPP. Filter missing values before passing an observation to the core API.

`localize(query_peaks, model)` returns one two-dimensional coordinate per query. `decode="posterior_median"` selects the paper's primary estimator. `decode="map"` selects the posterior mode. An empty query returns the survey centroid. `save_anchor_map` and `load_anchor_map` serialize the fitted numerical arrays and parameters in an NPZ file.

The fitted `AnchorMap` contains effective source coordinates, source-count fields at the survey positions, source-discovery diagnostics, and residual delay weights. The source coordinate's third entry is its vertical offset relative to the receiver plane. It does not identify a physical AP or a wall.

The code follows four stages:

1. `discovery.py` proposes range surfaces and selects those supported across survey positions.
2. `fitting.py` assigns survey peaks to the sources and residual process.
3. `fields.py` interpolates expected source counts and residual delay weights over space.
4. `likelihood.py` scores query delay sets; `localization.py` searches and decodes the location distribution.

The fitting loop retains the published association updates. File organization and optional acceleration share this one implementation.

## Experiment files

| Location | Main arrays or content |
|---|---|
| `data/nist/design.json` | Room polygons, AP placements, survey/query coordinates, object layouts |
| `data/nist/receiver/*.npz` | `ranges`, with missing peaks represented by NaN; peaks retain receiver power order in the file |
| `data/nist/native/*.npz` | Exact native path ranges ranked by gain before the peak cap |
| `data/nist/channels/` | Compressed native channel records and simulator scene inputs |
| `data/uwb/train.npz`, `test.npz` | Measured coordinates and per-anchor peak sets |
| `data/uwb/survey.npz` | Retained survey indices and geometry |
| `data/power/*.npz` | `train_power`, `test_power`, coordinates, training folds, range-bin spacing |
| `data/cnn/observations/` | Additional survey observations used to reconstruct CNN encodings |
| `reference/` | Published prediction arrays, diagnostics, and SAPP maps |

NPZ files are read with pickle disabled. Query truth accompanies prediction files for evaluation; fitting reads the survey/training arrays. `experiments/common.py` centralizes file locations, set conversion, and atomic writes.

The receiver uses a `3e8` m/s range convention and the original array-spacing helper's `299792458` m/s wavelength convention. Retaining both reproduces the evaluated array response exactly. The preparation regression check compares the extracted peaks with the released arrays.

## Data provenance

The NIST quasi-deterministic simulations use generator revision `1ccf0cb61c1741cc3471db8ec0373c5383da58be`. The release contains the existing native channel outputs and scenario inputs for three rooms and six AP placements. `data/nist/generator.json` retains the generator and random-initialization metadata. Material-library paths in the scene configurations are relative to their `Input` directory. The original NIST simulator and publications are cited in `paper/references.bib`.

The measured inputs derive from the [Fraunhofer IIS fingerprinting dataset for positioning](https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/pos/fingerprinting-dataset-for-positioning.html). The repository includes the processed arrays used in this manuscript's UWB evaluation. They were prepared from the project's existing local download.

The starting manuscript archive was `SAPP_anonymous_v26_source_and_results.zip`, SHA-256 `77f7f48583d80d41ed1b3004fb07503f86fed47efed68e82b57f6bcaf47d80fa`. Manuscript v27 retained those numerical experiments. `data/manifest.json` records the hashes of the actual inputs distributed here.

Dataset and upstream software rights remain with their respective providers. See `THIRD_PARTY.md` for attribution and the separation between repository code and data.
