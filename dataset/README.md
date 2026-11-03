# Datasets

This directory contains documentation only. Download or generate data separately;
no recordings, generated trajectories, or prepared reproduction datasets are
distributed with this repository. Local data files in this directory are ignored
by Git.

## Sources used in the paper

The references below correspond to *Discovering Phase Space Structure in Learned
Hamiltonian Systems*, NeurIPS 2026, Sections 5 and Appendix A.

| Data | Source and paper reference |
| --- | --- |
| Simulated mass-spring, ideal pendulum, double pendulum, two-body, and three-body systems | The paper uses the [HGN reimplementation simulation pipeline](https://github.com/CampusAI/Hamiltonian-Generative-Networks), Balsells Rodas, Canal, and Taschin, reference [49]. This is a simulator source repository, not a download of the exact SPS-GAN training set. Appendix A also describes damped-pendulum and forced-Duffing experiments; reproducing them requires the corresponding simulation settings and extensions. |
| Real single-pendulum trajectories | Schmidt and Lipson, [*Distilling Free-Form Natural Laws from Experimental Data*](https://doi.org/10.1126/science.1165893), Science 324, 81–85 (2009), reference [46]. Obtain the data through the article's supplementary materials. The [original HNN real-data loader](https://github.com/greydanus/hamiltonian-nn/blob/master/experiment-real/data.py) documents the archive `invar_datasets.zip` and recording `real_pend_h_1`; its historical download URL may no longer work. |
| Real double-pendulum trajectories | Kaheman et al., *The Experimental Multi-Arm Pendulum on a Cart: A Benchmark System for Chaos, Learning, and Control*, HardwareX 15, e00465 (2023), reference [42]. See the authors' [data directory](https://github.com/dynamicslab/MultiArm-Pendulum/tree/main/Datas) and [repository documentation](https://github.com/dynamicslab/MultiArm-Pendulum). |
| Real single-pendulum videos | Castañeda Garcia et al., *Learning Physics From Video: Unsupervised Physical Parameter Estimation for Continuous Dynamical Systems*, CVPR 2025, reference [41]. The authors' [official repository](https://github.com/Alejandro-neuro/Learning_physics_from_video) links to [Delfys75](https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction); the paper uses pendulum recordings. |
| Real double-pendulum videos | Kaheman et al., reference [42]. The authors provide the recordings as `Video Data.zip` in [Zenodo record 6633719](https://zenodo.org/records/6633719), linked from their repository. |

Select `trajectory` or `video` under `[data] mode` in `config/example.ini`.
Follow each source's own license and citation requirements. The repository's
MIT license does not relicense external datasets.

## Preparing trajectories

Prepare a NumPy `.npz` archive with these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `trajectories` | `[N, T, 2, n]` | Cartesian positions for `N` sequences, `T` timesteps, the `x`/`y` coordinate axis, and up to `n` particles. |
| `mask` (optional) | `[N, n]` or `[n]` | Binary particle-validity mask. Use zero for padded particles and one for observed particles. |
| `conditioning` (required when `conditioning_dim > 0`) | `[N, conditioning_dim]` | Numeric system information, such as fixed system encodings or physical parameters, in a consistent column order. |

Use finite `float32` values. Pad systems with fewer particles with zeros and
supply their masks. Keep all sequences in an archive at the same timestep,
length, coordinate convention, and maximum particle count. Configure the model's
particle count and conditioning dimension to match the archive. Omit
`conditioning` for an unconditional experiment.

For example, given an already prepared array of Cartesian coordinates with shape
`[N, T, n, 2]`, export it from the repository root as follows:

```python
from pathlib import Path

import numpy as np

# Supply coordinates extracted from your chosen source. This example assumes
# all particles are valid and sequences have already been segmented/resampled.
coordinates = np.load("/path/to/cartesian_coordinates.npy", allow_pickle=False)
if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
    raise ValueError("Expected coordinates with shape [N, T, n, 2]")
trajectories = np.asarray(coordinates.transpose(0, 1, 3, 2), dtype=np.float32)
if not np.isfinite(trajectories).all():
    raise ValueError("Coordinates must be finite")
mask = np.ones((trajectories.shape[0], trajectories.shape[-1]), dtype=np.float32)
Path("dataset").mkdir(exist_ok=True)
np.savez_compressed("dataset/pendulum.npz", trajectories=trajectories, mask=mask)
```

This is a format-conversion example, not preprocessing for the raw source
archives. Depending on the source, preprocessing may require converting angles
to Cartesian positions, selecting recordings, resampling, and forming sequence
windows. Do not treat an angle/velocity pair as an `x`/`y` position pair. Record
the source version, selected recordings, coordinate conversion, scaling, sampling
interval, and split procedure alongside your local experiment.

Appendix A specifies simulated sequences of 30 frames at a timestep of 0.05,
with 16-frame discriminator subsequences, as configured in `config/example.ini`.
The paper's long-horizon double-pendulum
evaluation uses 200 frames. These are paper settings, not guarantees about the
sampling rate or layout of downloaded real recordings. The paper's simulated
data is generated from the initial-condition and physical-parameter ranges in
Appendix A; the upstream simulator's defaults alone do not establish an exact
reproduction.

## Preparing videos

Video mode reads NPZ archives with the following arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `videos` | `[N,T,C,96,96]` | RGB (`C=3`) or grayscale (`C=1`) frame sequences. |
| `conditioning` (required when `conditioning_dim > 0`) | `[N,conditioning_dim]` | Numeric system descriptors in the same column order for all clips. |

Store pixels as uint8 in `[0,255]` or as finite floating values already normalized
to `[-1,1]`. The loader normalizes uint8 pixels after cropping each clip, retaining
compact byte storage in memory. Floating arrays are used as supplied; `[0,1]`
floating images must be converted explicitly with `2 * images - 1`. The loader
does not infer pixel scales, resize frames, or change the sampling interval.

Set `[video] channels` to match the archive. All frames must be 96×96, and each
clip must contain at least `[data] num_frames` frames. Longer clips are randomly
cropped. `discriminator_frames` selects the temporal window used by the video
discriminator and must be at least 16 and no greater than `num_frames` (or `0`
to use the full sequence).

For raw recordings, set `mode = video` and run:

```bash
uv run --python 3.11 sps-gan prepare-video --config config/example.ini \
  --input /path/to/recording.mp4 --stride 1 --max-clips 1000
```

The command writes to `[paths] dataset`, or an explicit `--output` path. It
converts color channels, resizes to 96×96, retains every `stride`-th frame, and
forms nonoverlapping clips. Incomplete trailing clips are discarded. The output
contains the preparation settings and source frame rate as metadata. Set the
configuration's `dt` to the reported prepared frame spacing; for example, 30 FPS
at stride 3 gives `dt = 0.1`. Split source recordings before preparing training
and evaluation datasets. Preparation preserves all complete clips; choose
source segments externally if the experiment requires motion filtering.

For existing frame arrays stored as `[N,T,H,W,C]`, transpose the axes before
exporting:

```python
videos = frames.transpose(0, 1, 4, 2, 3)  # uint8, already resized to 96x96
np.savez_compressed("dataset/pendulum_video.npz", videos=videos)
```

Then point `[paths] dataset` to that file. Archive normalization, sampling
interval, color channels, and the chosen recordings are part of the experiment
definition; record them with the dataset preparation metadata.
