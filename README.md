# Structured Phase Space GAN

Structured Phase Space GAN (SPS-GAN) learns latent Hamiltonian dynamics from
Cartesian trajectories or videos. A learned map initializes phase space and a
Hamiltonian network evolves it. The decoder generates positions or image frames;
adversarial training and cyclic-coordinate regularization encourage realistic
observations and compact latent structure.

Code for [Discovering Phase Space Structure in Learned Hamiltonian Systems](https://arxiv.org/abs/2509.23003)
(NeurIPS 2026), by Yulong Yang, Jiayin Liu, Vineet Bansal, and Christine
Allen-Blanchette. This package provides conservative trajectory and video
training, sampling, and latent-coordinate analysis.

## Layout

```text
src/
  sps_gan/          # Models, losses, data interface, training, sampling, PCA
dataset/
  README.md         # Paper data sources and preparation format; no bundled data
config/
  example.ini       # The single experiment template
AGENTS.md
CLAUDE.md           # @AGENTS.md
README.md
LICENSE             # MIT
.gitignore
pyproject.toml
python.version
uv.lock
```

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11,
then run from the repository root:

```bash
uv sync --python 3.11 --locked
uv run --python 3.11 sps-gan --help
```

`python.version` records the intended Python version. Pass `--python 3.11`
explicitly because uv's automatically discovered filename is `.python-version`.
The lockfile fixes the runtime environment, including PyTorch 2.7.1 and NumPy
1.26.4. CPU is the example's default device. Select `cuda` or `cuda:0` in the
configuration to use a compatible GPU, or `auto` to select CUDA when available.

## Prepare data and train

Follow [dataset/README.md](dataset/README.md) for data sources and preparation.
Select the observation type in `config/example.ini`:

```ini
[data]
mode = trajectory
```

Set `mode = video` for video training. This switches the dataset loader,
decoder, discriminators, objective, and sample output together.

| Mode | NPZ input | Decoder and discriminators | Adversarial objective |
| --- | --- | --- | --- |
| `trajectory` | `trajectories[N,T,2,particles]` | Position MLP and recurrent trajectory discriminator | Hinge loss |
| `video` | `videos[N,T,C,96,96]` | PixelShuffle image decoder, image discriminator, and video discriminator | Image + video BCE |

Both modes use the same Hamiltonian dynamics, R1 regularization, and full-sequence
cyclic-coordinate loss. Optional `conditioning[N,conditioning_dim]` vectors are
supplied to the phase-space map, Hamiltonian, decoder, and discriminators. In
trajectory mode, an optional particle `mask` supports padded systems. No data,
simulators, baselines, or pretrained weights are distributed here.

Edit `config/example.ini` to point to your prepared archive and choose an output
directory. Use separate output directories for trajectory and video runs. All
INI paths resolve relative to the INI file, regardless of the working directory.
Adjust the particle count or video channels, conditioning width, and timestep to
match the data. Trajectory coordinates are used without normalization. Video
inputs are normalized to `[-1,1]` to match the decoder's Tanh output.

The template lists every supported INI setting. `noise_dim` controls the input
motion code (`0` uses `latent_dim`); `latent_dim` is the full phase-space dimension, split equally into
position and momentum coordinates. `content_dim = 0` disables the independent
content code for trajectory training. The discriminator evaluates contiguous
windows of `discriminator_frames` timesteps; use `0` for the full training window.
The cyclic loss always uses the full generated sequence. Video discriminator
windows must contain at least 16 frames.

Network depths and activations are fixed: two hidden layers with ReLU in the
phase-space map and Hamiltonian, three hidden layers with Softplus in the decoder,
and a two-layer bidirectional GRU trajectory discriminator. Video mode uses
96×96 RGB or grayscale frames, a convolutional PixelShuffle decoder, and 2D/3D
convolutional discriminators. Its content code, batch size, convolution widths,
channels, and BCE targets are configured under `[video]`. In video mode,
`[video] batch_size` and `content_dim` replace their trajectory counterparts;
optimizer settings, training steps, latent dimensions, and regularization
coefficients remain shared. `discriminator_margin` applies only to trajectories.

```bash
uv run --python 3.11 sps-gan check-config --config config/example.ini
uv run --python 3.11 sps-gan train --config config/example.ini
```

`check-config` checks settings and displays resolved paths; training also checks
that the dataset exists and has valid arrays. Each training step updates the
mode's discriminator(s), then performs the configured number of generator updates. The final
partial batch is retained. Longer input sequences are cropped to random
contiguous windows. Prepare training/evaluation splits externally before making
overlapping windows.

Training creates a resolved configuration, a log, and checkpoints under the
configured output directory. The default `output/` directory is ignored by Git;
add an ignore rule if you choose a different location. Start a new run with a new
output directory. Resume the latest checkpoint with:

```bash
uv run --python 3.11 sps-gan train --config config/example.ini --resume
```

Checkpoints contain model/optimizer state, the training step, batch order, and
PyTorch RNG state. Resume with the same dataset and model/data settings; `steps`
is the total target step count. Increasing it extends training. Learning rate and
Adam betas may be changed on resume. Reproducibility across different hardware
and PyTorch versions is not guaranteed.

## Prepare a video recording

Set `mode = video`, select `channels = 1` or `3` under `[video]`, and point
`[paths] dataset` to the desired prepared archive. Convert a local recording with:

```bash
uv run --python 3.11 sps-gan prepare-video --config config/example.ini \
  --input /path/to/recording.mp4 --stride 1
```

This resizes frames to 96×96 and creates nonoverlapping clips. Set `[data] dt` to
the reported `dt` (stride divided by the source frame rate), then use the same
`train` command above. `--max-clips` can limit preparation memory use. Preparation
and training hold their selected dataset in memory. The preparation command
creates unconditional data; add conditioning vectors externally for conditional
experiments. Split recordings into training/evaluation sets before preparation.

## Generate observations and inspect phase space

After a complete example run, use its final checkpoint:

```bash
uv run --python 3.11 sps-gan sample --config config/example.ini \
  --checkpoint output/pendulum/checkpoint_00020000.pt \
  --samples 128 --frames 200 --output output/pendulum/samples.npz

uv run --python 3.11 sps-gan pca --config config/example.ini \
  --checkpoint output/pendulum/checkpoint_00020000.pt \
  --samples 1024 --output output/pendulum/pca.json
```

Sampling needs the configuration and checkpoint, but does not load the training
dataset. For conditional models, supply `--conditioning 1,0,0` with exactly the
configured number of values, using the same encoding as training. Use
`--particles` to limit active particles in a padded model. Generated archives
use the input dataset format. CLI output paths resolve from the working directory.

In video mode, sample one recording as MP4:

```bash
uv run --python 3.11 sps-gan sample --config config/example.ini \
  --checkpoint output/video/checkpoint_00020000.pt \
  --samples 1 --frames 60 --output output/video/sample.mp4
```

Use your configured output directory for the checkpoint path. For multiple video
samples, choose an `.npz` output; it stores uint8 `videos[N,T,C,96,96]` and
conditioning vectors. MP4 playback defaults to `1 / dt` frames per second;
`--fps` overrides playback speed. The sample-count defaults are one video or 128
trajectories. `--particles` applies only to trajectory sampling.

PCA reports explained variance for initial latent position coordinates, retaining
at most 64 components and explicitly reporting whether the requested variance
threshold was reached. This is an exploratory diagnostic, not a physical
degrees-of-freedom estimate or a benchmark metric.

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Yulong Yang, Jiayin Liu, Vineet Bansal, and Christine
Allen-Blanchette.

External datasets retain their own terms and citations, linked in
[dataset/README.md](dataset/README.md).

## Citation

If you use SPS-GAN in your research, please cite
[our paper](https://arxiv.org/abs/2509.23003):

```bibtex
@inproceedings{yang2026discovering,
  title     = {Discovering Phase Space Structure in Learned Hamiltonian Systems},
  author    = {Yang, Yulong and Liu, Jiayin and Bansal, Vineet and Allen-Blanchette, Christine},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2509.23003}
}
```
