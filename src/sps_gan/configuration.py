"""Explicit, validated experiment settings; importing the package reads no files."""

import configparser
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    dataset: Path
    output: Path


@dataclass(frozen=True)
class Data:
    num_frames: int = 30
    discriminator_frames: int = 0
    max_particles: int = 1
    conditioning_dim: int = 0
    dt: float = 0.05
    mode: str = "trajectory"


@dataclass(frozen=True)
class Model:
    latent_dim: int = 20
    noise_dim: int = 0
    content_dim: int = 0
    hidden_dim: int = 100
    decoder_hidden_dim: int = 512
    discriminator_hidden_dim: int = 218
    minibatch_discrimination: bool = False


@dataclass(frozen=True)
class Training:
    device: str = "cpu"
    seed: int = 72
    steps: int = 20000
    batch_size: int = 128
    learning_rate: float = 0.00005
    beta1: float = 0.3
    beta2: float = 0.999
    r1_gamma: float = 10.0
    cyclic_weight: float = 0.03
    discriminator_margin: float = 1.0
    generator_updates: int = 1
    log_every: int = 10
    save_every: int = 100


@dataclass(frozen=True)
class Video:
    image_size: int = 96
    channels: int = 3
    content_dim: int = 50
    generator_filters: int = 32
    discriminator_filters: int = 32
    batch_size: int = 16
    real_label: float = 0.9
    generator_label: float = 1.0


@dataclass(frozen=True)
class Config:
    paths: Paths
    data: Data
    model: Model
    training: Training
    video: Video = field(default_factory=Video)

    def as_dict(self):
        result = asdict(self)
        result["paths"] = {k: str(v) for k, v in result["paths"].items()}
        return result

    def save(self, path):
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_dict(self.as_dict())
        with Path(path).open("w") as stream:
            parser.write(stream)


def load_config(path):
    """Read an INI file, resolving paths relative to the configuration file."""
    path = Path(path).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    with path.open() as stream:
        parser.read_file(stream)
    types = {"paths": Paths, "data": Data, "model": Model, "training": Training, "video": Video}
    if parser.defaults():
        raise ValueError("[DEFAULT] is not supported; use the named configuration sections")
    required = {"paths", "data", "model", "training"}
    if not required <= set(parser.sections()) or set(parser.sections()) - set(types):
        raise ValueError("Configuration requires [paths], [data], [model], [training]; [video] is optional")
    sections = {}
    for name, cls in types.items():
        if name not in parser:
            sections[name] = cls()
            continue
        fields = cls.__dataclass_fields__
        unknown = set(parser[name]) - set(fields)
        if unknown:
            raise ValueError(f"Unknown [{name}] settings: {', '.join(sorted(unknown))}")
        values = {}
        for key, raw in parser[name].items():
            kind = fields[key].type
            try:
                if kind is Path:
                    if not raw.strip():
                        raise ValueError("path cannot be empty")
                    value = Path(raw).expanduser()
                    values[key] = (path.parent / value).resolve()
                elif kind is bool:
                    values[key] = parser[name].getboolean(key)
                else:
                    values[key] = kind(raw)
            except ValueError as error:
                raise ValueError(f"Invalid [{name}] {key}: {error}") from error
        if name == "paths" and set(values) != set(fields):
            raise ValueError("[paths] requires dataset and output")
        sections[name] = cls(**values)
    config = Config(**sections)
    _validate(config)
    return config


def _validate(config):
    d, m, t, v = config.data, config.model, config.training, config.video
    if d.mode not in ("trajectory", "video"):
        raise ValueError("mode must be trajectory or video")
    for section in (d, m, t, v):
        for name, value in asdict(section).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
    for name, value in {
        "num_frames": d.num_frames, "max_particles": d.max_particles, "dt": d.dt,
        "latent_dim": m.latent_dim, "hidden_dim": m.hidden_dim,
        "decoder_hidden_dim": m.decoder_hidden_dim,
        "discriminator_hidden_dim": m.discriminator_hidden_dim,
        "steps": t.steps, "batch_size": t.batch_size, "learning_rate": t.learning_rate,
        "generator_updates": t.generator_updates, "log_every": t.log_every,
        "save_every": t.save_every, "discriminator_margin": t.discriminator_margin,
        "video batch_size": v.batch_size, "generator_filters": v.generator_filters,
        "discriminator_filters": v.discriminator_filters,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if d.num_frames < 2 or m.latent_dim % 2:
        raise ValueError("num_frames must be >= 2 and latent_dim must be even")
    if d.discriminator_frames != 0 and not 2 <= d.discriminator_frames <= d.num_frames:
        raise ValueError("discriminator_frames must be 0 or lie between 2 and num_frames")
    if min(d.conditioning_dim, m.noise_dim, m.content_dim, v.content_dim, t.r1_gamma, t.cyclic_weight, t.seed) < 0:
        raise ValueError("Dimensions, regularization weights, and seed must be nonnegative")
    if t.seed >= 2**32:
        raise ValueError("seed must be smaller than 2**32")
    if not (0 <= t.beta1 < 1 and 0 <= t.beta2 < 1):
        raise ValueError("Adam betas must lie in [0, 1)")
    if v.image_size != 96 or v.channels not in (1, 3):
        raise ValueError("Video networks require image_size = 96 and channels = 1 or 3")
    if not (0 < v.real_label <= 1 and 0 < v.generator_label <= 1):
        raise ValueError("Video BCE labels must lie in (0, 1]")
    if d.mode == "video" and (d.discriminator_frames or d.num_frames) < 16:
        raise ValueError("Video discrimination requires at least 16 frames")
    if t.device not in ("cpu", "auto", "cuda") and not (
        t.device.startswith("cuda:") and t.device[5:].isdigit()
    ):
        raise ValueError("device must be cpu, auto, cuda, or cuda:<index>")
