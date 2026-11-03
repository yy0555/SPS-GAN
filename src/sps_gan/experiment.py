"""Shared training, checkpointing, and sampling for SPS-GAN observations."""

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .configuration import _validate
from .dataset import TrajectoryDataset
from .models import HNNPhaseSpace, TrajectoryDiscriminator, TrajectoryGenerator
from .updates import update_Dv, update_G

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, config, *, training=True):
        _validate(config)
        self.config = config
        d, m, t = config.data, config.model, config.training
        self.mode = d.mode
        self.batch_size = config.video.batch_size if self.mode == "video" else t.batch_size
        self.content_dim = config.video.content_dim if self.mode == "video" else m.content_dim
        self.discriminator_frames = d.discriminator_frames or d.num_frames
        device = t.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; select device = cpu")
        self.device = torch.device(device)
        # Seed before constructing any networks or drawing a dataset permutation.
        torch.manual_seed(t.seed)
        np.random.seed(t.seed)
        self.dataset = None
        if training:
            if self.mode == "video":
                from .video_dataset import VideoDataset

                self.dataset = VideoDataset(
                    config.paths.dataset, d.num_frames, config.video.image_size,
                    config.video.channels, d.conditioning_dim,
                )
            else:
                self.dataset = TrajectoryDataset(
                    config.paths.dataset, d.num_frames, d.max_particles, d.conditioning_dim
                )
        self.rnn = HNNPhaseSpace(
            device=self.device, input_size=(m.noise_dim or m.latent_dim) + d.conditioning_dim,
            hidden_size=m.hidden_dim, output_size=m.latent_dim, dt=d.dt,
            ndim_label=d.conditioning_dim,
        )
        self.rnn.initWeight()
        self.Di = None
        if self.mode == "video":
            from .video_models import ImageGenerator, ImageDiscriminator, VideoDiscriminator

            v = config.video
            self.Gi = ImageGenerator(
                m.latent_dim + self.content_dim + d.conditioning_dim,
                channels=v.channels, base_filters=v.generator_filters,
            ).to(self.device)
            self.Di = ImageDiscriminator(
                channels=v.channels, base_filters=v.discriminator_filters,
                conditioning_dim=d.conditioning_dim,
            ).to(self.device)
            self.Dv = VideoDiscriminator(
                channels=v.channels, base_filters=v.discriminator_filters,
                conditioning_dim=d.conditioning_dim, frames=self.discriminator_frames,
            ).to(self.device)
        else:
            self.Gi = TrajectoryGenerator(
                nz=m.latent_dim + self.content_dim + d.conditioning_dim,
                hidden_dim=m.decoder_hidden_dim, traj_len=d.num_frames, n_particles=d.max_particles,
            ).to(self.device)
            self.Dv = TrajectoryDiscriminator(
                d.conditioning_dim, hidden_dim=m.discriminator_hidden_dim,
                max_n=d.max_particles, minibatch_disc=m.minibatch_discrimination,
            ).to(self.device)
        self.optimizers = {
            name: torch.optim.Adam(model.parameters(), lr=t.learning_rate, betas=(t.beta1, t.beta2))
            for name, model in self.models.items()
        }
        self.step = 0
        self._order = torch.empty(0, dtype=torch.long)
        self._cursor = 0
        for model in self.models.values():
            model.train(training)

    @property
    def models(self):
        models = {"rnn": self.rnn, "Gi": self.Gi, "Dv": self.Dv}
        if self.Di is not None:
            models["Di"] = self.Di
        return models

    def next_batch(self):
        if self.dataset is None:
            raise ValueError("This experiment was opened for sampling only")
        if self._cursor >= len(self._order):
            self._order = torch.randperm(len(self.dataset))
            self._cursor = 0
        indices = self._order[self._cursor:self._cursor + self.batch_size]
        self._cursor += len(indices)
        trajectory, mask, condition = default_collate([self.dataset[int(i)] for i in indices])
        return {
            "videos": trajectory.to(self.device), "mask": mask.to(self.device),
            "label_and_props": condition.to(self.device),
        }

    def generate(self, condition, mask=None, *, frames=None):
        """Generate a batch, retaining derivatives required by the cyclic loss."""
        d, m = self.config.data, self.config.model
        frames = d.num_frames if frames is None else frames
        batch = condition.shape[0]
        if condition.shape != (batch, d.conditioning_dim):
            raise ValueError("Conditioning shape does not match configuration")
        if self.mode == "trajectory" and (mask is None or mask.shape != (batch, d.max_particles)):
            raise ValueError("Particle mask shape does not match configuration")
        noise = torch.randn(batch, m.noise_dim or m.latent_dim, device=self.device)
        states, derivatives = self.rnn(torch.cat((condition, noise), dim=1), frames)
        states = states.transpose(0, 1)
        content = torch.randn(batch, 1, self.content_dim, device=self.device).expand(-1, frames, -1)
        decoded_input = torch.cat((states, content, condition[:, None].expand(-1, frames, -1)), dim=-1)
        if self.mode == "video":
            images = self.Gi(decoded_input.reshape(batch * frames, -1, 1, 1))
            observations = images.reshape(batch, frames, *images.shape[1:]).permute(0, 2, 1, 3, 4)
        else:
            observations = self.Gi(decoded_input, mask)
        return {
            "videos": observations, "latent": states,
            "dlatent": derivatives.transpose(0, 1) if derivatives is not None else None,
        }

    def train_step(self):
        for model in self.models.values():
            model.train()
        real = self.next_batch()
        if self.mode == "video":
            losses = self._video_step(real)
        else:
            losses = self._trajectory_step(real)
        if not all(np.isfinite(value) for value in losses.values()):
            raise FloatingPointError("Training produced a non-finite loss")
        self.step += 1
        return losses

    def _trajectory_step(self, real):
        condition, mask = real["label_and_props"], real["mask"]
        t = self.config.training
        fake = self.generate(condition, mask)
        losses, _ = update_Dv(
            dis_v=self.Dv, real_data=real, fake_data=fake,
            optim_Dv=self.optimizers["Dv"], r1_gamma=t.r1_gamma, gamma=t.discriminator_margin,
            discriminator_frames=self.config.data.discriminator_frames,
        )
        for _ in range(t.generator_updates):
            fake = self.generate(condition, mask)
            losses.update(update_G(
                q_size=self.config.model.latent_dim // 2, cyclic_coord_loss=t.cyclic_weight,
                model_dv=self.Dv, model_gi=self.Gi, model_rnn=self.rnn, fake_data=fake,
                optim_Gi=self.optimizers["Gi"], optim_RNN=self.optimizers["rnn"],
                label_props_colors=condition, video_mask=mask,
                discriminator_frames=self.config.data.discriminator_frames,
            ))
        return losses

    def _video_step(self, real):
        from .video_updates import update_video_discriminators, update_video_generator

        condition = real["label_and_props"]
        t, v = self.config.training, self.config.video
        # Discriminator updates need generated pixels, but no decoder graph.
        # The Hamiltonian enables autograd locally for its equations of motion.
        with torch.no_grad():
            fake = self.generate(condition)
        losses = update_video_discriminators(
            image_discriminator=self.Di, video_discriminator=self.Dv,
            real_videos=real["videos"], fake_videos=fake["videos"], conditioning=condition,
            optim_image=self.optimizers["Di"], optim_video=self.optimizers["Dv"],
            frames=self.discriminator_frames, r1_gamma=t.r1_gamma, real_label=v.real_label,
        )
        del fake
        for _ in range(t.generator_updates):
            fake = self.generate(condition)
            losses.update(update_video_generator(
                image_discriminator=self.Di, video_discriminator=self.Dv,
                generator=self.Gi, dynamics=self.rnn, fake_data=fake, conditioning=condition,
                optim_generator=self.optimizers["Gi"], optim_dynamics=self.optimizers["rnn"],
                frames=self.discriminator_frames, q_size=self.config.model.latent_dim // 2,
                cyclic_weight=t.cyclic_weight, target_label=v.generator_label,
            ))
        return losses

    def save_checkpoint(self, path):
        """Atomically save weights, optimizer state, and the batch/RNG position."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 2, "step": self.step, "config": self.config.as_dict(),
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "order": self._order, "cursor": self._cursor,
            "dataset_size": len(self.dataset) if self.dataset is not None else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path, *, resume=False):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state.get("format_version") not in (1, 2):
            raise ValueError("Unsupported SPS-GAN checkpoint format")
        saved_data = state["config"]["data"]
        saved_data.setdefault("mode", "trajectory")
        for section in ("model", "data"):
            if state["config"][section] != asdict(getattr(self.config, section)):
                raise ValueError(f"Checkpoint {section} settings do not match the configuration")
        if self.mode == "video":
            architecture = ("image_size", "channels", "content_dim", "generator_filters", "discriminator_filters")
            saved_video = state["config"].get("video", {})
            if any(saved_video.get(key) != getattr(self.config.video, key) for key in architecture):
                raise ValueError("Checkpoint video architecture does not match the configuration")
        for name, model in self.models.items():
            model.load_state_dict(state["models"][name])
        self.step = state["step"]
        if resume:
            if self.dataset is None or state["dataset_size"] != len(self.dataset):
                raise ValueError("Resume requires the original dataset with the same sample count")
            for name, optimizer in self.optimizers.items():
                optimizer.load_state_dict(state["optimizers"][name])
                # Explicitly allow learning-rate/beta changes in the resumed config.
                for group in optimizer.param_groups:
                    group["lr"] = self.config.training.learning_rate
                    group["betas"] = (self.config.training.beta1, self.config.training.beta2)
            self._order = state["order"].cpu()
            self._cursor = state["cursor"]
            torch.set_rng_state(state["rng"].cpu())
            if self.device.type == "cuda" and state["cuda_rng"]:
                torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])

    def train(self, *, resume=False):
        output = self.config.paths.output
        existing = sorted(output.glob("checkpoint_*.pt"))
        if resume:
            if not existing:
                raise ValueError(f"No checkpoint to resume in {output}")
            self.load_checkpoint(existing[-1], resume=True)
        elif existing:
            raise ValueError(f"Checkpoints already exist in {output}; use --resume or another output path")
        output.mkdir(parents=True, exist_ok=True)
        self.config.save(output / "configuration.ini")
        t = self.config.training
        handler = logging.FileHandler(output / "training.log")
        logger.addHandler(handler)
        try:
            while self.step < t.steps:
                losses = self.train_step()
                if self.step % t.log_every == 0 or self.step == t.steps:
                    metrics = " ".join(f"{key}={value:.6f}" for key, value in losses.items())
                    logger.info("step %d/%d %s", self.step, t.steps, metrics)
                if self.step % t.save_every == 0 or self.step == t.steps:
                    self.save_checkpoint(output / f"checkpoint_{self.step:08d}.pt")
        finally:
            logger.removeHandler(handler)
            handler.close()
