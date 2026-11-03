"""Video-mode training, sampling, checkpoint, and configuration checks."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sps_gan.configuration import Config, Data, Model, Paths, Training, Video, load_config
from sps_gan.experiment import Experiment
from sps_gan.video_dataset import VideoDataset
from sps_gan.video_models import ImageDiscriminator, ImageGenerator, VideoDiscriminator


class VideoExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset_path = self.root / "videos.npz"
        # Extra source frames exercise random cropping when training resumes.
        np.savez(
            self.dataset_path,
            videos=np.random.default_rng(37).integers(
                0, 256, size=(3, 18, 1, 96, 96), dtype=np.uint8,
            ),
        )
        self.config = Config(
            paths=Paths(self.dataset_path, self.root / "output"),
            data=Data(mode="video", num_frames=16, discriminator_frames=16),
            model=Model(latent_dim=4, noise_dim=3, hidden_dim=8, content_dim=0,
                        decoder_hidden_dim=8, discriminator_hidden_dim=8),
            training=Training(seed=29, steps=2, batch_size=7, learning_rate=1e-3,
                              r1_gamma=0.1, cyclic_weight=0.01, log_every=1, save_every=1),
            video=Video(image_size=96, channels=1, content_dim=2, generator_filters=1,
                        discriminator_filters=1, batch_size=2),
        )

    def states(self, experiment):
        # state_dict includes BatchNorm running statistics and update counters.
        return {
            name: {key: value.detach().clone() for key, value in model.state_dict().items()}
            for name, model in experiment.models.items()
        }

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_video_mode_routes_models_and_trains_short_final_batch(self):
        experiment = Experiment(self.config)
        self.assertEqual(set(experiment.models), {"rnn", "Gi", "Di", "Dv"})
        self.assertIsInstance(experiment.dataset, VideoDataset)
        self.assertIsInstance(experiment.Gi, ImageGenerator)
        self.assertIsInstance(experiment.Di, ImageDiscriminator)
        self.assertIsInstance(experiment.Dv, VideoDiscriminator)
        self.assertEqual(experiment.batch_size, 2)  # Video profile overrides training.batch_size=7.
        self.assertEqual(experiment.content_dim, 2)
        self.assertEqual(experiment.Gi.nz, 6)
        before = self.states(experiment)
        batch_shapes = []
        decoder_graphs = []
        hook = experiment.Gi.register_forward_hook(
            lambda module, inputs, output: decoder_graphs.append(output.requires_grad)
        )
        self.addCleanup(hook.remove)
        next_batch = experiment.next_batch

        def record_batch():
            batch = next_batch()
            batch_shapes.append(tuple(batch["videos"].shape))
            self.assertEqual(batch["mask"].shape, (batch["videos"].shape[0], 0))
            return batch

        experiment.next_batch = record_batch
        for _ in range(2):
            losses = experiment.train_step()
            self.assertTrue(all(np.isfinite(value) for value in losses.values()))
            self.assertTrue({"Di", "Dv", "Gi", "Gv", "cyclic", "G_total", "Di_r1", "Dv_r1"} <= losses.keys())
            self.assertGreaterEqual(losses["Di_r1"], 0)
            self.assertGreaterEqual(losses["Dv_r1"], 0)
            self.assertGreaterEqual(losses["cyclic"], 0)
            self.assertAlmostEqual(losses["G_total"], losses["Gi"] + losses["Gv"] + losses["cyclic"], places=5)
        self.assertEqual(batch_shapes, [(2, 1, 16, 96, 96), (1, 1, 16, 96, 96)])
        self.assertEqual(decoder_graphs, [False, True, False, True])
        for name, model in experiment.models.items():
            self.assertTrue(any(
                not torch.equal(before[name][key], value)
                for key, value in model.named_parameters()
            ), name)

    def test_checkpoint_resumes_losses_weights_batchnorm_and_rng_exactly(self):
        uninterrupted = Experiment(self.config)
        uninterrupted.train_step()
        checkpoint = self.root / "resume.pt"
        uninterrupted.save_checkpoint(checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        expected_losses = [uninterrupted.train_step(), uninterrupted.train_step()]
        expected_states = self.states(uninterrupted)
        expected_rng = torch.get_rng_state().clone()

        resumed = Experiment(self.config)
        resumed.load_checkpoint(checkpoint, resume=True)
        self.assertEqual(resumed.step, 1)
        self.assertEqual(resumed._cursor, 2)
        for name, optimizer in resumed.optimizers.items():
            self.assert_nested_equal(optimizer.state_dict(), saved["optimizers"][name])
        actual_losses = [resumed.train_step(), resumed.train_step()]
        self.assert_nested_equal(actual_losses, expected_losses)
        self.assert_nested_equal(self.states(resumed), expected_states)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_training_writes_checkpoint_and_sampling_needs_no_dataset(self):
        from sps_gan.analysis import initial_coordinate_pca

        trained = Experiment(replace(self.config, training=replace(self.config.training, steps=1)))
        trained.train()
        checkpoint = self.config.paths.output / "checkpoint_00000001.pt"
        self.assertTrue(checkpoint.is_file())
        self.assertTrue((self.config.paths.output / "training.log").is_file())
        self.assertEqual(load_config(self.config.paths.output / "configuration.ini"), trained.config)
        self.dataset_path.unlink()
        sample = Experiment(self.config, training=False)
        sample.load_checkpoint(checkpoint)
        before = self.states(sample)
        with torch.no_grad():
            result = sample.generate(torch.empty(2, 0), frames=5)
        self.assertEqual(result["videos"].shape, (2, 1, 5, 96, 96))
        self.assertTrue(torch.isfinite(result["videos"]).all())
        self.assertFalse(result["videos"].requires_grad)
        self.assertGreaterEqual(result["videos"].min().item(), -1)
        self.assertLessEqual(result["videos"].max().item(), 1)
        # Evaluation-mode sampling must not update BatchNorm running statistics.
        self.assert_nested_equal(self.states(sample), before)
        report = initial_coordinate_pca(sample, torch.empty(1, 0), samples=16)
        self.assertEqual(report["q_dimension"], 2)
        self.assertTrue(np.isfinite(report["explained_variance_ratio"]).all())

    def test_cross_mode_and_video_architecture_checkpoints_are_rejected(self):
        video = Experiment(self.config)
        checkpoint = self.root / "video.pt"
        video.save_checkpoint(checkpoint)
        trajectory = replace(self.config, data=replace(self.config.data, mode="trajectory"))
        with self.assertRaisesRegex(ValueError, "data settings"):
            Experiment(trajectory, training=False).load_checkpoint(checkpoint)
        changed = replace(self.config, video=replace(self.config.video, generator_filters=2))
        with self.assertRaisesRegex(ValueError, "video architecture"):
            Experiment(changed, training=False).load_checkpoint(checkpoint)

    def test_conditional_video_uses_supplied_vectors(self):
        with np.load(self.dataset_path, allow_pickle=False) as data:
            videos = data["videos"]
        np.savez(self.dataset_path, videos=videos,
                 conditioning=np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32))
        config = replace(self.config, data=replace(self.config.data, conditioning_dim=2))
        experiment = Experiment(config)
        self.assertEqual(experiment.rnn.phase_space_map.linear1.in_features, 5)
        self.assertEqual(experiment.Gi.nz, 8)
        self.assertEqual(experiment.Di.conditioning_dim, 2)
        self.assertEqual(experiment.Dv.conditioning_dim, 2)
        losses = experiment.train_step()
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))


class VideoConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.ini"
        self.text = """[paths]
dataset = videos.npz
output = output
[data]
mode = video
num_frames = 16
discriminator_frames = 16
[model]
latent_dim = 4
noise_dim = 3
[training]
steps = 2
[video]
channels = 1
generator_filters = 1
discriminator_filters = 1
content_dim = 2
batch_size = 2
"""

    def load(self, text):
        self.path.write_text(text)
        return load_config(self.path)

    def test_video_section_is_optional_for_trajectory_configuration(self):
        text = self.text.split("[video]")[0].replace("mode = video\n", "")
        config = self.load(text)
        self.assertEqual(config.data.mode, "trajectory")
        self.assertEqual(config.video, Video())
        self.assertEqual(self.load(self.text).video.channels, 1)

    def test_invalid_video_settings_are_rejected(self):
        cases = [
            self.text.replace("mode = video", "mode = images"),
            self.text.replace("num_frames = 16", "num_frames = 8").replace("discriminator_frames = 16", "discriminator_frames = 8"),
            self.text.replace("discriminator_frames = 16", "discriminator_frames = 8"),
            self.text.replace("channels = 1", "channels = 2"),
            self.text.replace("channels = 1", "channels = 1\nimage_size = 32"),
            self.text.replace("generator_filters = 1", "generator_filters = 0"),
            self.text.replace("content_dim = 2", "content_dim = -1"),
            self.text.replace("batch_size = 2", "batch_size = 0"),
            self.text + "real_label = 1.5\n",
            self.text + "generator_label = nan\n",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)


if __name__ == "__main__":
    unittest.main()
