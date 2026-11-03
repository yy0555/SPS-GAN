"""End-to-end CPU training and resume checks with disposable synthetic inputs."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sps_gan.configuration import Config, Data, Model, Paths, Training, load_config
from sps_gan.experiment import Experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset_path = self.root / "trajectories.npz"
        # Extra frames exercise stochastic cropping as well as shuffled batches.
        np.savez(
            self.dataset_path,
            trajectories=np.random.default_rng(19).normal(size=(3, 5, 2, 1)).astype(np.float32),
        )
        self.config = Config(
            paths=Paths(self.dataset_path, self.root / "output"),
            data=Data(num_frames=3, max_particles=1),
            model=Model(
                latent_dim=4, content_dim=2, hidden_dim=8,
                decoder_hidden_dim=8, discriminator_hidden_dim=8,
            ),
            training=Training(
                seed=17, steps=3, batch_size=2, learning_rate=1e-3,
                r1_gamma=1.0, cyclic_weight=0.03, log_every=1, save_every=1,
            ),
        )

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assert_nested_equal(actual_item, expected_item)
        else:
            self.assertEqual(actual, expected)

    def weights(self, experiment):
        return {
            name: {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            for name, model in experiment.models.items()
        }

    def test_seed_controls_model_initialization(self):
        first = self.weights(Experiment(self.config))
        torch.rand(123)  # Construction must not depend on the caller's RNG position.
        second = self.weights(Experiment(self.config))
        self.assert_nested_equal(second, first)
        changed = replace(self.config, training=replace(self.config.training, seed=18))
        third = self.weights(Experiment(changed))
        self.assertTrue(any(
            not torch.equal(first[name][key], third[name][key])
            for name in first for key in first[name]
        ))

    def test_training_handles_short_batch_and_writes_resumable_checkpoints(self):
        experiment = Experiment(self.config)
        initial = self.weights(experiment)
        batch_sizes = []
        original_next_batch = experiment.next_batch

        def tracked_batch():
            batch = original_next_batch()
            batch_sizes.append(len(batch["videos"]))
            return batch

        experiment.next_batch = tracked_batch
        experiment.train()
        self.assertEqual(batch_sizes, [2, 1, 2])
        self.assertEqual(experiment.step, 3)
        for name, model in experiment.models.items():
            self.assertTrue(any(
                not torch.equal(initial[name][key], value)
                for key, value in model.state_dict().items()
            ), name)
        output = self.config.paths.output
        self.assertEqual(len(list(output.glob("checkpoint_*.pt"))), 3)
        self.assertTrue((output / "training.log").is_file())
        self.assertEqual(load_config(output / "configuration.ini"), self.config)
        self.assertFalse(list(output.glob("*.tmp")))
        with self.assertRaisesRegex(ValueError, "already exist"):
            Experiment(self.config).train()
        # Resuming a completed run must not execute another update.
        resumed = Experiment(self.config)
        resumed.train(resume=True)
        self.assertEqual(resumed.step, 3)
        self.assert_nested_equal(self.weights(resumed), self.weights(experiment))

    def test_checkpoint_restores_optimizer_batch_order_and_rng(self):
        uninterrupted = Experiment(self.config)
        uninterrupted.train_step()
        saved_order = uninterrupted._order.clone()
        checkpoint = self.root / "resume.pt"
        uninterrupted.save_checkpoint(checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        expected_losses = [uninterrupted.train_step(), uninterrupted.train_step()]
        expected_weights = self.weights(uninterrupted)
        expected_rng = torch.get_rng_state().clone()

        resumed = Experiment(self.config)
        resumed.load_checkpoint(checkpoint, resume=True)
        self.assertEqual(resumed.step, 1)
        self.assertEqual(resumed._cursor, 2)
        torch.testing.assert_close(resumed._order, saved_order, rtol=0, atol=0)
        for name, optimizer in resumed.optimizers.items():
            self.assert_nested_equal(optimizer.state_dict(), saved["optimizers"][name])
        # Step 2 uses the final single sample; step 3 draws a new permutation.
        actual_losses = [resumed.train_step(), resumed.train_step()]
        self.assert_nested_equal(actual_losses, expected_losses)
        self.assert_nested_equal(self.weights(resumed), expected_weights)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_sampling_checkpoint_does_not_require_training_dataset(self):
        trained = Experiment(self.config)
        trained.train_step()
        checkpoint = self.root / "sample.pt"
        trained.save_checkpoint(checkpoint)
        self.dataset_path.unlink()
        sample = Experiment(self.config, training=False)
        sample.load_checkpoint(checkpoint)
        self.assertIsNone(sample.dataset)
        with torch.no_grad():
            generated = sample.generate(torch.empty(2, 0), torch.ones(2, 1), frames=5)
        self.assertEqual(generated["videos"].shape, (2, 5, 2, 1))
        self.assertTrue(torch.isfinite(generated["videos"]).all())
        self.assertFalse(generated["videos"].requires_grad)
        with self.assertRaisesRegex(ValueError, "sampling only"):
            sample.next_batch()

    def test_independent_motion_noise_and_zero_content_train_sample_and_pca(self):
        from sps_gan.analysis import initial_coordinate_pca

        config = replace(
            self.config,
            model=replace(self.config.model, noise_dim=3, latent_dim=4, content_dim=0),
            data=replace(self.config.data, discriminator_frames=2),
        )
        experiment = Experiment(config)
        self.assertEqual(experiment.rnn.phase_space_map.linear1.in_features, 3)
        self.assertEqual(experiment.rnn.hnn.linear1.in_features, 4)
        self.assertEqual(experiment.Gi.main[0].in_features, 4)
        losses = experiment.train_step()
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        generated = experiment.generate(torch.empty(2, 0), torch.ones(2, 1), frames=5)
        self.assertEqual(generated["latent"].shape, (2, 5, 4))
        self.assertEqual(generated["videos"].shape, (2, 5, 2, 1))
        result = initial_coordinate_pca(experiment, torch.empty(1, 0), samples=16)
        self.assertEqual(result["q_dimension"], 2)
        self.assertTrue(np.isfinite(result["explained_variance_ratio"]).all())

    def test_incompatible_checkpoint_and_missing_resume_are_rejected(self):
        experiment = Experiment(self.config)
        with self.assertRaisesRegex(ValueError, "No checkpoint"):
            experiment.train(resume=True)
        checkpoint = self.root / "original.pt"
        experiment.save_checkpoint(checkpoint)
        changed = replace(self.config, data=replace(self.config.data, dt=0.1))
        with self.assertRaisesRegex(ValueError, "data settings"):
            Experiment(changed).load_checkpoint(checkpoint)
        np.savez(self.dataset_path, trajectories=np.ones((2, 5, 2, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "sample count"):
            Experiment(self.config).load_checkpoint(checkpoint, resume=True)

    def test_version_one_trajectory_checkpoint_loads_without_mode_fields(self):
        trained = Experiment(self.config)
        trained.train_step()
        checkpoint = self.root / "version_one.pt"
        trained.save_checkpoint(checkpoint)
        state = torch.load(checkpoint, weights_only=True)
        state["format_version"] = 1
        state["config"]["data"].pop("mode")
        state["config"].pop("video")
        torch.save(state, checkpoint)
        loaded = Experiment(self.config, training=False)
        loaded.load_checkpoint(checkpoint)
        self.assertEqual(loaded.mode, "trajectory")
        self.assert_nested_equal(self.weights(loaded), self.weights(trained))


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config_path = self.root / "config" / "example.ini"
        self.config_path.parent.mkdir()
        self.text = """[paths]
dataset = ../dataset/trajectories.npz
output = ../output/experiment
[data]
num_frames = 3
[model]
latent_dim = 4
minibatch_discrimination = false
[training]
steps = 2
"""

    def load(self, text):
        self.config_path.write_text(text)
        return load_config(self.config_path)

    def test_paths_are_relative_to_config_and_saved_values_round_trip(self):
        config = self.load(self.text)
        self.assertEqual(config.paths.dataset, self.root / "dataset" / "trajectories.npz")
        self.assertEqual(config.paths.output, self.root / "output" / "experiment")
        self.assertFalse(config.model.minibatch_discrimination)
        self.assertEqual(config.training.batch_size, 128)
        saved = self.root / "saved.ini"
        config.save(saved)
        self.assertEqual(load_config(saved), config)

    def test_invalid_configuration_is_rejected(self):
        cases = [
            self.text + "unknown_option = 1\n",
            self.text + "\n[unexpected]\nvalue = 1\n",
            self.text.replace("latent_dim = 4", "latent_dim = 3"),
            self.text.replace("num_frames = 3", "num_frames = 1"),
            self.text.replace("num_frames = 3", "num_frames = 3\ndiscriminator_frames = 4"),
            self.text.replace("num_frames = 3", "num_frames = 3\ndiscriminator_frames = -1"),
            self.text.replace("latent_dim = 4", "latent_dim = 4\nnoise_dim = -1"),
            self.text.replace("steps = 2", "steps = 0"),
            self.text.replace("steps = 2", "learning_rate = nan"),
            self.text.replace("steps = 2", "device = mps"),
            self.text.replace("steps = 2", "beta1 = 1"),
            self.text.replace("steps = 2", "seed = -1"),
            self.text.replace("steps = 2", "seed = 4294967296"),
            self.text.replace("minibatch_discrimination = false", "minibatch_discrimination = perhaps"),
            self.text.replace("dataset = ../dataset/trajectories.npz", "dataset ="),
            self.text.replace("output = ../output/experiment\n", ""),
            "[DEFAULT]\nsteps = 2\n" + self.text,
        ]
        for invalid in cases:
            with self.subTest(config=invalid):
                with self.assertRaises(ValueError):
                    self.load(invalid)


if __name__ == "__main__":
    unittest.main()
