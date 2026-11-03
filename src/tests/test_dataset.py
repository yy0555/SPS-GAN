"""Data-contract checks using temporary, synthetic arrays only."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sps_gan.dataset import TrajectoryDataset


class TrajectoryDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trajectories.npz"
        self.trajectories = np.arange(120, dtype=np.float64).reshape(3, 10, 2, 2)

    def save(self, **arrays):
        np.savez(self.path, **arrays)
        return self.path

    def test_padding_mask_conditioning_and_independent_items(self):
        self.save(
            trajectories=self.trajectories,
            mask=np.array([1, 0]),
            conditioning=np.array([[1, 2], [3, 4], [5, 6]]),
        )
        dataset = TrajectoryDataset(self.path, 10, 3, conditioning_dim=2)
        trajectory, mask, conditioning = dataset[1]
        self.assertEqual(len(dataset), 3)
        self.assertEqual(trajectory.shape, (10, 2, 3))
        self.assertEqual(trajectory.dtype, torch.float32)
        torch.testing.assert_close(mask, torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(conditioning, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(trajectory[:, :, 0], torch.tensor(self.trajectories[1, :, :, 0]).float())
        self.assertEqual(torch.count_nonzero(trajectory[:, :, 1:]).item(), 0)
        trajectory.zero_()
        conditioning.zero_()
        self.assertGreater(dataset[1][0].sum().item(), 0)
        torch.testing.assert_close(dataset[1][2], torch.tensor([3.0, 4.0]))

    def test_seeded_crop_is_contiguous_and_unconditional(self):
        self.save(trajectories=self.trajectories)
        dataset = TrajectoryDataset(self.path, 4, 2)
        torch.manual_seed(17)
        first = dataset[0]
        torch.manual_seed(17)
        second = dataset[0]
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[0][1:] - first[0][:-1], torch.full((3, 2, 2), 4.0))
        self.assertEqual(first[2].shape, (0,))
        torch.testing.assert_close(first[1], torch.ones(2))

    def test_sample_specific_masks(self):
        self.save(trajectories=self.trajectories, mask=np.array([[1, 0], [0, 1], [1, 1]]))
        dataset = TrajectoryDataset(self.path, 10, 2)
        self.assertEqual(dataset[0][0][:, :, 1].sum().item(), 0)
        self.assertEqual(dataset[1][0][:, :, 0].sum().item(), 0)

    def test_missing_dataset_has_preparation_guidance(self):
        with self.assertRaisesRegex(FileNotFoundError, "dataset/README.md"):
            TrajectoryDataset(self.path, 4, 2)

    def test_invalid_trajectory_shapes_and_values(self):
        cases = [
            np.zeros((2, 5, 2)),
            np.zeros((2, 5, 3, 2)),
            np.zeros((0, 5, 2, 2)),
            np.zeros((2, 5, 2, 0)),
            np.zeros((2, 3, 2, 2)),
            np.zeros((2, 5, 2, 3)),
            np.full((2, 5, 2, 2), np.nan),
            np.full((2, 5, 2, 2), np.inf),
            np.full((2, 5, 2, 2), 1e100),
            np.zeros((2, 5, 2, 2), dtype=complex),
            np.full((2, 5, 2, 2), "x"),
            np.empty((2, 5, 2, 2), dtype=object),
        ]
        for values in cases:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                self.save(trajectories=values)
                with self.assertRaises(ValueError):
                    TrajectoryDataset(self.path, 4, 2)

    def test_invalid_masks(self):
        for mask in [np.ones(3), np.zeros(2), np.array([1, 0.5]), np.array([1, np.nan]), np.zeros((3, 2))]:
            with self.subTest(mask=mask):
                self.save(trajectories=self.trajectories, mask=mask)
                with self.assertRaises(ValueError):
                    TrajectoryDataset(self.path, 4, 2)

    def test_conditioning_must_match_requested_dimension(self):
        for conditioning in [None, np.zeros((3, 1)), np.full((3, 2), np.inf)]:
            with self.subTest(conditioning=conditioning):
                values = {"trajectories": self.trajectories}
                if conditioning is not None:
                    values["conditioning"] = conditioning
                self.save(**values)
                with self.assertRaises(ValueError):
                    TrajectoryDataset(self.path, 4, 2, conditioning_dim=2)

    def test_missing_key_and_non_archive_are_rejected(self):
        self.save(other=self.trajectories)
        with self.assertRaisesRegex(ValueError, "trajectories"):
            TrajectoryDataset(self.path, 4, 2)
        with self.path.open("wb") as handle:
            np.save(handle, self.trajectories)
        with self.assertRaisesRegex(ValueError, "NPZ"):
            TrajectoryDataset(self.path, 4, 2)

    def test_invalid_constructor_dimensions(self):
        self.save(trajectories=self.trajectories)
        for frames, particles, conditioning in [(0, 2, 0), (4, 0, 0), (4, 2, -1), (4.5, 2, 0), (True, 2, 0)]:
            with self.subTest(values=(frames, particles, conditioning)):
                with self.assertRaises(ValueError):
                    TrajectoryDataset(self.path, frames, particles, conditioning)


if __name__ == "__main__":
    unittest.main()
