"""Prepared-video shape, normalization, and sampling contract checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sps_gan.video_dataset import VideoDataset


class VideoDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "videos.npz"

    def save(self, **arrays):
        np.savez(self.path, **arrays)
        return self.path

    def test_byte_pixels_are_normalized_per_window_without_expanding_storage(self):
        videos = np.zeros((2, 3, 3, 96, 96), dtype=np.uint8)
        videos[1, 0] = 255
        videos[1, 1] = 128
        conditioning = np.array([[1.0, 2.0], [3.0, 4.0]])
        dataset = VideoDataset(self.save(videos=videos, conditioning=conditioning), 3,
                               conditioning_dim=2)
        video, mask, condition = dataset[1]
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.videos.dtype, np.uint8)
        self.assertEqual(dataset.videos.nbytes, videos.nbytes)
        self.assertEqual(video.shape, (3, 3, 96, 96))
        self.assertEqual(video.dtype, torch.float32)
        self.assertEqual(mask.shape, (0,))
        torch.testing.assert_close(video[:, 0], torch.ones(3, 96, 96))
        torch.testing.assert_close(video[:, 1], torch.full((3, 96, 96), 128 * 2 / 255 - 1))
        torch.testing.assert_close(video[:, 2], -torch.ones(3, 96, 96))
        torch.testing.assert_close(condition, torch.tensor([3.0, 4.0]))
        video.zero_()
        condition.zero_()
        torch.testing.assert_close(dataset[1][0][:, 0], torch.ones(3, 96, 96))
        torch.testing.assert_close(dataset[1][2], torch.tensor([3.0, 4.0]))

    def test_float_grayscale_pixels_keep_their_supplied_normalization(self):
        videos = np.zeros((1, 3, 1, 96, 96), dtype=np.float64)
        videos[:, 0] = -0.5
        videos[:, 2] = 0.75
        dataset = VideoDataset(self.save(videos=videos), 3, channels=1)
        state = torch.get_rng_state().clone()
        video, mask, condition = dataset[0]
        torch.testing.assert_close(video, torch.from_numpy(videos[0]).float().permute(1, 0, 2, 3))
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertEqual(video.shape, (1, 3, 96, 96))
        self.assertEqual(mask.numel(), 0)
        self.assertEqual(condition.numel(), 0)
        self.assertEqual(video[:, 1].abs().sum().item(), 0)

    def test_seeded_contiguous_cropping_includes_last_valid_start(self):
        videos = np.broadcast_to(
            np.linspace(-1, 1, 6, dtype=np.float32)[None, :, None, None, None],
            (1, 6, 1, 96, 96),
        )
        dataset = VideoDataset(self.save(videos=videos), 3, channels=1)
        torch.manual_seed(17)
        first = dataset[0][0]
        torch.manual_seed(17)
        torch.testing.assert_close(dataset[0][0], first)
        torch.testing.assert_close(first[:, 1:] - first[:, :-1], torch.full((1, 2, 96, 96), 0.4))
        with patch("sps_gan.video_dataset.torch.randint", return_value=torch.tensor(3)) as draw:
            last = dataset[0][0]
        draw.assert_called_once_with(4, ())
        torch.testing.assert_close(last[0, :, 0, 0], torch.tensor([0.2, 0.6, 1.0]))

    def test_missing_input_or_array_and_non_archive_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "dataset/README.md"):
            VideoDataset(self.path, 3)
        self.save(other=np.zeros(1))
        with self.assertRaisesRegex(ValueError, "videos"):
            VideoDataset(self.path, 3)
        with self.path.open("wb") as stream:
            np.save(stream, np.zeros((1, 3, 3, 96, 96), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "NPZ"):
            VideoDataset(self.path, 3)

    def test_invalid_shapes_and_pixel_values(self):
        good_shape = (1, 3, 3, 96, 96)
        cases = [
            np.zeros((1, 3, 96, 96)),
            np.zeros((0, 3, 3, 96, 96)),
            np.zeros((1, 2, 3, 96, 96)),
            np.zeros((1, 3, 1, 96, 96)),
            np.zeros((1, 3, 3, 95, 96)),
            np.full(good_shape, np.nan),
            np.full(good_shape, np.inf),
            np.full(good_shape, -1.01),
            np.full(good_shape, 255.0),
            np.zeros(good_shape, dtype=np.int16),
            np.zeros(good_shape, dtype=bool),
            np.zeros(good_shape, dtype=complex),
            np.full(good_shape, "x"),
            np.empty(good_shape, dtype=object),
        ]
        for values in cases:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                self.save(videos=values)
                with self.assertRaises(ValueError):
                    VideoDataset(self.path, 3)

    def test_conditioning_requires_matching_dimensions_and_finite_values(self):
        videos = np.zeros((2, 3, 3, 96, 96), dtype=np.uint8)
        for condition in (None, np.zeros((2, 1)), np.zeros((1, 2)),
                          np.full((2, 2), np.nan), np.full((2, 2), 1e100)):
            with self.subTest(condition=condition):
                arrays = {"videos": videos}
                if condition is not None:
                    arrays["conditioning"] = condition
                self.save(**arrays)
                with self.assertRaises(ValueError):
                    VideoDataset(self.path, 3, conditioning_dim=2)
        self.save(videos=videos, conditioning=np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "conditioning"):
            VideoDataset(self.path, 3)

    def test_invalid_dimensions_fail_before_loading_data(self):
        for options in ({"num_frames": 0}, {"num_frames": True}, {"num_frames": 1.5},
                        {"image_size": 32}, {"channels": 2}, {"channels": 0},
                        {"conditioning_dim": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                VideoDataset(self.path, **({"num_frames": 3} | options))


if __name__ == "__main__":
    unittest.main()
