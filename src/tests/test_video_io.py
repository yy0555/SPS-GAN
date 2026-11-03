"""Video preparation and MP4 export contracts without a codec dependency."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from sps_gan.video_io import prepare_video, write_video


class VideoIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "input.mp4"
        self.source.touch()
        self.output = self.root / "dataset.npz"

    def reader(self, frames, fps=50.0):
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.__iter__.return_value = iter(frames)
        reader.get_meta_data.return_value = {"fps": fps}
        return reader

    def test_stride_clips_metadata_and_no_pixel_normalization(self):
        frames = [np.full((8, 12, 3), value, dtype=np.uint8) for value in range(13)]
        with patch("sps_gan.video_io.imageio.get_reader", return_value=self.reader(frames)):
            metadata = prepare_video(self.source, self.output, num_frames=3, stride=2)
        self.assertEqual(metadata["clips"], 2)
        self.assertEqual(metadata["dt"], 0.04)
        with np.load(self.output, allow_pickle=False) as archive:
            videos = archive["videos"]
            self.assertEqual(videos.shape, (2, 3, 3, 96, 96))
            self.assertEqual(videos.dtype, np.uint8)
            np.testing.assert_array_equal(videos[:, :, 0, 0, 0], [[0, 2, 4], [6, 8, 10]])
            self.assertEqual(json.loads(str(archive["metadata"])), metadata)
            self.assertEqual(float(archive["source_fps"]), 50.0)

    def test_grayscale_max_clips_and_unknown_fps(self):
        frames = [np.full((96, 96, 3), 127, dtype=np.uint8) for _ in range(7)]
        reader = self.reader(frames, fps=None)
        with patch("sps_gan.video_io.imageio.get_reader", return_value=reader):
            metadata = prepare_video(self.source, self.output, num_frames=2,
                                     channels=1, max_clips=1)
        self.assertEqual(metadata["source_frames_read"], 2)
        self.assertIsNone(metadata["dt"])
        with np.load(self.output, allow_pickle=False) as archive:
            self.assertEqual(archive["videos"].shape, (1, 2, 1, 96, 96))
            self.assertTrue((archive["videos"] == 127).all())
            self.assertNotIn("dt", archive)

    def test_incomplete_source_does_not_create_output(self):
        frames = [np.zeros((96, 96, 3), dtype=np.uint8)]
        with patch("sps_gan.video_io.imageio.get_reader", return_value=self.reader(frames)):
            with self.assertRaisesRegex(ValueError, "no complete"):
                prepare_video(self.source, self.output, num_frames=2)
        self.assertFalse(self.output.exists())

    def test_invalid_preparation_arguments_and_overwrites(self):
        for kwargs in ({"num_frames": 0}, {"stride": -1}, {"channels": 2},
                       {"max_clips": 0}, {"image_size": 32}, {"num_frames": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_video(self.source, self.output, **kwargs)
        with self.assertRaisesRegex(ValueError, "different files"):
            prepare_video(self.source, self.source)
        self.output.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_video(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"preserve")

    def test_export_transposes_and_repeats_grayscale_without_rescaling(self):
        output = self.root / "generated.mp4"
        frames = np.full((2, 1, 96, 96), 64, dtype=np.uint8)
        writer = MagicMock()
        writer.__enter__.return_value = writer
        with patch("sps_gan.video_io.imageio.get_writer", return_value=writer) as factory:
            write_video(output, frames, fps=20)
        self.assertTrue(output.exists())
        self.assertEqual(factory.call_args.kwargs["fps"], 20)
        self.assertEqual(factory.call_args.kwargs["codec"], "libx264")
        self.assertEqual(writer.append_data.call_count, 2)
        image = writer.append_data.call_args.args[0]
        self.assertEqual(image.shape, (96, 96, 3))
        self.assertTrue((image == 64).all())
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_video(output, frames, fps=20)

    def test_export_rejects_invalid_frames_and_cleans_failed_encoding(self):
        output = self.root / "generated.mp4"
        valid = np.zeros((2, 3, 96, 96), dtype=np.uint8)
        for frames in (valid.astype(np.float32), valid[:, :, :32], valid[:0]):
            with self.subTest(shape=frames.shape), self.assertRaisesRegex(ValueError, "frames must"):
                write_video(output, frames, fps=20)
        for fps in (0, float("nan"), True):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "fps"):
                write_video(output, valid, fps=fps)
        with patch("sps_gan.video_io.imageio.get_writer", side_effect=RuntimeError("codec failure")):
            with self.assertRaisesRegex(RuntimeError, "codec failure"):
                write_video(output, valid, fps=20)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("*.mp4")), [self.source])


if __name__ == "__main__":
    unittest.main()
