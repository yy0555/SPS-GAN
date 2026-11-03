"""Load prepared video clips from an external NumPy archive.

The ``videos`` array has shape ``(N, T, C, 96, 96)``. Pixels are either uint8
in [0, 255], or floating values already normalized to [-1, 1]. The loader
does not guess floating-point pixel ranges, decode videos, resize images, or
change frame spacing. Optional ``conditioning`` has shape ``(N, D)``.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import _integer, _validate_numeric


class VideoDataset(Dataset):
    """Sample contiguous video windows with PyTorch's seeded RNG.

    Items are ``(video, mask, conditioning)`` CPU float32 tensors: video has
    shape ``(C, num_frames, 96, 96)`` and values in [-1, 1], mask is empty,
    and conditioning has shape ``(conditioning_dim,)``. The empty mask keeps
    the dataset interface uniform; video pixels have no particle masks.

    NPZ arrays are loaded into memory once. Byte pixels remain uint8 in
    storage and are converted to float32 only for the selected window.
    """

    def __init__(self, path, num_frames, image_size=96, channels=3, conditioning_dim=0):
        self.path = Path(path).expanduser()
        self.num_frames = _integer(num_frames, "num_frames", 1)
        self.image_size = _integer(image_size, "image_size", 1)
        self.channels = _integer(channels, "channels", 1)
        self.conditioning_dim = _integer(conditioning_dim, "conditioning_dim", 0)
        if self.image_size != 96:
            raise ValueError("Video models require image_size = 96")
        if self.channels not in (1, 3):
            raise ValueError("Video channels must be 1 (grayscale) or 3 (RGB)")
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Video dataset not found: {self.path}. "
                "Prepare video clips as described in dataset/README.md, "
                "then set [paths] dataset to the resulting .npz file."
            )

        archive = np.load(self.path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("The video dataset must be an NPZ archive with a videos array")
        with archive:
            if "videos" not in archive:
                raise ValueError("The NPZ archive must contain a videos array")
            videos = archive["videos"]
            if videos.ndim != 5 or videos.shape[2:] != (
                self.channels, self.image_size, self.image_size
            ):
                raise ValueError(
                    f"videos must have shape (N, T, {self.channels}, 96, 96)"
                )
            samples, frames = videos.shape[:2]
            if samples == 0:
                raise ValueError("videos must contain at least one sample")
            if frames < self.num_frames:
                raise ValueError(
                    f"videos have {frames} frames; at least {self.num_frames} are required"
                )
            if videos.dtype != np.uint8 and videos.dtype.kind != "f":
                raise ValueError("videos must contain uint8 pixels or floating pixels in [-1, 1]")
            if videos.dtype.kind == "f":
                _validate_numeric(videos, "videos")
                chunks = np.nditer(
                    videos, flags=["external_loop", "buffered"],
                    op_flags=["readonly"], buffersize=1_000_000, order="K",
                )
                for chunk in chunks:
                    if np.any(chunk < -1) or np.any(chunk > 1):
                        raise ValueError("Floating videos must already be normalized to [-1, 1]")

            if "conditioning" in archive:
                conditioning = archive["conditioning"]
            elif self.conditioning_dim:
                raise ValueError(
                    "The NPZ archive must contain conditioning when conditioning_dim > 0"
                )
            else:
                conditioning = np.empty((samples, 0), dtype=np.float32)
            if conditioning.shape != (samples, self.conditioning_dim):
                raise ValueError(
                    f"conditioning must have shape ({samples}, {self.conditioning_dim})"
                )
            _validate_numeric(conditioning, "conditioning")
            self.videos = videos
            self.conditioning = torch.from_numpy(
                np.array(conditioning, dtype=np.float32, copy=True)
            )

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, index):
        source = self.videos[index]
        available = source.shape[0] - self.num_frames
        start = int(torch.randint(available + 1, ()).item()) if available else 0
        clip = np.array(source[start:start + self.num_frames], dtype=np.float32, copy=True)
        if source.dtype == np.uint8:
            clip *= 2.0 / 255.0
            clip -= 1.0
        video = torch.from_numpy(clip).permute(1, 0, 2, 3).contiguous()
        return video, torch.empty(0, dtype=torch.float32), self.conditioning[index].clone()
