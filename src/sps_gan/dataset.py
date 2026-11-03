"""Load external position trajectories without bundling datasets or simulators.

The NPZ contract is deliberately independent of the source of the observations:

* ``trajectories``: real numeric array with shape ``(N, T, 2, n)``. The third
  axis is Cartesian ``(x, y)`` and the last axis indexes particles or landmarks.
* ``mask`` (optional): binary array with shape ``(N, n)`` or ``(n,)``. At least
  one particle must be present in every sample. Defaults to all ones.
* ``conditioning`` (optional): numeric array with shape ``(N, conditioning_dim)``.
  Required when ``conditioning_dim > 0``; labels and physical properties must
  already be encoded as numeric feature vectors.

Export source observations with ``numpy.savez(path, trajectories=positions)``.
Convert physical angles, momenta, images, or flattened states to Cartesian
positions before exporting; the loader does not infer those representations.
Dataset sources and preparation guidance are in ``dataset/README.md``.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_numeric(array, name):
    """Check values in chunks to bound the size of validation temporaries."""
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values, got {array.dtype}")
    limit = np.finfo(np.float32).max
    # nditer also avoids flattening copies for Fortran-order or broadcast arrays.
    chunks = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=1_000_000,
        order="K",
    )
    for chunk in chunks:
        if not np.isfinite(chunk).all():
            raise ValueError(f"{name} must contain only finite values")
        if np.any(chunk > limit) or np.any(chunk < -limit):
            raise ValueError(f"{name} values must be representable as float32")


class TrajectoryDataset(Dataset):
    """In-memory NPZ dataset with random contiguous windows and particle padding.

    Every item is ``(trajectory, mask, conditioning)`` as CPU float32 tensors
    with shapes ``(num_frames, 2, max_n)``, ``(max_n,)``, and
    ``(conditioning_dim,)`` respectively. Masked and padded coordinates are zero.
    Windows are sampled with PyTorch's RNG, so ``torch.manual_seed`` controls
    cropping. The archive is loaded once; pickle/object arrays are never enabled.
    """

    def __init__(self, path, num_frames, max_n, conditioning_dim=0):
        self.path = Path(path).expanduser()
        self.num_frames = _integer(num_frames, "num_frames", 1)
        self.max_n = _integer(max_n, "max_n", 1)
        self.conditioning_dim = _integer(conditioning_dim, "conditioning_dim", 0)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Trajectory dataset not found: {self.path}. "
                "Download and prepare data as described in dataset/README.md, "
                "then set [paths] dataset to the resulting .npz file."
            )

        archive = np.load(self.path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("The dataset must be an NPZ archive with a trajectories array")
        with archive:
            if "trajectories" not in archive:
                raise ValueError("The NPZ archive must contain a trajectories array")
            trajectories = archive["trajectories"]
            if trajectories.ndim != 4 or trajectories.shape[2] != 2:
                raise ValueError("trajectories must have shape (N, T, 2, n)")
            samples, frames, _, particles = trajectories.shape
            if samples == 0 or particles == 0:
                raise ValueError("trajectories must contain samples and particles")
            if frames < self.num_frames:
                raise ValueError(
                    f"trajectories have {frames} frames; at least {self.num_frames} are required"
                )
            if particles > self.max_n:
                raise ValueError(
                    f"trajectories have {particles} particles, exceeding max_n={self.max_n}"
                )
            _validate_numeric(trajectories, "trajectories")

            mask = archive["mask"] if "mask" in archive else np.ones(particles)
            if mask.shape == (particles,):
                mask = np.broadcast_to(mask, (samples, particles))
            elif mask.shape != (samples, particles):
                raise ValueError("mask must have shape (n,) or (N, n)")
            _validate_numeric(mask, "mask")
            for start in range(0, samples, 1024):
                chunk = mask[start : start + 1024]
                if not np.all((chunk == 0) | (chunk == 1)):
                    raise ValueError("mask must contain only zero or one")
                if np.any(chunk.sum(axis=1) == 0):
                    raise ValueError("mask must select at least one particle per sample")

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

            self.trajectories = torch.from_numpy(
                np.array(trajectories, dtype=np.float32, copy=True, order="C")
            )
            self.masks = torch.from_numpy(np.array(mask, dtype=np.float32, copy=True))
            self.conditioning = torch.from_numpy(
                np.array(conditioning, dtype=np.float32, copy=True)
            )
        self.n_particles = particles

    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, index):
        source = self.trajectories[index]
        available = source.shape[0] - self.num_frames
        start = int(torch.randint(available + 1, ()).item()) if available else 0
        trajectory = torch.zeros(self.num_frames, 2, self.max_n, dtype=torch.float32)
        mask = torch.zeros(self.max_n, dtype=torch.float32)
        mask[: self.n_particles] = self.masks[index]
        trajectory[:, :, : self.n_particles] = (
            source[start : start + self.num_frames] * self.masks[index]
        )
        return trajectory, mask, self.conditioning[index].clone()
