"""Prepare pixel datasets and export generated videos.

Prepared archives contain uint8 ``videos[N,T,C,96,96]``. Export accepts one
uint8 clip ``[T,C,96,96]``; normalization belongs to the dataset/model interface.
"""

import json
import math
from pathlib import Path
import tempfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _new_output(path):
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists: {path}")
    return path


def _frame_array(frame, image_size, channels):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
        raise ValueError("Decoded frames must contain uint8 grayscale or RGB pixels")
    if frame.ndim == 3:
        if frame.shape[-1] not in (1, 3, 4):
            raise ValueError("Decoded frames must have one, three, or four channels")
        if frame.shape[-1] == 1:
            frame = frame[..., 0]
    picture = Image.fromarray(frame).convert("RGB" if channels == 3 else "L")
    picture = picture.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(picture, dtype=np.uint8)
    return array.transpose(2, 0, 1) if channels == 3 else array[None]


def prepare_video(source, output, *, num_frames=30, stride=1, image_size=96,
                  channels=3, max_clips=None):
    """Convert one local video into nonoverlapping clips after temporal sampling.

    Every ``stride``-th decoded frame is retained, starting with frame zero.
    Incomplete trailing clips are discarded. ``max_clips`` limits decoding and
    memory use. All retained clips are assembled in memory before NPZ writing.
    The archive's scalar JSON ``metadata`` records the preparation settings;
    scalar ``dt`` and ``source_fps`` arrays are also included when FPS is known.
    """
    num_frames = _positive_integer(num_frames, "num_frames")
    stride = _positive_integer(stride, "stride")
    image_size = _positive_integer(image_size, "image_size")
    if image_size != 96:
        raise ValueError("Video preparation requires image_size = 96")
    channels = _positive_integer(channels, "channels")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if max_clips is not None:
        max_clips = _positive_integer(max_clips, "max_clips")
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    if source.resolve() == output.resolve():
        raise ValueError("Source and output must be different files")
    if not source.is_file():
        raise FileNotFoundError(f"Video source not found: {source}")
    output = _new_output(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Prepared dataset output must have an .npz extension")

    clips, pending = [], []
    frames_read = 0
    with imageio.get_reader(str(source), format="FFMPEG") as reader:
        raw_fps = reader.get_meta_data().get("fps")
        try:
            source_fps = float(raw_fps)
        except (TypeError, ValueError):
            source_fps = None
        if source_fps is not None and (not math.isfinite(source_fps) or source_fps <= 0):
            source_fps = None
        for index, frame in enumerate(reader):
            frames_read += 1
            if index % stride:
                continue
            pending.append(_frame_array(frame, image_size, channels))
            if len(pending) == num_frames:
                clips.append(np.stack(pending))
                pending.clear()
                if max_clips is not None and len(clips) >= max_clips:
                    break
    if not clips:
        raise ValueError(
            f"Source contains no complete {num_frames}-frame clip at stride {stride}"
        )
    metadata = {
        "clips": len(clips), "num_frames": num_frames, "stride": stride,
        "image_size": image_size, "channels": channels,
        "source_fps": source_fps, "dt": stride / source_fps if source_fps else None,
        "source_frames_read": frames_read,
    }
    arrays = {"videos": np.stack(clips), "metadata": np.asarray(json.dumps(metadata))}
    if source_fps is not None:
        arrays["source_fps"] = np.asarray(source_fps)
        arrays["dt"] = np.asarray(metadata["dt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        output.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def write_video(path, frames, *, fps):
    """Write one uint8 ``[T,C,96,96]`` clip as an H.264 MP4 file.

    Grayscale frames are repeated over RGB channels for the video codec. Input
    values are neither rescaled nor normalized. Existing files are preserved.
    """
    path = _new_output(path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("Video output must have an .mp4 extension")
    if isinstance(fps, bool):
        raise ValueError("fps must be finite and positive")
    try:
        fps = float(fps)
    except (TypeError, ValueError) as error:
        raise ValueError("fps must be finite and positive") from error
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    frames = np.asarray(frames)
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] < 1
            or frames.shape[1] not in (1, 3) or frames.shape[2:] != (96, 96)):
        raise ValueError("frames must be uint8 with shape (T, C, 96, 96), T >= 1, C = 1 or 3")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary MP4 so a failed encoding leaves no partial result.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mp4", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with imageio.get_writer(str(temporary), format="FFMPEG", mode="I",
                                fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
            for frame in frames:
                rgb = np.repeat(frame, 3, axis=0) if frame.shape[0] == 1 else frame
                writer.append_data(np.ascontiguousarray(rgb.transpose(1, 2, 0)))
        # A hard link publishes the completed file without replacing an existing one.
        path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)
