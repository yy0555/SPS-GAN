"""Command-line entry point; heavyweight imports are deferred until needed."""

import argparse
import json
import logging
from pathlib import Path

from . import __version__
from .configuration import load_config


def get_parser():
    parser = argparse.ArgumentParser(description="SPS-GAN trajectory and video training and sampling")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("train", "Train SPS-GAN from a prepared dataset"),
        ("sample", "Generate trajectories or videos from a trained checkpoint"),
        ("pca", "Inspect variance in initial latent coordinates"),
        ("check-config", "Validate and display resolved configuration"),
        ("prepare-video", "Convert a recording into an NPZ video dataset"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", type=Path, required=True)
        if name == "train":
            command.add_argument("--resume", action="store_true", help="Resume latest checkpoint in output")
        if name in ("sample", "pca"):
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--samples", type=int, default=None if name == "sample" else 1024,
                                 help="Sample count (default: 128 trajectories, 1 video, 1024 PCA points)")
            command.add_argument("--conditioning", default="", help="Comma-separated numeric conditioning vector")
        if name == "sample":
            command.add_argument("--frames", type=int, help="Defaults to configured sequence length")
            command.add_argument("--particles", type=int, help="Active particles; defaults to max_particles")
            command.add_argument("--fps", type=float, help="MP4 frame rate (default: 1 / dt)")
        if name == "pca":
            command.add_argument("--threshold", type=float, default=0.93)
        if name == "prepare-video":
            command.add_argument("--input", type=Path, required=True, help="Source recording")
            command.add_argument("--output", type=Path, help="NPZ destination; defaults to [paths] dataset")
            command.add_argument("--stride", type=int, default=1, help="Keep every nth source frame")
            command.add_argument("--max-clips", type=int, help="Limit the number of clips to prepare")
    return parser


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "check-config":
            print(json.dumps(config.as_dict(), indent=2))
            return 0
        if args.command == "prepare-video":
            if config.data.mode != "video":
                raise ValueError("Set [data] mode = video before preparing a recording")
            if config.data.conditioning_dim:
                raise ValueError("prepare-video creates unconditional data; add conditioning arrays externally")
            from .video_io import prepare_video

            metadata = prepare_video(
                args.input, args.output or config.paths.dataset, num_frames=config.data.num_frames,
                stride=args.stride, image_size=config.video.image_size, channels=config.video.channels,
                max_clips=args.max_clips,
            )
            print(json.dumps(metadata, indent=2))
            if metadata["dt"] is None:
                logging.info("Frame rate unavailable; set [data] dt using the recording's sampling interval")
            else:
                logging.info("Set [data] dt = %s before training", metadata["dt"])
            return 0
        from .experiment import Experiment

        experiment = Experiment(config, training=args.command == "train")
        if args.command == "train":
            experiment.train(resume=args.resume)
            return 0
        import numpy as np
        import torch

        if args.samples is None:
            args.samples = 1 if config.data.mode == "video" else 128
        if args.samples < 1:
            raise ValueError("samples must be positive")
        values = [float(value) for value in args.conditioning.split(",")] if args.conditioning else []
        if len(values) != config.data.conditioning_dim or not np.isfinite(values).all():
            raise ValueError("Supply a finite --conditioning vector matching conditioning_dim")
        condition = torch.tensor([values], dtype=torch.float32, device=experiment.device)
        if not torch.isfinite(condition).all():
            raise ValueError("Conditioning values must be representable as finite float32 numbers")
        experiment.load_checkpoint(args.checkpoint)
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        if args.command == "sample":
            if args.frames is not None and args.frames < 1:
                raise ValueError("frames must be positive")
            conditions = condition.expand(args.samples, -1)
            if config.data.mode == "video":
                if args.particles is not None:
                    raise ValueError("--particles is only supported in trajectory mode")
                if args.output.suffix.lower() not in (".npz", ".mp4"):
                    raise ValueError("Video output must end in .npz or .mp4")
                if args.output.suffix.lower() == ".mp4" and args.samples != 1:
                    raise ValueError("MP4 output requires --samples 1; use NPZ for multiple videos")
                fps = (1 / config.data.dt) if args.fps is None else args.fps
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError("fps must be finite and positive")
                with torch.no_grad():
                    data = experiment.generate(conditions, frames=args.frames)
                    pixels = ((data["videos"].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                pixels = pixels.permute(0, 2, 1, 3, 4).cpu().numpy()
                if args.output.suffix.lower() == ".mp4":
                    from .video_io import write_video

                    write_video(args.output, pixels[0], fps=fps)
                else:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("wb") as stream:
                        np.savez_compressed(stream, videos=pixels,
                                            conditioning=conditions.cpu().numpy(), dt=config.data.dt)
            else:
                if args.fps is not None:
                    raise ValueError("--fps is only supported in video mode")
                if args.output.suffix.lower() != ".npz":
                    raise ValueError("Trajectory output must end in .npz")
                particles = config.data.max_particles if args.particles is None else args.particles
                if not 1 <= particles <= config.data.max_particles:
                    raise ValueError("particles must lie between 1 and max_particles")
                mask = torch.zeros(args.samples, config.data.max_particles, device=experiment.device)
                mask[:, :particles] = 1
                with torch.no_grad():
                    data = experiment.generate(conditions, mask, frames=args.frames)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("wb") as stream:
                    np.savez_compressed(stream, trajectories=data["videos"].cpu().numpy(),
                                        mask=mask.cpu().numpy(), conditioning=conditions.cpu().numpy())
        else:
            from .analysis import initial_coordinate_pca

            result = initial_coordinate_pca(experiment, condition, samples=args.samples, threshold=args.threshold)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
