"""SPS-GAN frame decoder and image/video discriminators for 96-pixel video."""

import torch
from torch import nn


def _validate_dimensions(channels, base_filters, conditioning_dim=0):
    if channels < 1 or base_filters < 1 or conditioning_dim < 0:
        raise ValueError("Channels and filter counts must be positive; conditioning_dim cannot be negative")


class ImageGenerator(nn.Module):
    """Decode one latent code per frame using sub-pixel convolutions."""

    def __init__(self, nz, channels=3, base_filters=32):
        super().__init__()
        _validate_dimensions(channels, base_filters)
        if nz < 1:
            raise ValueError("nz must be positive")
        self.nz = nz
        self.channels = channels

        def upsample(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1, bias=False),
                nn.PixelShuffle(2), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ]

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, base_filters * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(base_filters * 8), nn.ReLU(inplace=True),
            *upsample(base_filters * 8, base_filters * 4),
            *upsample(base_filters * 4, base_filters * 2),
            *upsample(base_filters * 2, base_filters),
            nn.Conv2d(base_filters, channels * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2), nn.Tanh(),
        )

    def forward(self, latent):
        if latent.ndim != 4 or latent.shape[1:] != (self.nz, 1, 1):
            raise ValueError("Frame latents must have shape (batch, nz, 1, 1)")
        return self.main(latent)


class ImageDiscriminator(nn.Module):
    """Classify frames with an optional conditioning channel."""

    def __init__(self, channels=3, base_filters=32, conditioning_dim=0):
        super().__init__()
        _validate_dimensions(channels, base_filters, conditioning_dim)
        self.channels = channels
        self.conditioning_dim = conditioning_dim
        self.label_handler = nn.Linear(conditioning_dim, 96 * 96) if conditioning_dim else None
        layers = [
            nn.Conv2d(channels + int(conditioning_dim > 0), base_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for multiplier in (2, 4, 8):
            layers.extend([
                nn.Conv2d(base_filters * (multiplier // 2), base_filters * multiplier,
                          4, 2, 1, bias=False),
                nn.BatchNorm2d(base_filters * multiplier), nn.LeakyReLU(0.2, inplace=True),
            ])
        layers.extend([nn.Conv2d(base_filters * 8, 1, 6, 1, 0, bias=False), nn.Sigmoid()])
        self.main = nn.Sequential(*layers)

    def forward(self, images, conditioning):
        if images.ndim != 4 or images.shape[1:] != (self.channels, 96, 96):
            raise ValueError("Images must have shape (batch, channels, 96, 96)")
        if conditioning.shape != (images.shape[0], self.conditioning_dim):
            raise ValueError("Image conditioning shape does not match conditioning_dim")
        if self.label_handler is not None:
            condition_channel = self.label_handler(conditioning).reshape(-1, 1, 96, 96)
            images = torch.cat((condition_channel, images), dim=1)
        return self.main(images).reshape(-1)


class VideoDiscriminator(nn.Module):
    """Classify fixed-length video windows with spatiotemporal convolutions."""

    def __init__(self, channels=3, base_filters=32, conditioning_dim=0, frames=16):
        super().__init__()
        _validate_dimensions(channels, base_filters, conditioning_dim)
        if not isinstance(frames, int) or frames < 16:
            raise ValueError("Video discriminator windows require at least 16 frames")
        self.channels = channels
        self.conditioning_dim = conditioning_dim
        self.frames = frames
        self.label_handler = nn.Linear(conditioning_dim, 96 * 96) if conditioning_dim else None
        layers = [
            nn.Conv3d(channels + int(conditioning_dim > 0), base_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for multiplier in (2, 4, 8):
            layers.extend([
                nn.Conv3d(base_filters * (multiplier // 2), base_filters * multiplier,
                          4, 2, 1, bias=False),
                nn.BatchNorm3d(base_filters * multiplier), nn.LeakyReLU(0.2, inplace=True),
            ])
        layers.extend([
            nn.Flatten(), nn.Linear(base_filters * 8 * (frames // 16) * 6 * 6, 1), nn.Sigmoid(),
        ])
        self.main = nn.Sequential(*layers)

    def forward(self, videos, conditioning):
        if videos.ndim != 5 or videos.shape[1:] != (self.channels, self.frames, 96, 96):
            raise ValueError("Video windows must have shape (batch, channels, frames, 96, 96)")
        if conditioning.shape != (videos.shape[0], self.conditioning_dim):
            raise ValueError("Video conditioning shape does not match conditioning_dim")
        if self.label_handler is not None:
            condition_channel = self.label_handler(conditioning).reshape(-1, 1, 1, 96, 96)
            condition_channel = condition_channel.expand(-1, -1, self.frames, -1, -1)
            videos = torch.cat((condition_channel, videos), dim=1)
        return self.main(videos).reshape(-1)
