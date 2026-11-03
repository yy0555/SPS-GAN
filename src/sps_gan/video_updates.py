"""Image/video BCE objectives with probability-space R1 and cyclic regularization."""

import torch
from torch.nn import functional as F


def _sample_images(videos):
    """Select an independently sampled frame from each video."""
    if videos.ndim != 5 or videos.shape[2] < 1:
        raise ValueError("Videos must have shape (batch, channels, time, height, width)")
    frames = torch.randint(videos.shape[2], (videos.shape[0],), device=videos.device)
    return videos[torch.arange(videos.shape[0], device=videos.device), :, frames]


def _sample_window(videos, frames):
    """Select a contiguous time window, including the final legal start index."""
    if videos.ndim != 5 or not isinstance(frames, int) or not 1 <= frames <= videos.shape[2]:
        raise ValueError("Video window length must lie between 1 and the sequence length")
    if frames == videos.shape[2]:
        return videos
    start = int(torch.randint(videos.shape[2] - frames + 1, ()))
    return videos[:, :, start:start + frames]


def _discriminator_step(discriminator, real, fake, conditioning, optimizer, r1_gamma, real_label):
    optimizer.zero_grad(set_to_none=True)
    real = real.detach().requires_grad_(r1_gamma > 0)
    real_scores = discriminator(real, conditioning)
    real_loss = F.binary_cross_entropy(real_scores, torch.full_like(real_scores, real_label))
    penalty = real_scores.new_zeros(())
    if r1_gamma > 0:
        gradient = torch.autograd.grad(real_scores.sum(), real, create_graph=True)[0]
        penalty = 0.5 * r1_gamma * gradient.flatten(1).square().sum(1).mean()
    fake_scores = discriminator(fake.detach(), conditioning)
    fake_loss = F.binary_cross_entropy(fake_scores, torch.zeros_like(fake_scores))
    loss = real_loss + fake_loss + penalty
    loss.backward()
    optimizer.step()
    return {"real": real_loss.item(), "fake": fake_loss.item(),
            "r1": penalty.item(), "total": loss.item()}


def update_video_discriminators(*, image_discriminator, video_discriminator,
                                real_videos, fake_videos, conditioning,
                                optim_image, optim_video, frames, r1_gamma=10.0, real_label=0.9):
    """Update the image and video discriminators once each using BCE and R1."""
    if not 0 <= real_label <= 1 or r1_gamma < 0:
        raise ValueError("BCE targets must lie in [0, 1] and r1_gamma must be nonnegative")
    conditioning = conditioning.detach()
    video_losses = _discriminator_step(
        video_discriminator, _sample_window(real_videos, frames),
        _sample_window(fake_videos, frames), conditioning, optim_video, r1_gamma, real_label,
    )
    image_losses = _discriminator_step(
        image_discriminator, _sample_images(real_videos), _sample_images(fake_videos),
        conditioning, optim_image, r1_gamma, real_label,
    )
    return {
        "Di": image_losses["total"], "Dv": video_losses["total"],
        "Di_real": image_losses["real"], "Di_fake": image_losses["fake"],
        "Dv_real": video_losses["real"], "Dv_fake": video_losses["fake"],
        "Di_r1": image_losses["r1"], "Dv_r1": video_losses["r1"],
    }


def update_video_generator(*, image_discriminator, video_discriminator, generator, dynamics,
                           fake_data, conditioning, optim_generator, optim_dynamics,
                           frames, q_size, cyclic_weight, target_label=1.0):
    """Optimize both BCE objectives and the full-rollout cyclic-coordinate loss."""
    if not 0 <= target_label <= 1 or cyclic_weight < 0:
        raise ValueError("BCE targets must lie in [0, 1] and cyclic_weight must be nonnegative")
    optim_generator.zero_grad(set_to_none=True)
    optim_dynamics.zero_grad(set_to_none=True)
    parameters = list(image_discriminator.parameters()) + list(video_discriminator.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        videos = fake_data["videos"]
        video_scores = video_discriminator(_sample_window(videos, frames), conditioning)
        image_scores = image_discriminator(_sample_images(videos), conditioning)
        video_loss = F.binary_cross_entropy(video_scores, torch.full_like(video_scores, target_label))
        image_loss = F.binary_cross_entropy(image_scores, torch.full_like(image_scores, target_label))
        cyclic_loss = video_loss.new_zeros(())
        derivatives = fake_data["dlatent"]
        if cyclic_weight > 0 and derivatives is not None:
            cyclic_loss = derivatives[:, :, q_size:].abs().sum() / videos.shape[0] * cyclic_weight
        loss = video_loss + image_loss + cyclic_loss
        loss.backward()
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
    optim_generator.step()
    optim_dynamics.step()
    return {"Gi": image_loss.item(), "Gv": video_loss.item(),
            "cyclic": cyclic_loss.item(), "G_total": loss.item()}
