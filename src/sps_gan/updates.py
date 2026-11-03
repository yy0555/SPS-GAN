"""SPS-GAN hinge adversarial losses, R1, and cyclic-coordinate regularization."""

import torch
from torch.nn import functional as F


def _trajectory_window(trajectories, frames):
    """Select one random contiguous window per batch; zero uses all frames."""
    if trajectories.ndim < 2:
        raise ValueError("Trajectories require a batch and time dimension")
    if not isinstance(frames, int) or frames < 0 or frames > trajectories.shape[1]:
        raise ValueError("discriminator_frames must be zero or a positive integer no larger than the sequence")
    if frames == 0 or frames == trajectories.shape[1]:
        return trajectories
    start = int(torch.randint(trajectories.shape[1] - frames + 1, ()))
    return trajectories[:, start:start + frames]


def update_Dv(*, dis_v, real_data, fake_data, optim_Dv, r1_gamma=10.0, gamma=0.9,
              discriminator_frames=0):
    """Take one hinge discriminator step with R1 on the sampled real window."""
    need_r1 = r1_gamma > 0
    real = _trajectory_window(real_data["videos"], discriminator_frames).detach().requires_grad_(need_r1)
    fake = _trajectory_window(fake_data["videos"], discriminator_frames).detach()
    condition = real_data["label_and_props"].detach()
    mask = real_data["mask"]
    optim_Dv.zero_grad(set_to_none=True)
    # Native GRU operations support the second derivatives needed by R1;
    # cuDNN's RNN backward does not.
    with torch.backends.cudnn.flags(enabled=False if need_r1 else torch.backends.cudnn.enabled):
        real_scores = dis_v(real, condition, mask)
        fake_scores = dis_v(fake, condition, mask)
        real_loss = F.relu(gamma - real_scores).mean()
        fake_loss = F.relu(gamma + fake_scores).mean()
        penalty = real_scores.new_zeros(())
        if need_r1:
            gradient = torch.autograd.grad(real_scores.sum(), real, create_graph=True)[0]
            penalty = 0.5 * r1_gamma * gradient.flatten(1).square().sum(1).mean()
        loss = real_loss + fake_loss + penalty
        loss.backward()
    optim_Dv.step()
    return (
        {"Dv_real": real_loss.item(), "Dv_fake": fake_loss.item(), "Dv": loss.item()},
        {"Dv_real": real_scores.detach().mean(), "Dv_fake": fake_scores.detach().mean()},
    )


def update_G(*, q_size, cyclic_coord_loss, model_dv, model_gi, model_rnn,
             fake_data, optim_Gi, optim_RNN, label_props_colors, video_mask,
             discriminator_frames=0):
    """Take one generator step using hinge loss and summed L1 regularization.

    Only the critic input is cropped. The cyclic penalty sums over every rollout
    step and momentum coordinate and averages over the batch.
    """
    optim_Gi.zero_grad(set_to_none=True)
    optim_RNN.zero_grad(set_to_none=True)
    flags = [parameter.requires_grad for parameter in model_dv.parameters()]
    try:
        for parameter in model_dv.parameters():
            parameter.requires_grad_(False)
        critic_input = _trajectory_window(fake_data["videos"], discriminator_frames)
        scores = model_dv(critic_input, label_props_colors, video_mask)
        loss = -scores.mean()
        derivatives = fake_data["dlatent"]
        if cyclic_coord_loss > 0 and derivatives is not None:
            momentum_derivatives = derivatives[:, :, q_size:]
            loss = loss + momentum_derivatives.abs().sum() / scores.shape[0] * cyclic_coord_loss
        loss.backward()
    finally:
        for parameter, flag in zip(model_dv.parameters(), flags):
            parameter.requires_grad_(flag)
    optim_Gi.step()
    optim_RNN.step()
    return {"Gv": loss.item()}
