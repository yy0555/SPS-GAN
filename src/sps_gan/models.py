"""SPS-GAN latent Hamiltonian dynamics and trajectory networks.

Trajectories use ``(batch, frames, 2, particles)``; latent states concatenate q, p.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nonlinearity="relu"):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        activations = {"relu": nn.ReLU, "softplus": nn.Softplus, "tanh": nn.Tanh}
        if nonlinearity not in activations:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        self.nonlinear = activations[nonlinearity]()
        for layer in (self.linear1, self.linear2, self.output):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.output(self.nonlinear(self.linear2(self.nonlinear(self.linear1(x)))))


class HNNPhaseSpace(nn.Module):
    """Map noise to phase space, then apply the Hamiltonian rollout.

    ``input_size`` includes optional conditioning dimensions; ``output_size`` is
    the even dimension of the latent state alone. Conditioning precedes noise
    at the input and precedes q,p at the Hamiltonian input.
    """

    def __init__(self, *, device="cpu", input_size, hidden_size, output_size,
                 dt=0.05, ndim_physics=0, ndim_label=0):
        super().__init__()
        if output_size < 2 or output_size % 2:
            raise ValueError("output_size must be a positive even phase-space dimension")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.ndim_physics = ndim_physics
        self.ndim_label = ndim_label
        self.condition_dim = ndim_physics + ndim_label
        self.input_size = input_size
        self.output_size = output_size
        if self.condition_dim < 0 or input_size <= self.condition_dim:
            raise ValueError("input_size must include noise beyond the conditioning dimensions")
        # The input motion code and output phase state have independent dimensions.
        self.hnn = MLP(output_size + self.condition_dim, hidden_size, 1)
        self.phase_space_map = MLP(input_size, hidden_size, output_size)
        self.to(device)

    def initWeight(self):
        """Initialize network weights with Xavier uniform initialization."""
        for name, parameter in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)

    def initHidden(self, batch_size):
        """Hamiltonian rollouts have no persistent hidden state."""

    def leap_frog_step(self, x, label_and_props=None):
        """Apply an explicit kick/drift/kick update.

        Differentiation passes through the intermediate kick, including its
        derivative with respect to the input momentum p. This is not a general
        implicit symplectic solver for arbitrary
        nonseparable Hamiltonians.
        """
        with torch.enable_grad():
            q, p = x.chunk(2, dim=-1)
            q.requires_grad_(True)
            p.requires_grad_(True)
            if label_and_props is None:
                label_and_props = x.new_empty((x.shape[0], 0))
            if label_and_props.shape != (x.shape[0], self.condition_dim):
                raise ValueError("Conditioning shape does not match the Hamiltonian")
            energy = self.hnn(torch.cat((label_and_props, q, p), dim=-1))
            dpdt = -torch.autograd.grad(energy.sum(), q, create_graph=True)[0]
            p_half = p + dpdt * (self.dt / 2)
            energy = self.hnn(torch.cat((label_and_props, q, p_half), dim=-1))
            dqdt = torch.autograd.grad(energy.sum(), p, create_graph=True)[0]
            q_next = q + dqdt * self.dt
            energy = self.hnn(torch.cat((label_and_props, q_next, p_half), dim=-1))
            dpdt = -torch.autograd.grad(energy.sum(), q_next, create_graph=True)[0]
            p_next = p_half + dpdt * (self.dt / 2)
            return torch.cat((q_next, p_next), dim=-1), torch.cat((dqdt, dpdt), dim=-1)

    def forward(self, noise, n_frames):
        """Return states and derivatives in (frames, batch, latent) order.

        Hamilton's equations require autograd even during sampling. Detach the
        returned tensors when saving samples; do not use inference_mode here.
        """
        if n_frames < 1:
            raise ValueError("n_frames must be positive")
        if noise.ndim != 2 or noise.shape[1] != self.input_size:
            raise ValueError("Noise shape does not match input_size")
        with torch.enable_grad():
            x = self.phase_space_map(noise)
            condition = noise[:, :self.condition_dim]
            outputs, derivatives = [x], []
            for _ in range(n_frames - 1):
                x, dx = self.leap_frog_step(x, condition)
                outputs.append(x)
                derivatives.append(dx)
            return torch.stack(outputs), torch.stack(derivatives) if derivatives else None


class TrajectoryGenerator(nn.Module):
    """Decode each latent state to Cartesian particle positions."""

    def __init__(self, nz=60, hidden_dim=512, traj_len=30, n_particles=10, ngpu=1):
        super().__init__()
        if ngpu != 1:
            raise ValueError("Only single-device training is supported")
        self.traj_len = traj_len
        self.n_particles = n_particles
        self.traj_dim = 2
        self.main = nn.Sequential(
            nn.Linear(nz, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim), nn.Softplus(),
            nn.Linear(hidden_dim, 2 * n_particles),
        )

    def forward(self, z, mask):
        if z.ndim != 3:
            raise ValueError("Latents must have shape (batch, frames, features)")
        batch, frames, _ = z.shape
        output = self.main(z).reshape(batch, frames, 2, self.n_particles)
        return output * mask.reshape(-1, 1, 1, self.n_particles)


class MinibatchDiscrimination(nn.Module):
    """Optional cross-sample features for the trajectory discriminator."""

    def __init__(self, in_features, out_features, kernel_dim=5):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dim))

    def forward(self, x):
        features = (x @ self.T.reshape(x.shape[1], -1)).reshape(
            x.shape[0], -1, self.T.shape[2]
        )
        distances = (features.unsqueeze(0) - features.unsqueeze(1)).abs().sum(-1)
        # Includes the constant self contribution.
        return torch.cat((x, (-distances).exp().sum(1)), dim=1)


class TrajectoryDiscriminator(nn.Module):
    """Bidirectional GRU trajectory critic with raw hinge-loss scores."""

    def __init__(self, cond_dim, hidden_dim=128, num_layers=2, pos_dim=2,
                 bidirectional=True, max_n=3, minibatch_disc=False):
        super().__init__()
        self.cond_dim = cond_dim
        self.minibatch_disc = minibatch_disc
        self.rnn = nn.GRU(pos_dim * max_n + cond_dim, hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional)
        features = hidden_dim * (2 if bidirectional else 1)
        if minibatch_disc:
            self.mb_layer = MinibatchDiscrimination(features, 16)
            features += 16
        self.fc = nn.Linear(features, 1)

    def forward(self, traj, cond, mask):
        batch, frames, _, particles = traj.shape
        traj = traj * mask.reshape(-1, 1, 1, particles)
        flattened = traj.permute(0, 1, 3, 2).reshape(batch, frames, -1)
        condition = cond.unsqueeze(1).expand(batch, frames, -1)
        output, _ = self.rnn(torch.cat((flattened, condition), dim=-1))
        if self.minibatch_disc:
            return self.fc(self.mb_layer(output[:, -1])).unsqueeze(1)
        return self.fc(output)
