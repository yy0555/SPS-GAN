"""Numerical regression and differentiability checks for the released model."""

import unittest

import torch

from sps_gan.models import HNNPhaseSpace, TrajectoryDiscriminator, TrajectoryGenerator
from sps_gan.updates import update_Dv, update_G


class ModelTests(unittest.TestCase):
    def test_research_rollout_regression(self):
        model = HNNPhaseSpace(input_size=4, hidden_size=5, output_size=4, dt=0.05)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.arange(parameter.numel()).reshape_as(parameter)
                                / parameter.numel() - 0.2)
        noise = torch.linspace(-0.6, 0.7, 8).reshape(2, 4)
        states, derivatives = model(noise, 4)
        # Reference outputs for the fixed parameters and input noise above.
        expected_state = torch.tensor([
            [-0.0698857009, 0.5026571751, 0.7343999743, 1.2661430836],
            [-0.3758285642, 1.8677141666, 3.7704570293, 5.9731993675],
        ])
        expected_derivative = torch.tensor([
            [1.1360000372, 1.2720000744, -0.8640000820, -1.0000001192],
            [1.1360000372, 1.2720000744, -0.8640000820, -1.0000001192],
        ])
        torch.testing.assert_close(states[-1], expected_state)
        torch.testing.assert_close(derivatives[0], expected_derivative)

    def test_sampling_lengths_and_masks(self):
        dynamics = HNNPhaseSpace(input_size=4, hidden_size=8, output_size=4)
        decoder = TrajectoryGenerator(nz=4, hidden_dim=8, traj_len=30, n_particles=2)
        with torch.no_grad():
            states, _ = dynamics(torch.randn(3, 4), 7)
            trajectories = decoder(states.transpose(0, 1), torch.tensor([[1.0, 0.0]]))
        self.assertEqual(trajectories.shape, (3, 7, 2, 2))
        self.assertTrue(torch.equal(trajectories[..., 1], torch.zeros(3, 7, 2)))

    def test_r1_and_generator_steps(self):
        torch.manual_seed(9)
        dynamics = HNNPhaseSpace(input_size=4, hidden_size=8, output_size=4)
        decoder = TrajectoryGenerator(nz=4, hidden_dim=8, n_particles=1)
        critic = TrajectoryDiscriminator(0, hidden_dim=8, max_n=1)
        optim_d = torch.optim.Adam(critic.parameters(), lr=1e-3)
        optim_g = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        optim_h = torch.optim.Adam(dynamics.parameters(), lr=1e-3)
        condition, mask = torch.empty(3, 0), torch.ones(3, 1)
        states, derivatives = dynamics(torch.randn(3, 4), 4)
        fake = {"videos": decoder(states.transpose(0, 1), mask),
                "dlatent": derivatives.transpose(0, 1)}
        real = {"videos": torch.randn(3, 4, 2, 1),
                "label_and_props": condition, "mask": mask}
        before_d = critic.fc.weight.detach().clone()
        before_g = decoder.main[0].weight.detach().clone()
        before_h = dynamics.hnn.linear1.weight.detach().clone()
        d_loss, _ = update_Dv(dis_v=critic, real_data=real, fake_data=fake,
                              optim_Dv=optim_d, r1_gamma=1.0)
        g_loss = update_G(q_size=2, cyclic_coord_loss=0.01, model_dv=critic,
                          model_gi=decoder, model_rnn=dynamics, fake_data=fake,
                          optim_Gi=optim_g, optim_RNN=optim_h,
                          label_props_colors=condition, video_mask=mask)
        self.assertTrue(torch.isfinite(torch.tensor([d_loss["Dv"], g_loss["Gv"]])).all())
        self.assertFalse(torch.equal(before_d, critic.fc.weight))
        self.assertFalse(torch.equal(before_g, decoder.main[0].weight))
        self.assertFalse(torch.equal(before_h, dynamics.hnn.linear1.weight))
        self.assertTrue(all(p.requires_grad for p in critic.parameters()))


if __name__ == "__main__":
    unittest.main()
