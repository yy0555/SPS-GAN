"""Critic-window selection, R1 gradients, and full-rollout cyclic loss checks."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sps_gan.updates import _trajectory_window, update_Dv, update_G


class RecordingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.inputs = []

    def forward(self, trajectory, condition, mask):
        self.inputs.append(trajectory.detach().clone())
        return self.weight * trajectory.sum(dim=(2, 3)).unsqueeze(-1)


class UpdateTests(unittest.TestCase):
    def test_full_window_preserves_input_and_rng(self):
        trajectory = torch.randn(2, 30, 2, 1)
        state = torch.get_rng_state().clone()
        self.assertIs(_trajectory_window(trajectory, 0), trajectory)
        self.assertIs(_trajectory_window(trajectory, 30), trajectory)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        for invalid in (-1, 31, 1.5):
            with self.subTest(frames=invalid), self.assertRaisesRegex(ValueError, "discriminator_frames"):
                _trajectory_window(trajectory, invalid)

    def test_real_and_fake_windows_are_selected_independently(self):
        critic = RecordingCritic()
        frames = torch.arange(30.0).view(1, 30, 1, 1).expand(2, -1, 2, 1)
        real = {"videos": frames, "mask": torch.ones(2, 1), "label_and_props": torch.empty(2, 0)}
        optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
        with patch("sps_gan.updates.torch.randint", side_effect=[torch.tensor(2), torch.tensor(7)]) as draw:
            update_Dv(dis_v=critic, real_data=real, fake_data={"videos": frames},
                      optim_Dv=optimizer, discriminator_frames=16, r1_gamma=0)
        self.assertEqual(draw.call_count, 2)
        torch.testing.assert_close(critic.inputs[0], frames[:, 2:18])
        torch.testing.assert_close(critic.inputs[1], frames[:, 7:23])

    def test_r1_uses_gradients_of_real_window(self):
        critic = RecordingCritic()
        real = {"videos": torch.zeros(2, 30, 2, 1), "mask": torch.ones(2, 1),
                "label_and_props": torch.empty(2, 0)}
        optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
        loss, _ = update_Dv(dis_v=critic, real_data=real,
                            fake_data={"videos": torch.zeros(2, 30, 2, 1)},
                            optim_Dv=optimizer, discriminator_frames=16, r1_gamma=1.0)
        self.assertEqual([x.shape[1] for x in critic.inputs], [16, 16])
        # D(x)=w*sum_coordinates(x) at each frame, hence ||grad D||²=T*2*w².
        expected_r1 = 0.5 * 16 * 2 * 0.2**2
        self.assertAlmostEqual(loss["Dv"], 2 * 0.9 + expected_r1, places=6)
        self.assertAlmostEqual(critic.weight.item(), 0.2 - 0.01 * 16 * 2 * 0.2, places=6)

    def test_generator_cyclic_penalty_uses_all_rollout_steps(self):
        losses = []
        for window in (0, 16):
            with self.subTest(window=window):
                critic = RecordingCritic()
                decoder = nn.Linear(1, 1, bias=False)
                dynamics = nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    decoder.weight.fill_(0.5)
                    dynamics.weight.fill_(0.25)
                video = decoder.weight.reshape(1, 1, 1, 1).expand(2, 30, 2, 1)
                derivatives = dynamics.weight.reshape(1, 1, 1).expand(2, 29, 4)
                loss = update_G(
                    q_size=2, cyclic_coord_loss=0.03, model_dv=critic,
                    model_gi=decoder, model_rnn=dynamics,
                    fake_data={"videos": video, "dlatent": derivatives},
                    optim_Gi=torch.optim.SGD(decoder.parameters(), lr=0.01),
                    optim_RNN=torch.optim.SGD(dynamics.parameters(), lr=0.01),
                    label_props_colors=torch.empty(2, 0), video_mask=torch.ones(2, 1),
                    discriminator_frames=window,
                )
                self.assertEqual(critic.inputs[0].shape[1], window or 30)
                expected_cyclic = 29 * 2 * 0.25 * 0.03
                self.assertAlmostEqual(loss["Gv"], -0.2 + expected_cyclic, places=6)
                # All 29 momentum-derivative steps must contribute to its gradient.
                self.assertAlmostEqual(dynamics.weight.item(), 0.25 - 0.01 * 29 * 2 * 0.03, places=6)
                self.assertTrue(critic.weight.requires_grad)
                losses.append(loss["Gv"])
        self.assertAlmostEqual(losses[0], losses[1], places=6)


if __name__ == "__main__":
    unittest.main()
