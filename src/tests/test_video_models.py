"""Video network shapes, conditioning, and differentiable BCE/R1 training."""

import unittest
from unittest.mock import patch

import torch

from sps_gan.models import HNNPhaseSpace
from sps_gan.video_models import ImageDiscriminator, ImageGenerator, VideoDiscriminator
from sps_gan.video_updates import (
    _sample_images, _sample_window, update_video_discriminators, update_video_generator,
)


class VideoModelTests(unittest.TestCase):
    def test_generator_and_conditional_discriminators(self):
        torch.manual_seed(16)
        generator = ImageGenerator(nz=5, channels=3, base_filters=1).eval()
        image_discriminator = ImageDiscriminator(channels=3, base_filters=1, conditioning_dim=2).eval()
        video_discriminator = VideoDiscriminator(channels=3, base_filters=1, conditioning_dim=2).eval()
        conditions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with torch.no_grad():
            frames = generator(torch.randn(2, 5, 1, 1))
            self.assertEqual(frames.shape, (2, 3, 96, 96))
            self.assertGreaterEqual(frames.min().item(), -1)
            self.assertLessEqual(frames.max().item(), 1)
            image_scores = image_discriminator(frames, conditions)
            video_scores = video_discriminator(frames[:, :, None].expand(-1, -1, 16, -1, -1), conditions)
        self.assertEqual(image_scores.shape, (2,))
        self.assertEqual(video_scores.shape, (2,))
        self.assertTrue(torch.all((image_scores > 0) & (image_scores < 1)))
        self.assertTrue(torch.all((video_scores > 0) & (video_scores < 1)))
        with self.assertRaisesRegex(ValueError, "conditioning"):
            image_discriminator(frames, torch.empty(2, 0))
        with self.assertRaisesRegex(ValueError, "at least 16"):
            VideoDiscriminator(frames=15)

    def test_per_sample_images_and_inclusive_clip_boundaries(self):
        videos = torch.arange(2 * 20.0).reshape(2, 1, 20, 1, 1)
        with patch("sps_gan.video_updates.torch.randint", return_value=torch.tensor([1, 19])):
            images = _sample_images(videos)
        self.assertEqual(images.shape, (2, 1, 1, 1))
        torch.testing.assert_close(images.flatten(), torch.tensor([1.0, 39.0]))
        with patch("sps_gan.video_updates.torch.randint", return_value=torch.tensor(4)) as draw:
            window = _sample_window(videos, 16)
        torch.testing.assert_close(window, videos[:, :, 4:20])
        self.assertEqual(draw.call_args.args[0], 5)
        rng = torch.get_rng_state().clone()
        self.assertIs(_sample_window(videos, 20), videos)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "window length"):
            _sample_window(videos, 21)

    def test_bce_r1_and_cyclic_step_update_all_networks(self):
        torch.manual_seed(7)
        dynamics = HNNPhaseSpace(input_size=4, output_size=4, hidden_size=8)
        generator = ImageGenerator(nz=4, channels=1, base_filters=1)
        image_discriminator = ImageDiscriminator(channels=1, base_filters=1)
        video_discriminator = VideoDiscriminator(channels=1, base_filters=1, frames=16)
        networks = (dynamics, generator, image_discriminator, video_discriminator)
        optim_h, optim_g, optim_i, optim_v = [
            torch.optim.Adam(model.parameters(), lr=1e-3) for model in networks
        ]
        before = [[p.detach().clone() for p in model.parameters()] for model in networks]
        states, derivatives = dynamics(torch.randn(2, 4), 16)
        latents = states.transpose(0, 1).reshape(32, 4, 1, 1)
        frames = generator(latents)
        videos = frames.reshape(2, 16, 1, 96, 96).transpose(1, 2)
        fake_data = {"videos": videos, "dlatent": derivatives.transpose(0, 1)}
        conditioning = torch.empty(2, 0)
        d_losses = update_video_discriminators(
            image_discriminator=image_discriminator, video_discriminator=video_discriminator,
            real_videos=torch.rand_like(videos) * 2 - 1, fake_videos=videos,
            conditioning=conditioning, optim_image=optim_i, optim_video=optim_v,
            frames=16, r1_gamma=1.0, real_label=0.9,
        )
        expected_cyclic = derivatives[:, :, 2:].abs().sum().item() / 2 * 0.03
        g_losses = update_video_generator(
            image_discriminator=image_discriminator, video_discriminator=video_discriminator,
            generator=generator, dynamics=dynamics, fake_data=fake_data,
            conditioning=conditioning, optim_generator=optim_g, optim_dynamics=optim_h,
            frames=16, q_size=2, cyclic_weight=0.03, target_label=1.0,
        )
        self.assertTrue(torch.isfinite(torch.tensor(list(d_losses.values()) + list(g_losses.values()))).all())
        self.assertGreater(d_losses["Di_r1"], 0)
        self.assertGreater(d_losses["Dv_r1"], 0)
        self.assertAlmostEqual(g_losses["cyclic"], expected_cyclic, places=6)
        self.assertAlmostEqual(g_losses["G_total"],
                               g_losses["Gi"] + g_losses["Gv"] + g_losses["cyclic"], places=6)
        for previous, model in zip(before, networks):
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(previous, model.parameters())))
        self.assertTrue(all(p.requires_grad for model in networks for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
