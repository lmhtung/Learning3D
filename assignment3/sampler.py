import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        device = ray_bundle.origins.device
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(
            self.min_depth,
            self.max_depth, 
            self.n_pts_per_ray).to(device)
        #Add jitter 
        jitter = (torch.rand_like(z_vals) - 0.5) * (self.max_depth - self.min_depth) / self.n_pts_per_ray
        z_vals += jitter

        # TODO (Q1.4): Sample points from z values (n_rays, n_pts_per_ray, 3)
        # origins (n_rays, 3)
        # directions (n_rays, 3)
        rays_o = ray_bundle.origins.unsqueeze(1).repeat(1, z_vals.shape[0], 1)
        rays_d = ray_bundle.directions.unsqueeze(1).repeat(1, z_vals.shape[0], 1)
        z_vals = z_vals.unsqueeze(-1).unsqueeze(0)
        sample_points = rays_o + z_vals * rays_d

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}