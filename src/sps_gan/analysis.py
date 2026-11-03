"""PCA of initial latent coordinates for exploratory dimension analysis."""

import numpy as np
import torch
from sklearn.decomposition import PCA


def initial_coordinate_pca(experiment, condition, *, samples=1024, threshold=0.93):
    """Fit up to 64 PCs, using randomized SVD for high-dimensional coordinates.

    A variance threshold is a diagnostic, not a physical degrees-of-freedom
    guarantee. Report explicitly when the retained components miss it.
    """
    if samples < 2 or not 0 < threshold <= 1:
        raise ValueError("PCA needs at least two samples and threshold in (0, 1]")
    m = experiment.config.model
    with torch.no_grad():
        noise_dim = m.noise_dim or m.latent_dim
        noise = torch.randn(samples, noise_dim, device=experiment.device)
        conditions = condition.expand(samples, -1)
        initial = experiment.rnn.phase_space_map(torch.cat((conditions, noise), dim=1))
        coordinates = initial[:, :m.latent_dim // 2].cpu().numpy()
    components = min(samples, coordinates.shape[1], 64)
    if float(np.var(coordinates, axis=0).sum()) == 0:
        raise ValueError("Initial latent coordinates have zero variance; PCA is undefined")
    pca = PCA(n_components=components, svd_solver=(
        "randomized" if coordinates.shape[1] > components else "full"
    ), random_state=0).fit(coordinates)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    reached = bool(cumulative[-1] >= threshold)
    return {
        "samples": samples, "q_dimension": coordinates.shape[1],
        "components_retained": components, "threshold": threshold,
        "threshold_reached": reached,
        "components_for_threshold": int(np.searchsorted(cumulative, threshold) + 1) if reached else None,
        "explained_variance_ratio": ratios.tolist(),
    }
