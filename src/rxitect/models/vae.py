from typing import Tuple

import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d, latent_dimension):
        """
        Fully Connected layers to encode SELFIES repr. molecule to latent space
        Args:
            in_dimension: TODO
            layer_1d: TODO
            layer_2d: TODO
            layer_3d: TODO
            latent_dimension: TODO
        """
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU(),
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    def reparameterize(mu, log_var) -> torch.Tensor:
        """
        Calculates the reparameterized latent space, this allows back-propogation
        despite the stochastic component in the network.
        Args:
            mu: TODO
            log_var: TODO
        Returns:
            The reparameterized latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return z

    def forward(self, x) -> Tuple[torch.Tensor, float, float]:
        """
        Forward pass through the Encoder
        Args:
            x: TODO
        Returns:
            A tuple containing the latent vector (z), the mean (mu) and log variance (log_var).
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
