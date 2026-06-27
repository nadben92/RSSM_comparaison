"""RSSM: assembles encoder, decoder, backbone, prior, and posterior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from config import Config
from models.backbone import BackboneState, SequenceBackbone, build_backbone
from models.decoder import CNNDecoder
from models.encoder import CNNEncoder
from utils import reparameterize


@dataclass
class RSSMOutput:
    """Container for one forward pass over a sequence."""

    recon: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    kl_loss_raw: torch.Tensor
    kl_per_dim: torch.Tensor


class GaussianHead(nn.Module):
    """MLP head that outputs diagonal Gaussian parameters (mu, sigma)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.log_sigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu(h)
        # softplus + epsilon guarantees strictly positive std
        sigma = torch.nn.functional.softplus(self.log_sigma(h)) + 1e-4
        return mu, sigma


class RSSM(nn.Module):
    """Recurrent State-Space Model in the DreamerV1/V2 spirit.

    Temporal order at each step t (CRITICAL — do not reorder):
        1. state_t = backbone.step(state_{t-1}, z_{t-1}, a_{t-1}); h_t = hidden(state_t)
        2. prior:  p(z_t | h_t)
        3. e_t = encoder(o_t)
        4. posterior: q(z_t | h_t, e_t)
        5. z_t ~ q  (reparam trick, always from posterior during training)
        6. o_hat_t = decoder(h_t, z_t)
    """

    def __init__(self, config: Config, backbone: SequenceBackbone | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = CNNEncoder(config)
        self.decoder = CNNDecoder(config)
        self.backbone: SequenceBackbone = backbone or build_backbone(config)

        self.prior = GaussianHead(config.hidden_dim, config.latent_dim)
        self.posterior = GaussianHead(
            config.hidden_dim + config.embed_dim,
            config.latent_dim,
        )

    def _init_states(
        self, batch_size: int, device: torch.device
    ) -> tuple[BackboneState, torch.Tensor, torch.Tensor]:
        """Zero-initialize backbone state, z, and the previous action."""
        state = self.backbone.init_state(batch_size, device)
        z = torch.zeros(batch_size, self.config.latent_dim, device=device)
        a_prev = torch.zeros(batch_size, self.config.action_dim, device=device)
        return state, z, a_prev

    def forward_step(
        self,
        state_prev: BackboneState,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
        obs: torch.Tensor,
        sample: bool = True,
    ) -> tuple[BackboneState, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Single RSSM step following the prescribed temporal order.

        Returns:
            backbone state, z_t, recon, and a dict of intermediate tensors.
        """
        # 1. Update deterministic memory BEFORE seeing the observation
        state = self.backbone.step(state_prev, z_prev, a_prev)
        h = self.backbone.hidden(state)

        # 2. Prior predicts z without the image
        mu_p, sigma_p = self.prior(h)

        # 3. Observe and encode
        embed = self.encoder(obs)

        # 4. Posterior refines the belief with the observation
        mu_q, sigma_q = self.posterior(torch.cat([h, embed], dim=-1))

        # 5. Sample ALWAYS from posterior during training (reparam trick)
        if sample:
            z = reparameterize(mu_q, sigma_q)
        else:
            z = mu_q

        # 6. Reconstruct
        recon = self.decoder(h, z)

        extras = {
            "mu_p": mu_p,
            "sigma_p": sigma_p,
            "mu_q": mu_q,
            "sigma_q": sigma_q,
            "h": h,
            "state": state,
        }
        return state, z, recon, extras

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        free_nats: float = 3.0,
    ) -> RSSMOutput:
        """Unroll the RSSM over a batch of sequences.

        Args:
            obs_seq: ``(B, T, C, H, W)`` observations in ``[0, 1]``.
            action_seq: ``(B, T, action_dim)`` continuous actions.
            free_nats: Optional floor on balanced KL scalar (0 = disabled).

        Returns:
            RSSMOutput with reconstructions and scalar losses.
        """
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device

        state, z, a_prev = self._init_states(batch_size, device)

        recons: list[torch.Tensor] = []
        kl_raw_list: list[torch.Tensor] = []
        kl_per_dim_list: list[torch.Tensor] = []
        kl_balanced_list: list[torch.Tensor] = []
        alpha = self.config.kl_balance_scale

        for t in range(seq_len):
            obs_t = obs_seq[:, t]
            a_t = action_seq[:, t]

            state, z, recon, extras = self.forward_step(state, z, a_prev, obs_t, sample=True)
            recons.append(recon)

            mu_q, sigma_q = extras["mu_q"], extras["sigma_q"]
            mu_p, sigma_p = extras["mu_p"], extras["sigma_p"]

            # Diagnostic KL(q || p) — no detach, for logging and bar charts
            q_dist = Normal(mu_q, sigma_q)
            p_dist = Normal(mu_p, sigma_p)
            kl_t = kl_divergence(q_dist, p_dist)  # (B, latent_dim)

            # DreamerV2 KL balancing: train prior faster than posterior
            q_sg = Normal(mu_q.detach(), sigma_q.detach())
            p_sg = Normal(mu_p.detach(), sigma_p.detach())
            kl_lhs = kl_divergence(q_sg, p_dist)   # gradients → prior
            kl_rhs = kl_divergence(q_dist, p_sg)   # gradients → posterior
            kl_balanced = alpha * kl_lhs + (1 - alpha) * kl_rhs

            kl_per_dim_list.append(kl_t)
            kl_raw_list.append(kl_t.sum(dim=-1))
            kl_balanced_list.append(kl_balanced)

            # a_t becomes a_{t-1} for the next step
            a_prev = a_t

        recon_stack = torch.stack(recons, dim=1)  # (B, T, C, H, W)
        kl_per_dim = torch.stack(kl_per_dim_list, dim=1)  # (B, T, latent_dim)

        # Motion-weighted recon: upweight pixels that change between frames
        motion = torch.zeros_like(obs_seq)
        motion[:, 1:] = (obs_seq[:, 1:] - obs_seq[:, :-1]).abs()
        motion = motion.mean(dim=2, keepdim=True).detach()  # (B, T, 1, H, W)
        weight = 1.0 + self.config.lambda_motion * motion
        sq_err = (recon_stack - obs_seq) ** 2
        recon_loss = (weight * sq_err).sum(dim=[2, 3, 4]).mean()
        kl_loss_raw = torch.mean(torch.stack(kl_raw_list, dim=1))
        kl_balanced_total = torch.stack(kl_balanced_list, dim=1).sum(dim=-1).mean()
        kl_loss = kl_balanced_total
        if free_nats > 0:
            kl_loss = torch.clamp(kl_loss, min=free_nats)

        return RSSMOutput(
            recon=recon_stack,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            kl_loss_raw=kl_loss_raw,
            kl_per_dim=kl_per_dim,
        )

    @torch.no_grad()
    def encode_context(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[BackboneState, torch.Tensor]:
        """Bootstrap backbone state and z from real context frames via the posterior.

        Args:
            obs_seq: ``(B, T_ctx, C, H, W)``.
            action_seq: ``(B, T_ctx, action_dim)``.

        Returns:
            Final ``(backbone_state, z)`` after processing all context frames.
        """
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device
        state, z, a_prev = self._init_states(batch_size, device)

        for t in range(seq_len):
            state, z, _, _ = self.forward_step(
                state, z, a_prev, obs_seq[:, t], sample=False
            )
            a_prev = action_seq[:, t]

        return state, z

    @torch.no_grad()
    def imagine_step(
        self,
        state: BackboneState,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[BackboneState, torch.Tensor, torch.Tensor]:
        """One open-loop imagination step using ONLY the prior.

        Follows the same backbone-first ordering as training, but z is sampled
        from p(z|h) instead of q(z|h,e).

        Returns:
            next backbone state, z_sampled, decoded_image
        """
        state_next = self.backbone.step(state, z, action)
        h_next = self.backbone.hidden(state_next)
        mu_p, sigma_p = self.prior(h_next)
        z_next = reparameterize(mu_p, sigma_p)
        recon = self.decoder(h_next, z_next)
        return state_next, z_next, recon
