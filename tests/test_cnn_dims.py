"""CNN encoder/decoder spatial dimension checks."""

from __future__ import annotations

import torch

from config import Config
from models.decoder import CNNDecoder
from models.encoder import CNNEncoder


def test_encoder_decoder_32x32() -> None:
    config = Config(img_size=32)
    encoder = CNNEncoder(config)
    decoder = CNNDecoder(config)

    assert config.spatial_size == 4
    assert config.flatten_dim == 128 * 4 * 4
    assert config.embed_dim == config.flatten_dim

    x = torch.randn(2, 3, 32, 32)
    e = encoder(x)
    assert e.shape == (2, config.embed_dim), f"embed mismatch: {e.shape}"

    h = torch.randn(2, config.hidden_dim)
    z = torch.randn(2, config.latent_dim)
    o_hat = decoder(h, z)
    assert o_hat.shape == (2, 3, 32, 32), f"decode mismatch: {o_hat.shape}"


def test_encoder_decoder_64x64() -> None:
    """Regression: 64x64 still maps cleanly with the same 3-layer stack."""
    config = Config(img_size=64)
    encoder = CNNEncoder(config)
    decoder = CNNDecoder(config)

    assert config.spatial_size == 8

    x = torch.randn(2, 3, 64, 64)
    o_hat = decoder(torch.randn(2, config.hidden_dim), torch.randn(2, config.latent_dim))
    assert encoder(x).shape == (2, config.embed_dim)
    assert o_hat.shape == (2, 3, 64, 64), f"decode mismatch: {o_hat.shape}"


if __name__ == "__main__":
    test_encoder_decoder_32x32()
    test_encoder_decoder_64x64()
    print("OK: encoder/decoder shapes verified for 32x32 and 64x64")
