# RSSM World Model

A **Recurrent State-Space Model (RSSM)** world model in the spirit of [DreamerV1/V2](https://arxiv.org/abs/1912.01603). The model learns a latent dynamics from pixel observations of a Gymnasium environment and can **reconstruct** real frames and **imagine** future trajectories in open loop — without seeing the real world.

## Architecture

The RSSM maintains two latent states at each time step:

| Symbol | Role | Dim (default) |
|--------|------|---------------|
| `h_t` | Deterministic memory (GRU hidden state) | 200 |
| `z_t` | Stochastic latent (diagonal Gaussian) | 32 |

Components:

- **CNN Encoder** — `(3, 64, 64) → embed_dim`
- **Posterior** `q(z_t | h_t, e_t)` — refines belief with the observation
- **Prior** `p(z_t | h_t)` — predicts next latent without seeing the image
- **GRU Backbone** — `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` (swappable via `SequenceBackbone`)
- **CNN Decoder** — `concat(h_t, z_t) → (3, 64, 64)`

### Temporal flow (training)

```
t=0          t=1          t=2
 │            │            │
 ▼            ▼            ▼
h ← GRU(h,z,a) ← GRU(h,z,a) ← ...
 │            │
 ├─ prior p(z|h)
 ├─ encode o_t → e_t
 ├─ posterior q(z|h,e) → z_t  (sample via reparam trick)
 └─ decode(h,z) → ô_t
```

**Critical ordering** at each step `t`:

```
1. h_t = backbone.step(h_{t-1}, z_{t-1}, a_{t-1})   # memory BEFORE observation
2. mu_p, sigma_p = prior(h_t)
3. e_t = encoder(o_t)
4. mu_q, sigma_q = posterior(h_t, e_t)
5. z_t = mu_q + sigma_q * eps                     # always from posterior
6. o_hat_t = decoder(h_t, z_t)
```

### Loss

```
L = recon_loss + β · kl_loss
```

- **recon_loss** — MSE between `o_hat_t` and `o_t`, averaged over batch and time
- **kl_loss** — `KL(q(z|h,e) || p(z|h))` with **free bits** (`clamp` per dim, min 3.0 nats)
- **β-annealing** — linear ramp from 0 → 1 over the first 5000 steps

### Imagination (open loop)

After encoding a short **context** of real frames via the posterior, the model rolls out `H` steps using **only the prior**:

```
h_t = backbone(h_{t-1}, z_{t-1}, a_{t-1})
z_t ~ p(z_t | h_t)
ô_t = decoder(h_t, z_t)
```

## Project structure

```
rssm-worldmodel/
├── README.md
├── requirements.txt
├── config.py              # Config dataclass
├── models/
│   ├── encoder.py         # CNN encoder
│   ├── decoder.py         # CNN decoder
│   ├── backbone.py        # SequenceBackbone + GRUBackbone
│   └── rssm.py            # RSSM assembly + loss
├── data.py                # Collection + Dataset
├── train.py                 # Training loop + CLI
├── imagine.py             # Imagination rollouts + GIF/figures
└── utils.py               # Seeds, device, helpers
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires **Python 3.10+** and **PyTorch 2.x**. Device is auto-detected: NVIDIA CUDA → Apple Silicon **MPS** (M1/M2/M3) → CPU.

**Mac M2 :** au lancement de `train.py`, tu devrais voir `Using device: mps`. Si tu vois `cpu`, installe une version récente de PyTorch (≥ 2.0) — MPS n'est pas disponible sur les builds trop anciennes.

## Usage

### 1. Collect data (automatic on first train run)

Data is collected by a random agent and cached to `data/cartpole_episodes.npz`:

```bash
python -c "from config import Config; from data import collect_and_save; collect_and_save(Config())"
```

Or force re-collection:

```bash
python train.py --force-collect --epochs 1 --no-wandb
```

### 2. Train

```bash
python train.py \
  --num-episodes 200 \
  --epochs 50 \
  --seq-len 50 \
  --batch-size 16 \
  --beta-max 1.0 \
  --free-nats 3.0 \
  --no-wandb
```

Checkpoints are saved to `checkpoints/rssm_epoch*.pt`.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num-episodes` | 200 | Episodes to collect |
| `--epochs` | 50 | Training epochs |
| `--seq-len` | 50 | Sequence chunk length |
| `--beta-max` | 1.0 | Final KL weight |
| `--free-nats` | 3.0 | Free bits per latent dim |
| `--no-wandb` | off | Disable Weights & Biases |

### 3. Generate dreams (imagination GIF)

```bash
python imagine.py \
  --checkpoint checkpoints/rssm_epoch050.pt \
  --context-len 5 \
  --horizon 15 \
  --episode-idx 0
```

Outputs in `outputs/`:

- `imagine_ep0.gif` — side-by-side: real context (left) vs imagined rollout (right)
- `kl_per_dim_ep0.png` — bar chart of KL per latent dimension

## Extending the backbone

The temporal module implements `SequenceBackbone.step(h_prev, z_prev, a_prev) -> h_next`. To add a Transformer:

```python
class TransformerBackbone(SequenceBackbone):
    def step(self, h_prev, z_prev, a_prev):
        ...
```

Then pass it to `RSSM(config, backbone=TransformerBackbone(config))`. No changes needed in `train.py` or `imagine.py`.

## References

- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (2019)
- Hafner et al., *Mastering Atari with Discrete World Models* (2020)
