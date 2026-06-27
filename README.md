# RSSM World Model — Comparative Study of Temporal Backbones

A **Recurrent State-Space Model (RSSM)** world model in the spirit of [PlaNet](https://arxiv.org/abs/1811.04551) and [DreamerV1/V2](https://arxiv.org/abs/1912.01603). The model learns latent dynamics from pixel observations and can **reconstruct** real frames and **imagine** future trajectories in open loop — without seeing the real world.

The core of this repo is a **controlled comparison of three temporal backbones** — GRU, LSTM, and Transformer — plugged into the *same* RSSM (same encoder, decoder, prior, posterior, loss, and dimensions). Only the temporal module changes, so any difference in predictive quality is attributable to the backbone itself.

## Research question

> On a controlled pixel-based dynamics task, which temporal backbone learns the best world model — and does architectural complexity (LSTM, Transformer) help over a simple GRU?

## Key result (Part 1)

On a **bouncing-ball** environment (a Markovian dynamics: the next state depends only on the current position and velocity), measured by ball-position error as a function of imagination horizon:

```
GRU  <  LSTM  <  Transformer       (lower position error = better)
```

The **GRU wins**. This is consistent with theory rather than surprising: the task has no long-range temporal dependency, so the Transformer's attention over the past brings no benefit while paying the cost of a weaker inductive bias and harder optimization. The simpler the backbone, the better it fits a Markovian task. All three backbones learn a well-populated latent `z` (no collapse), confirming the difference is in predictive accuracy, not representation failure.

> Part 2 (in progress) introduces a harder environment with **long-range dependencies** (e.g. occlusion / inertia), where the Transformer's memory is expected to matter. The interesting comparison is precisely *when* complexity starts to pay off.

## Architecture

The RSSM maintains two latent states at each time step:

| Symbol | Role | Dim (default) |
|--------|------|---------------|
| `h_t` | Deterministic memory (backbone hidden state) | 200 |
| `z_t` | Stochastic latent (diagonal Gaussian) | 32 |

Components:

- **CNN Encoder** — `(3, 32, 32) → embed_dim`
- **Posterior** `q(z_t | h_t, e_t)` — refines belief with the observation
- **Prior** `p(z_t | h_t)` — predicts the next latent without seeing the image
- **Temporal Backbone** — `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, swappable via the `SequenceBackbone` interface (GRU / LSTM / Transformer)
- **CNN Decoder** — `concat(h_t, z_t) → (3, 32, 32)`

### Critical temporal ordering (at each step `t`)

```
1. state_t = backbone.step(state_{t-1}, z_{t-1}, a_{t-1})   # memory BEFORE observation
2. h_t = backbone.hidden(state_t)
3. mu_p, sigma_p = prior(h_t)
4. e_t = encoder(o_t)
5. mu_q, sigma_q = posterior(h_t, e_t)
6. z_t = mu_q + sigma_q * eps                              # always sampled from posterior
7. o_hat_t = decoder(h_t, z_t)
```

The backbone state is **opaque**: GRU uses a single tensor `h`, LSTM uses `(h, c)`, the Transformer uses the accumulated token sequence. The RSSM treats it abstractly via `init_state` / `step` / `hidden`, which is what makes the three backbones interchangeable.

### Loss

```
L = recon_loss + β · kl_loss
```

- **recon_loss** — squared error between `o_hat_t` and `o_t`, **summed over pixels** and averaged over batch/time, and **weighted by motion**: pixels that move between consecutive frames are up-weighted so the model cannot ignore the small moving object in favor of the static background.
- **kl_loss** — `KL(q(z|h,e) || p(z|h))` with **KL balancing** (DreamerV2): the prior is trained to match the posterior faster than the posterior drifts toward the prior (`α = 0.8`), which prevents posterior collapse.
- **β-annealing** — linear ramp `0 → β_max` over a configurable fraction of total training steps.

> **Why these choices?** Earlier versions collapsed: with a plain averaged MSE, the moving object (≈1% of pixels) was ignored in favor of the background, and the latent `z` went empty. Summing over pixels + motion-weighting + KL balancing fixed it. See the debugging notes below.

### Imagination (open loop)

After encoding a short **context** of real frames via the posterior, the model rolls out `H` steps using **only the prior**:

```
state_t = backbone.step(state_{t-1}, z_{t-1}, a_{t-1})
z_t ~ p(z_t | h_t)
o_hat_t = decoder(h_t, z_t)
```

## Environment: bouncing ball

A synthetic environment generated in pure NumPy (no Gym rendering, no headless display issues): a filled disk (`radius ≈ 7`, ~15–20% of a 32×32 frame) bounces elastically off the walls. The large, well-contrasted object is reconstructible by the convolutional decoder — unlike a thin pendulum arm, which is why this environment was chosen.

The frames are stored as `uint8` and normalized to `[0,1]` on the fly in the dataset, to keep memory low (important on free Colab GPUs).

Observations: `(N, T, 3, 32, 32)` · actions: `(N, T, action_dim)` · lengths: `(N,)`.

## Evaluation metrics

- **Position error vs. imagination horizon** — the central figure. After a real context, the model imagines `H` steps; at each step the ball position (center of mass of the colored pixels) is extracted from the imagined frame and compared to the true position. Averaged over many episodes with a std band. This ignores the background and measures what matters: the dynamics.
- **KL per latent dimension** — bar chart showing which `z` dimensions are active (carry information). Used to confirm no collapse and to compare how each backbone allocates its latent.
- **Imagination GIF** — side-by-side real vs. imagined rollout, to qualitatively *see* where each backbone starts to diverge.

## Project structure

```
rssm-worldmodel/
├── README.md
├── requirements.txt
├── config.py              # Config dataclass
├── bouncing_ball.py       # synthetic environment generator
├── models/
│   ├── encoder.py         # CNN encoder (32x32)
│   ├── decoder.py         # CNN decoder (32x32, mirror of encoder)
│   ├── backbone.py        # SequenceBackbone + GRU / LSTM / Transformer
│   └── rssm.py            # RSSM assembly + loss (motion-weighted recon, KL balancing)
├── data.py                # dataset + chunking
├── train.py               # training loop + CLI (--backbone gru|lstm|transformer)
├── imagine.py             # imagination rollouts + GIF/figures
├── evaluation.py          # position-error-vs-horizon curves, multi-backbone comparison
└── utils.py               # seeds, device, helpers
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires **Python 3.10+** and **PyTorch 2.x**. Device is auto-detected: NVIDIA CUDA → Apple Silicon MPS → CPU.

## Usage

### 1. Generate the dataset

```bash
python train.py \
  --env-name bouncing_ball \
  --data-path data/bouncing_ball.npz \
  --force-collect \
  --ball-radius 7 \
  --num-episodes 500 \
  --max-steps 200 \
  --epochs 0 \
  --no-wandb
```

### 2. Train one backbone

The **only** flag that changes between the three runs is `--backbone`. Every other hyperparameter is held identical, which is the condition for a fair comparison.

```bash
python train.py \
  --env-name bouncing_ball \
  --data-path data/bouncing_ball.npz \
  --checkpoint-dir checkpoints/checkpoints_gru \
  --backbone gru \
  --epochs 50 \
  --seq-len 25 \
  --chunk-stride 10 \
  --batch-size 128 \
  --lambda-motion 5 \
  --kl-balance 0.8 \
  --num-workers 0 \
  --no-wandb
```

Repeat with `--backbone lstm` and `--backbone transformer` (and matching `--checkpoint-dir`). The parameter count of each backbone is logged at startup so the comparison can account for capacity, not just architecture.

### 3. Imagination GIF + KL diagnostic

```bash
python imagine.py \
  --checkpoint checkpoints/checkpoints_gru/rssm_epoch050.pt \
  --output-dir outputs/outputs_gru \
  --context-len 10 \
  --horizon 50
```

### 4. Comparative evaluation (the main figure)

```bash
python evaluation.py \
  --checkpoints checkpoints/checkpoints_gru/rssm_epoch050.pt \
                checkpoints/checkpoints_lstm/rssm_epoch050.pt \
                checkpoints/checkpoints_transformer/rssm_epoch050.pt \
  --labels GRU LSTM Transformer \
  --context-len 10 --horizon 50 --n-episodes 30 \
  --output-dir outputs/comparison
```

All checkpoints are evaluated on the **same episodes** for a valid comparison.

## Extending the backbone

The temporal module implements the `SequenceBackbone` interface:

```python
class SequenceBackbone(ABC, nn.Module):
    def init_state(self, batch_size, device): ...   # opaque state
    def step(self, state, z_prev, a_prev): ...       # advance one step
    def hidden(self, state): ...                     # extract h (B, hidden_dim)
```

GRU, LSTM, and Transformer all implement it. The RSSM never inspects the state structure, so a new backbone needs no changes to `train.py`, `imagine.py`, or `rssm.py`.

## Debugging notes (the hard-won lessons)

This project went through several real failure modes worth documenting:

- **Posterior collapse** — the latent `z` going empty. Diagnosed via the KL-per-dimension bar chart. Fixed with KL balancing (free bits alone was insufficient on predictable dynamics).
- **Background domination** — a plain averaged MSE let the model ignore the small moving object (the pendulum arm was ~1% of pixels) and paint only the static background. Fixed by summing the reconstruction over pixels and weighting by motion.
- **Object too small to reconstruct** — a 1-pixel-wide arm is unreconstructible by a convolutional decoder regardless of the loss. This motivated moving from Pendulum to a controllable bouncing-ball environment where the object size is a parameter.
- **Always validate visually before a long run** — inspect reconstructions and the KL bar chart, not just the loss value. A low MSE can hide an empty `z`.

## References

- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet, 2019)
- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (DreamerV1, 2020)
- Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2, 2021)
