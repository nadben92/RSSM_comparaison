# RSSM World Model — Comparative Study of Temporal Backbones

A pixel-based **Recurrent State-Space Model (RSSM)** world model and a controlled study of three temporal backbones (GRU, LSTM, Transformer) on the same architecture, loss, and latent dimensions.

## Motivation

World models learn **dynamics** — how the world evolves — rather than static correlations. Unlike LLMs trained on text redundancy, a visual world model must track a moving object over time and roll out futures in imagination. This repo asks: **on controlled pixel dynamics, does architectural complexity help?**

> **Research question:** Which temporal backbone learns the best world model — and when (if ever) does a Transformer’s long-range memory pay off over a simple GRU?

## Key results

### Part 1 — Simple bouncing ball (Markovian, no occlusion)

<p align="center">
  <img src="assets/imagine_gru.gif" alt="GRU open-loop imagination (no occlusion)" width="900"/>
</p>

Side-by-side real context vs. imagined rollout (GRU, no occlusion).

Ball-position error vs. imagination horizon. Lower is better.

![Position error — GRU vs LSTM vs Transformer (no occlusion)](assets/position_error_all_backbones.png)

**GRU < LSTM < Transformer** — the GRU wins. The task is Markovian (current position and velocity suffice); no long-range memory is needed. The GRU’s simple recurrent inductive bias is an advantage; the Transformer pays for generality without benefit here. All three backbones learn a well-populated latent `z` (no collapse) — the gap is predictive accuracy, not representation failure.

### Part 2 — Occlusion (short-term memory test)

A vertical opaque band hides the ball for 2–4 frames; predicting reappearance requires remembering its trajectory.

![Position error under occlusion](assets/position_error_occlusion.png)

**All three backbones perform similarly during occlusion.** This is an honest result, not a failure: a 2–4 frame gap still fits within the short memory of an RNN. The Transformer’s theoretical advantage would likely appear only at much longer dependencies (tens to hundreds of steps). **On these difficulty scales, architectural complexity does not help.**

<p align="center">
  <img src="assets/imagine_occlusion.gif" alt="Imagination under occlusion" width="900"/>
</p>

### Part 3 — Latent interpretability (without-occlusion models)

Read-only analysis of the stochastic latent `z` on models that reconstruct well. Three axes: latent traversal, causal ablation, latent structure.

![Latent traversal — ball displacement when varying z[i]](assets/latent_traversal.png)

The latent **encodes dynamics**: active dimensions track the ball over time. **Partial, distributed disentanglement** emerges — horizontal-dominant groups (`z[25]`, `z[19]`) and vertical-dominant groups (`z[5]`, `z[30]`), but spread across correlated dimensions, not clean single “x” or “y” axes.

![Causal importance vs KL per dimension](assets/ablation_vs_kl.png)

**KL ≠ causality** — the strongest technical insight. `z[19]` has moderate KL but is the most critical under ablation; `z[26]` has high KL but is largely redundant (ablating it barely hurts). Encoding information and being causally necessary are different.

Detailed analysis and reproduction: **[interpretability/README.md](interpretability/README.md)**

### Overall conclusion

On simple visual dynamics, **architectural complexity (LSTM, Transformer) does not help — and can hurt**; the GRU is well matched to the task. Transformer advantages would likely require **much longer** temporal dependencies than tested here. The learned latent is **partially interpretable** (emergent x/y structure, distributed across dimensions), and **causal importance of a dimension is not reducible to how much information it encodes**.

---

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
- **Temporal Backbone** — `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, swappable via `SequenceBackbone` (GRU / LSTM / Transformer)
- **CNN Decoder** — `concat(h_t, z_t) → (3, 32, 32)`

### Critical temporal ordering (at each step `t`)

```
1. state_t = backbone.step(state_{t-1}, z_{t-1}, a_{t-1})   # memory BEFORE observation
2. h_t = backbone.hidden(state_t)
3. mu_p, sigma_p = prior(h_t)
4. e_t = encoder(o_t)
5. mu_q, sigma_q = posterior(h_t, e_t)
6. z_t = mu_q + sigma_q * eps                              # sampled from posterior (training)
7. o_hat_t = decoder(h_t, z_t)
```

The backbone state is **opaque** (GRU: `h`; LSTM: `(h, c)`; Transformer: token sequence). The RSSM only uses `init_state` / `step` / `hidden`, which keeps backbones interchangeable.

### Loss

```
L = recon_loss + β · kl_loss
```

- **recon_loss** — squared error, **summed over pixels**, motion-weighted (moving pixels up-weighted via `lambda_motion`).
- **kl_loss** — `KL(q || p)` with **KL balancing** (DreamerV2, `α = 0.8`) to prevent posterior collapse.
- **β-annealing** — linear ramp over early training.

### Imagination (open loop)

After a short real **context** (posterior), roll out `H` steps with **prior only**:

```
state_t = backbone.step(state_{t-1}, z_{t-1}, a_{t-1})
z_t ~ p(z_t | h_t)
o_hat_t = decoder(h_t, z_t)
```

## Environment: bouncing ball

Synthetic NumPy environment: a filled disk (`radius ≈ 7`, ~15–20% of a 32×32 frame) bouncing elastically off walls. Large, high-contrast object — reconstructible by the CNN decoder (unlike a thin pendulum arm).

Frames stored as `uint8`, normalized to `[0,1]` on load. Observations: `(N, T, 3, 32, 32)` · actions: `(N, T, action_dim)` · lengths: `(N,)`.

Part 2 adds a vertical occlusion band (width configurable, e.g. w=17) and stores ground-truth `positions` for evaluation.

## Project structure

```
rssm_world_model/
├── README.md
├── assets/                 # figures for this README
├── requirements.txt
├── config.py
├── bouncing_ball.py
├── models/                 # encoder, decoder, backbone, rssm
├── data.py
├── train.py
├── imagine.py
├── evaluation.py           # position-error curves, multi-backbone comparison
├── interpretability/       # Part 3 — read-only latent analysis
└── utils.py
```

## Installation & usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires **Python 3.10+** and **PyTorch 2.x** (CUDA → MPS → CPU auto-detected).

**Generate dataset:**

```bash
python train.py --env-name bouncing_ball --data-path data/bouncing_ball.npz \
  --force-collect --ball-radius 7 --num-episodes 500 --max-steps 200 --epochs 0 --no-wandb
```

**Train one backbone** (only `--backbone` changes between runs):

```bash
python train.py --env-name bouncing_ball --data-path data/bouncing_ball.npz \
  --checkpoint-dir checkpoints/without_occult --backbone gru \
  --epochs 50 --seq-len 25 --chunk-stride 10 --batch-size 128 \
  --lambda-motion 5 --kl-balance 0.8 --no-wandb
```

Repeat with `--backbone lstm` and `--backbone transformer`. For occlusion: `--occlusion-width 17` and matching data path.

**Comparative evaluation:**

```bash
python evaluation.py \
  --checkpoints checkpoints/without_occult/rssm_gru_epoch050.pt \
                checkpoints/without_occult/rssm_lstm_epoch050.pt \
                checkpoints/without_occult/rssm_transformers_epoch050.pt \
  --labels GRU LSTM Transformer \
  --context-len 10 --horizon 50 --n-episodes 30 \
  --output-dir outputs/without_occult
```

**Imagination GIF + KL diagnostic:**

```bash
python imagine.py --checkpoint checkpoints/without_occult/rssm_gru_epoch050.pt \
  --output-dir outputs/without_occult --context-len 10 --horizon 50
```

## Extending the backbone

```python
class SequenceBackbone(ABC, nn.Module):
    def init_state(self, batch_size, device): ...
    def step(self, state, z_prev, a_prev): ...
    def hidden(self, state): ...                     # (B, hidden_dim)
```

GRU, LSTM, and Transformer implement this interface. Adding a new backbone requires no changes to `train.py`, `imagine.py`, or `rssm.py`.

## Debugging notes

Real failure modes this project hit:

- **Posterior collapse** — empty `z`. Diagnose with KL-per-dimension bars; fix with KL balancing (free bits alone was insufficient).
- **Background domination** — plain averaged MSE ignored the moving ball (~1% of pixels). Fixed by summing over pixels + motion weighting.
- **Object too small** — a 1-pixel pendulum arm is unreconstructible; motivated the controllable ball radius on 32×32.
- **Validate visually** — inspect reconstructions and KL bars before long runs; low MSE can hide an empty latent.

## References

- Hafner et al., *Learning Latent Dynamics for Planning from Pixels* (PlaNet, 2019)
- Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination* (DreamerV1, 2020)
- Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2, 2021)
