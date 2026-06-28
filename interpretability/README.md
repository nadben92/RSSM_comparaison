# Latent interpretability (Part 3)

Read-only analysis of trained RSSM checkpoints on **without-occlusion** models that reconstruct well (populated latent `z`). This folder does **not** modify `models/`, `train.py`, or the loss.

> **Prerequisite:** models with occlusion often collapse the latent; these analyses are not meaningful on them. Use checkpoints under `checkpoints/without_occult/`.

## Layout

| File | Axis | Question |
|------|------|----------|
| `latent_traversal.py` | 1 | What does each dimension of `z` encode? |
| `ablation.py` | 2 | Which dimensions are causally important? |
| `latent_structure.py` | 3 | How is the latent organized? |
| `utils_interp.py` | — | Loading, KL, decoding, ball position |

Default checkpoint (GRU, best Part 1 model): `checkpoints/without_occult/rssm_gru_epoch050.pt`  
Dataset: `data/bouncing_ball.npz`

Run all commands from the **repo root**.

## Commands

```bash
# Axis 1 — Latent traversal
python interpretability/latent_traversal.py \
  --checkpoint checkpoints/without_occult/rssm_gru_epoch050.pt \
  --range-mode temporal \
  --top-k 4 \
  --include-dims 19 \
  --output-dir interpretability/outputs/gru

# Axis 2 — Causal ablation
python interpretability/ablation.py \
  --checkpoint checkpoints/without_occult/rssm_gru_epoch050.pt \
  --output-dir interpretability/outputs/gru \
  --n-episodes 20

# Axis 3 — Latent structure
python interpretability/latent_structure.py \
  --checkpoint checkpoints/without_occult/rssm_gru_epoch050.pt \
  --output-dir interpretability/outputs/gru

# Compare GRU / LSTM / Transformer (KL allocation)
python interpretability/latent_structure.py \
  --compare-backbones \
  --output-dir interpretability/outputs/all_backbones
```

### Useful flags

| Flag | Script | Default | Description |
|------|--------|---------|-------------|
| `--range-mode temporal` | traversal | `temporal` | Sweep min/max of `z[i]` over episode (+ margin) |
| `--range-mode sigma` | traversal | — | Legacy: μ_q ± σ_q at reference frame |
| `--temporal-margin 0.2` | traversal | `0.2` | Extra margin beyond observed min/max |
| `--displacement-refs 5` | traversal | `5` | Reference contexts averaged for displacement bar chart |
| `--top-k` | traversal, ablation | `8` | Top-K active dims by KL |
| `--include-dims 19` | traversal | `[19]` | Force extra dimensions into the grid |
| `--n-episodes 20` | ablation | `20` | Episodes averaged for causal ranking |
| `--compare-backbones` | structure | off | Overlay KL curves for GRU/LSTM/Transformer |
| `--kl-threshold 0.01` | structure | `0.01` | Min KL for “active” dimension count |

## How to read the outputs

### Axis 1 — `latent_traversal_*.png`

- **Grid:** one row per active dimension (top-K by KL), columns = increasing values of `z[i]`.
- **Interpretation:**
  - Ball moves **horizontally** along the row → `z[i]` likely encodes **x** (or `vx`).
  - **Vertical** movement → **y** (or `vy`).
  - Size/contrast change without movement → other attribute.
  - Nothing changes → inactive dimension or coupled to `h`.
- `latent_traversal_displacement_*.png` — Δx and Δy in pixels across the sweep.

### Axis 2 — `ablation_causal_vs_kl_*.png`

- **Method:** during context encoding, replace `z[i]` with **μ_prior[i]** at each frame (instead of μ_posterior), then standard open-loop imagination. Metric: position error vs. ground truth.
- **Left panel:** Δ error (ablated − baseline). Higher = more causally important.
- **Right panel:** reference KL per dimension.
- **Interpretation:**
  - High KL + high Δ → useful, actively used.
  - High KL + low Δ → active but redundant (e.g. `z[26]`).
  - Low KL + high Δ → rare but critical (e.g. `z[19]`).

### Axis 3 — `structure_*.png`

1. **`structure_kl_*.png`** — KL per dimension; count above red threshold = effective dimensionality.
2. **`structure_correlation_*.png`** — correlation between active `z` on a trajectory (redundancy vs. disentanglement).
3. **`structure_temporal_*.png`** — `z[i](t)` vs. ground-truth position/velocity.

With `--compare-backbones`: overlaid KL curves for GRU/LSTM/Transformer.

## Methodological choices

- **Active dimensions:** top-K by mean KL (min threshold 0.01 nats), computed on the posterior.
- **Traversal sweep range:** defaults to **temporal** — min/max of `z[i]` over a full episode (posterior means), plus 20% margin. Avoids the σ_q trap (posterior uncertainty ≈ 0.08 while `z[i]` swings over ~40).
- **Traversal reference state:** `h` and all other `z` dims fixed at one context frame; only `z[i]` varies. Effect can depend on reference instant; displacement bar chart averages over `--displacement-refs` contexts.
- **Ablation:** at each **context** step, `z[i] ← μ_prior[i]` (dimension does not receive observation signal), then normal open-loop imagination.
- **Ball position:** center of mass via `evaluation.ball_position`. If the dataset lacks a `positions` field, positions are extracted from frames automatically.

## Output location

Default: `interpretability/outputs/gru/` (override with `--output-dir`). Raw `.npz` files are saved alongside figures.

## Key findings (GRU, without occlusion)

- Latent tracks ball dynamics over time; partial x/y structure emerges in **groups** of correlated dimensions, not single pure axes.
- **KL ≠ causality:** `z[19]` — moderate KL, highest ablation impact; `z[26]` — high KL, largely redundant.
