# Multi-Back Flow

**Optimization-Guided Flow Maps for Inverse Problems**

**Haoyang Jiang — William & Mary**

[![Tests](https://github.com/HaoyangJiang-WM/FWI/actions/workflows/tests.yml/badge.svg)](https://github.com/HaoyangJiang-WM/FWI/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Paper:** For the complete formulation, local analysis, and FWI study, see the [full paper in Markdown](paper/main.md) or the [LaTeX source](paper/main.tex).

Multi-Back Flow (MBF) is an inference-time framework for adapting a **frozen unconditional Flow Map** to a new inverse problem. The generative prior remains fixed, while the measurement operator and observations enter only through the inference objective.

The central idea is to replace a strictly monotone few-step trajectory with a controlled, non-monotone path. When a long Flow Map transition commits the reconstruction to a poor basin, MBF returns to an earlier generative time, applies a task-driven control, and moves forward again.

## Motivation

Inverse problems require both:

- **measurement consistency:** the reconstruction should explain the observations;
- **structural plausibility:** the reconstruction should remain consistent with the distribution of realistic solutions.

### Supervised inverse models

Supervised methods such as U-Net, InversionNet, and neural operators directly learn a map

```math
f_\theta:y\mapsto x.
```

They provide fast inference, but usually require task-specific paired data and are tied to the forward operators, acquisition geometries, noise levels, and parameter ranges represented during training.

They can also struggle when one observation is compatible with several distinct solutions. Under squared error, the population-optimal deterministic predictor is

```math
f^\star(y)=\mathbb E[x\mid y].
```

If the conditional distribution contains several sharp structures at different locations, their average may produce blurred, displaced, or superposed interfaces. This is not an inherent limitation of every U-Net reconstruction; it is a possible failure mode of deterministic pointwise regression under multimodal ambiguity.

### Conditional generative models

Conditional GANs and conditional diffusion models can represent richer solution distributions and often produce sharper reconstructions. However, the conditioning interface is typically defined during training. A new forward operator or acquisition setting may therefore require retraining or adaptation.

### Frozen generative priors

A more modular alternative is to train an unconditional generative model as a reusable structural prior and introduce the observation only at inference time.

Diffusion posterior methods such as DPS add likelihood information along the reverse diffusion process, but often require long denoising chains and repeated local corrections across many noise levels.

Optimization-Guided Diffusion (OGD) instead optimizes controls along a frozen DDIM trajectory. This turns inference into trajectory optimization, but the controls remain attached to a prescribed monotone denoising path.

MBF addresses a specific limitation of few-step trajectory optimization:

> Does a short, fixed, monotone path provide enough task-relevant control directions to escape an incorrect reconstruction basin?

## What is a Flow Map?

A two-time Flow Map directly transports a state between arbitrary generative times:

```math
\Phi_{t\leftarrow r}:\mathcal X_r\rightarrow\mathcal X_t.
```

For a monotone schedule

```math
1=t_0>t_1>\cdots>t_K=0,
```

few-step generation uses

```math
z_{t_{i+1}}
=
\Phi_{t_{i+1}\leftarrow t_i}(z_{t_i}).
```

Unlike a velocity model that must be numerically integrated over each interval, a learned Flow Map can directly perform long temporal transitions. It therefore supports one-step or few-step generation and allows the number of sampling steps to be selected after training.

Its two-time interface also makes generative time available as a bidirectional optimization coordinate, which is the key capability used by MBF.

## Why can a few-step trajectory become locked?

A controlled monotone Flow Map trajectory can be written as

```math
z_{t_{i+1}}
=
\Phi_{t_{i+1}\leftarrow t_i}
\left(z_{t_i}+B_i u_i\right).
```

Few-step generation reduces the number of Flow Map evaluations, but it also creates:

1. longer transitions between adjacent control locations;
2. fewer locations where task information can modify the trajectory.

An early long transition may therefore determine the reconstruction basin reached by the remaining path.

Let \(c\) denote the ordinary trajectory controls, \(J_c\) the endpoint Jacobian with respect to those controls, and \(g\) the task gradient at the current endpoint. The available first-order descent is governed by

```math
\left\lVert J_c^\top g\right\rVert.
```

When this quantity is small, the endpoint may still have substantial measurement error, while the current monotone parameterization provides little useful local descent. We refer to this situation as **few-step basin lock**.

## Multi-Back Flow

MBF introduces a backward-forward control module between a later time \(\tau\) and an earlier generative time \(\rho\):

```math
\mathcal B_{\tau,\rho}(z_\tau;b)
=
\Phi_{\tau\leftarrow\rho}
\left(
\Phi_{\rho\leftarrow\tau}(z_\tau)+Cb
\right),
\qquad \tau<\rho.
```

The module:

1. maps the current state back to an earlier, higher-noise generative time;
2. inserts a Back control \(b\) at that scale;
3. transports the modified state forward again;
4. continues toward the final reconstruction.

This is not a random restart or a fresh-noise annealing step. The backward-forward transition remains part of the same deterministic controlled trajectory.

Source, monotone, and Back controls are optimized jointly:

```math
\min_{c,b}\;
\mathcal L_y
\left(
x_{\mathrm{MBF}}(c,b)
\right)
+
\frac{1}{2}\left\lVert c\right\rVert_{\Lambda_c}^{2}
+
\frac{1}{2}\left\lVert b\right\rVert_{\Lambda_b}^{2}.
```

The Back controls enlarge the locally reachable endpoint space from the directions generated by \(J_c\) to those generated jointly by \(J_c\) and \(J_b\). They can therefore provide task-relevant descent directions that are absent from the current monotone path.

The released implementation also uses **zero-control anchoring**, ensuring that a zero Back control leaves the incoming state unchanged even when the learned backward-forward round trip is numerically imperfect. The complete definition is given in the [paper](paper/main.md) and [theory notes](docs/theory.md).

## Relation to existing methods

| Method | How measurements enter | Trajectory | Task-specific training |
|---|---|---|---|
| Supervised inverse network | Learns \(y\mapsto x\) from paired data | Single network evaluation | Yes |
| Conditional generative model | Learns the task condition during training | Conditional generative path | Usually |
| DPS | Adds a likelihood gradient during reverse diffusion | Monotone diffusion chain | No |
| DAPS | Alternates clean-space posterior updates and noise annealing | Repeated denoise/re-noise stages | No |
| OGD | Optimizes controls along a frozen DDIM path | Monotone controlled chain | No |
| MBF | Jointly optimizes controls and reopens earlier generative times | **Non-monotone two-time Flow Map** | No |

MBF is not a DPS update transplanted to Flow Maps. DPS modifies the current reverse-diffusion step using approximate posterior guidance. MBF treats the entire generative path as a trajectory-optimization problem and explicitly returns to earlier generative scales.

MBF also differs from DAPS. DAPS introduces fresh noise to decouple consecutive noise levels, whereas MBF constructs deterministic backward-forward control paths using the same frozen Flow Map.

## What is FWI?

Full-Waveform Inversion (FWI) reconstructs subsurface physical properties from recorded seismic waveforms.

A source generates a wavefield \(p(\mathbf r,t)\), which propagates through a medium with spatially varying velocity \(v(\mathbf r)\). In a constant-density acoustic model, the wave equation can be written as

```math
\frac{1}{v(\mathbf r)^2}
\frac{\partial^2 p(\mathbf r,t)}{\partial t^2}
-
\nabla^2 p(\mathbf r,t)
=
s(\mathbf r,t),
```

where:

- \(v(\mathbf r)\) is the unknown velocity model;
- \(p(\mathbf r,t)\) is the acoustic pressure field;
- \(s(\mathbf r,t)\) is the seismic source.

Receivers sample the simulated wavefield through an observation operator \(P\):

```math
y
=
Pp(v)+\eta
=
\mathcal A(v)+\eta,
```

where \(y\) contains the recorded seismic traces and \(\eta\) represents measurement noise and modeling error.

Classical FWI estimates the velocity model by minimizing waveform mismatch:

```math
\min_v\;
\frac{1}{2}
\left\lVert
\mathcal A(v)-y
\right\rVert_2^2
+
\lambda\mathcal R(v),
```

where \(\mathcal R(v)\) is an optional regularizer.

The gradient is obtained by differentiating through the wave equation, commonly using an adjoint-state method:

```math
\nabla_v\mathcal L_y(v)
=
D\mathcal A(v)^\top
\left(
\mathcal A(v)-y
\right).
```

FWI is highly nonconvex. The oscillatory waveform loss can contain many local minima, and an incorrect initial model may lead to **cycle skipping**, where simulated and observed waveforms align with the wrong oscillation cycles.

The inverse problem can also be ambiguous: different velocity structures may produce similar receiver measurements. At the same time, realistic subsurface models often contain sharp layers, faults, and interfaces. Deterministic pointwise regression may smooth these structures under multimodal ambiguity, while unconstrained waveform fitting may converge to a measurement-consistent but geologically implausible solution.

FWI is therefore a useful test of whether a frozen generative prior and non-monotone trajectory control can jointly preserve structural plausibility and measurement consistency.

## FWI case study

The current study applies a frozen unconditional Flow Map prior to synthetic acoustic FWI. It uses:

- a Flow Map trained on \(70\times70\) geological velocity models;
- a differentiable acoustic wave-equation forward operator;
- five geological evaluation cases;
- five prespecified raw seeds for each case.

### Baseline comparison

![Frozen baselines and selected MBF reconstructions](assets/figures/method_comparison.png)

The figure compares frozen task-trained baselines with MBF reconstructions.

For each geological case, the displayed MBF result is the post-hoc lowest-truth-MSE reconstruction among the five prespecified seeds. The figure is intended to visualize reconstruction structure; it is not an inference-time seed-selection rule or a compute-matched superiority claim.

### Complete five-case by five-seed panel

![All MBF reconstructions across five cases and five seeds](assets/figures/multiseed_models.png)

This panel shows all 25 prespecified runs and is the main view of reconstruction variability across raw seeds.

Machine-readable results are available in:

- [`results/final_25_summary.json`](results/final_25_summary.json)

FWI acquisition settings, evaluation definitions, and audit details are documented in:

- [`docs/fwi_appendix.md`](docs/fwi_appendix.md)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest -q
python analyze_final_panel.py
```

## Repository structure

```text
src/flowmap_multiback/   Core Multi-Back and evaluation utilities
tests/                   Algebraic and protocol tests
assets/figures/          Main result figures
results/                 Frozen aggregate records
docs/                    Theory and FWI implementation details
paper/                   Markdown and LaTeX paper sources
```

## Scope

This repository is a compact method, paper, and audit release.

A complete FWI reproduction additionally requires:

- the pretrained Flow Map checkpoint;
- benchmark tensors and split definitions;
- the differentiable acoustic forward backend;
- the complete optimization solver connecting these components.

## Paper and documentation

- [Full paper in Markdown](paper/main.md)
- [LaTeX source](paper/main.tex)
- [Theory notes](docs/theory.md)
- [FWI appendix](docs/fwi_appendix.md)

## Citation

```bibtex
@software{jiang2026multiback,
  author  = {Haoyang Jiang},
  title   = {Multi-Back Flow: Optimization-Guided Flow Maps for Inverse Problems},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/HaoyangJiang-WM/FWI}
}
```

## License

Copyright © 2026 Haoyang Jiang. Released under the [MIT License](LICENSE).