# Multi-Back Flow

**Optimization-Guided Flow Maps for Inverse Problems**

**Haoyang Jiang — William & Mary**

[![Tests](https://github.com/HaoyangJiang-WM/FWI/actions/workflows/tests.yml/badge.svg)](https://github.com/HaoyangJiang-WM/FWI/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Multi-Back Flow (MBF) adapts a **frozen unconditional Flow Map** to a new inverse problem at inference time. The generative prior remains fixed; measurement information enters only through the optimization objective.

The key idea is to replace a strictly monotone few-step trajectory with a controlled, non-monotone path. When a long Flow Map transition commits the reconstruction to a poor basin, MBF returns to an earlier generative time, applies a task-driven control, and moves forward again.

## Core idea

A standard controlled Flow Map trajectory follows

```math
z_{t_{i+1}}=\Phi_{t_{i+1}\leftarrow t_i}(z_{t_i}+B_i u_i),
\qquad 1=t_0>\cdots>t_K=0.
```

With only a few transitions, the trajectory exposes only a few control locations. The task gradient can therefore be poorly represented by the endpoint directions available to the monotone path.

MBF inserts a backward-forward control module:

```math
\mathcal{B}_{\tau,\rho}(z_\tau;b)
=\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(z_\tau)+Cb\right),
\qquad \tau<\rho.
```

The released implementation uses an anchored form:

```math
\mathcal{C}_{\tau,\rho}(x;b)
=x+\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(x)+b\right)
-\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(x)\right).
```

Hence $\mathcal{C}_{\tau,\rho}(x;0)=x$: zero Back control cannot alter the state through an imperfect numerical round trip.

Source, monotone, and Back controls are optimized jointly:

```math
\min_{c,b}\;
\mathcal{L}_y\bigl(x_{\mathrm{MBF}}(c,b)\bigr)
+\frac{1}{2}\lVert c\rVert_{\Lambda_c}^{2}
+\frac{1}{2}\lVert b\rVert_{\Lambda_b}^{2}.
```

## Relation to existing methods

| Method | Inference-time mechanism | Trajectory |
|---|---|---|
| DPS | Adds a likelihood gradient during reverse diffusion | Monotone diffusion chain |
| OGD | Optimizes controls along a frozen DDIM path | Monotone controlled chain |
| MBF | Jointly optimizes controls and reopens earlier generative times | **Non-monotone two-time Flow Map** |

MBF is therefore not a DPS update transplanted to Flow Maps. It is a trajectory-optimization method with explicit temporal recourse.

## FWI results

The current study uses one frozen Flow Map prior for synthetic full-waveform inversion, with five geological cases and five prespecified seeds per case.

### Baseline comparison

![Frozen baselines and selected MBF reconstructions](assets/figures/method_comparison.png)

This figure compares frozen task-trained baselines with one post-hoc lowest-MSE MBF seed per case. It is a visualization of reconstruction quality, not an inference-time seed-selection rule.

### Full five-case by five-seed panel

![All MBF reconstructions across five cases and five seeds](assets/figures/multiseed_models.png)

This panel shows all 25 prespecified runs and is the main view of cross-seed behavior.

Machine-readable results are available in [`results/final_25_summary.json`](results/final_25_summary.json). FWI-specific acquisition, evaluation, and audit details are documented in [`docs/fwi_appendix.md`](docs/fwi_appendix.md).

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
docs/                    Theory and FWI details
paper/                   Markdown and LaTeX paper sources
```

## Scope

This repository is a compact method, paper, and audit release. Reproducing the complete FWI solver additionally requires the pretrained Flow Map checkpoint, benchmark tensors, and differentiable acoustic backend used by the research code.

## Paper and documentation

- [Markdown paper](paper/main.md)
- [LaTeX paper](paper/main.tex)
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