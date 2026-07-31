# Go Back to Move Forward

**Symmetry-screened, inference-time Multi-Back control for a frozen generative flow**

**Author:** Haoyang Jiang, William & Mary

[![Tests](https://github.com/HaoyangJiang-WM/FWI/actions/workflows/tests.yml/badge.svg)](https://github.com/HaoyangJiang-WM/FWI/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is a **minimal-core and audit release**, not an end-to-end FWI reproduction package. It documents how a frozen unconditional flow was adapted without updating generator weights or adding the inverse-task condition during prior training. The implemented protocol screens eight fixed D4 transforms using measurement features, then optimizes anchored trajectory interventions against the measurement residual.

The current evidence is a focused case study: one frozen optimal-transport FlowMap prior on synthetic full-waveform inversion (FWI). The interface is designed for differentiable trajectory-based generative priors; applicability to other priors and tasks remains to be tested.

## Why this problem is difficult

Deterministic amortized estimators trained with pointwise regression losses can approximate a conditional mean. When an inverse problem admits several plausible interface locations, averaging sharp candidates can smooth or displace the interface. This is not an architectural impossibility for U-Nets; it is a failure mode produced by ambiguity, pointwise losses, finite resolution, and the evaluated training protocol.

Task-specific conditional generators address ambiguity differently, but they commonly couple the conditioning modality, acquisition geometry, or forward operator to training. A new operator or condition can therefore require retraining or adaptation. Our alternative keeps an unconditional prior frozen and introduces the observation only through the inference-time objective.

## Method

Let $\Phi_{t\leftarrow s}$ be a frozen differentiable flow, $A$ a differentiable observation operator, and $y$ a measurement. Starting from $\xi\sim\mathcal N(0,I)$, each D4 transform receives the same fit-only source-probe budget:

$$
s_g^{\rm probe}=\arg\min_{s\in\mathcal B_{\rm probe}}
\ell_{\rm fit}\!\left(A_{\rm fit}(R_0(Q_g\xi+s)),y_{\rm fit}\right),
$$

$$
k_g^{\rm sel}=K\!\left(A_{\rm sel}(R_0(Q_g\xi+s_g^{\rm probe})),y_{\rm sel}\right),
\qquad
k_g^{\rm fit}=K\!\left(A_{\rm fit}(R_0(Q_g\xi+s_g^{\rm probe})),y_{\rm fit}\right),
$$

$$
g^\star=\operatorname*{argmin}^{\rm lex}_{g\in D_4}
\left(k_g^{\rm sel},k_g^{\rm fit},\operatorname{ord}_{D_4}(g)\right),
$$

$$
(s^\star,v^\star)=\arg\min_{s,v}\ell\!\left(A\big(R(Q_{g^\star}\xi+s,v)\big),y\right),
\qquad \mathcal A(s,v)\le1.
$$

- Each fixed `Q_g` is an orthogonal D4 action, preserving the realized source norm and pointwise isotropic-Gaussian log density. Because `g*` is selected adaptively, selected sources are not claimed to remain Gaussian-distributed.
- The source-selection split is disjoint from probe fitting but is used to choose `g*`; it is **not** an independent test split.
- One endpoint is reported per prespecified case–seed run; there is no truth-based post-hoc selection among seeds or endpoints.

An anchored Multi-Back event is

$$
C_{\tau,\rho}(x;b)=x+\Phi_{\tau\leftarrow\rho}
  (\Phi_{\rho\leftarrow\tau}(x)+b)
-\Phi_{\tau\leftarrow\rho}(\Phi_{\rho\leftarrow\tau}(x)),
\qquad \tau<\rho.
$$

Consequently, $C(x;0)=x$ exactly for any frozen numerical flow. Zero Back control cannot change the incoming state through a learned or numerical round-trip defect.

After active constraints are projected out, the reduced damped local Gauss–Newton model has Schur complement

$$
S=H_{ss}-H_{sv}H_{vv}^{-1}H_{vs}.
$$

This is a local interpretation of recourse-coupled curvature, not a global convergence or recovery theorem. See the [theory notes](docs/theory.md).

## Results

The fixed final protocol uses five evaluation cases crossed with five prespecified raw seeds, for 25 runs. Truth is loaded only after each endpoint and evaluation record are frozen.

| Metric | Recorded internal source stage | Final reopened endpoint |
|---|---:|---:|
| Mean MSE | 0.01424 | **0.00929** |
| Maximum observed MSE | 0.03219 | **0.02305** |
| Mean boundary F1 (tol=2) | — | **0.89046** |
| MSE improvements | — | 19 / 25 |
| Boundary-F1 improvements | — | 22 / 25 |
| Mean runtime | — | 2,978 s / sample |

The final stage is associated with a mean MSE reduction of 0.00495, but MSE worsens in 6/25 runs. Because the endpoint solve reopens source and recourse blocks together, this delta cannot be attributed causally to Multi-Back alone. The 25 cells form a crossed case-by-seed design, not 25 independent tasks.

### Internal source-stage → final endpoint attribution

![Paired source-stage to final endpoint attribution](assets/figures/source_to_multiback_attribution.png)

This is paired component attribution within one pipeline, not a compute-matched independent baseline or a causal 2 × 2 ablation.

### Five cases by five raw seeds

![Multi-seed reconstruction panel](assets/figures/multiseed_models.png)

![Multi-seed metric heatmaps](assets/figures/multiseed_metrics.png)

The machine-readable aggregate is in [`results/final_25_summary.json`](results/final_25_summary.json). The full 25 × 8 D4 decision ledger is in [`results/d4_public_h_decision_audit.json`](results/d4_public_h_decision_audit.json). Its canonical payload hash is `094903548fe3908de71f191d253593c8d67c2df43ac2f3f9667fa328dd25e980`.

## Repository map

```text
src/flowmap_multiback/   D4 source ranking, anchored Multi-Back,
                         public-H metrics, and reporting metrics
tests/                   algebraic, metric, and protocol tests
assets/figures/          release figures
results/                 frozen aggregate metrics and decision ledger
docs/                    theory, FWI appendix, and matched protocol
paper/                   Markdown and LaTeX short-paper sources
analyze_final_panel.py   fail-closed panel and decision-ledger validation
plot_attribution.py      regenerate the attribution figure
```

## Installation and validation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest -q
python analyze_final_panel.py
python plot_attribution.py
```

Run these commands from the repository root. The analyzer verifies the balanced 5 × 5 panel, the 25 × 8 decision ledger, record hashes, and canonical release hash.

## Reproducibility boundary

The algebraic components and aggregate audit are public. The complete historical research solver has intentionally not been copied into this clean draft because it is coupled to private checkpoints, data adapters, and cluster-era modules. A full FWI rerun still requires a documented stable solver API, the pretrained FlowMap checkpoint, benchmark tensors, the acoustic backend, and their licenses and hashes.

## Limitations and required next experiments

- Demonstrated on one 70 × 70 synthetic FWI benchmark and one frozen flow prior.
- Requires a differentiable observation operator and access to intermediate flow states/JVPs or VJPs.
- The source stage ranks eight fixed D4 transforms using measurement features.
- Nonconvex optimization provides no global recovery guarantee.
- The current source-versus-full comparison is component attribution, not a compute-matched 2 × 2 causal ablation.

The frozen definitions for the missing matched experiments are in [`docs/matched_experiment_protocol.md`](docs/matched_experiment_protocol.md). This protocol does not imply that those experiments have already run.

## Paper

The readable short paper is in [`paper/main.md`](paper/main.md), with LaTeX source in [`paper/main.tex`](paper/main.tex). FWI-specific formulation and implementation details are in [`docs/fwi_appendix.md`](docs/fwi_appendix.md).

## Citation

Please cite this software using [`CITATION.cff`](CITATION.cff). GitHub also provides a **Cite this repository** button derived from that file.

```bibtex
@software{jiang2026goback,
  author  = {Haoyang Jiang},
  title   = {Go Back to Move Forward: Measurement-Guided Recourse for a Frozen Generative Flow},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/HaoyangJiang-WM/FWI}
}
```

## License

Copyright © 2026 Haoyang Jiang. The software is released under the [MIT License](LICENSE). Dataset files, pretrained checkpoints, and external dependencies may be governed by separate licenses and are not relicensed by this repository.
