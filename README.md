# Go Back to Move Forward

**Symmetry-screened, inference-time Multi-Back control for a frozen generative flow**

This is a **minimal-core and audit release**, not an end-to-end FWI reproduction package. It documents how a frozen unconditional flow was adapted without updating generator weights or adding the inverse-task condition during prior training. The implemented protocol screens eight fixed D4 transforms using measurement features, then optimizes anchored trajectory interventions against the measurement residual.

The current evidence is a focused case study: one frozen optimal-transport FlowMap prior on synthetic full-waveform inversion (FWI). The interface is designed for differentiable trajectory-based generative priors; applicability to other priors and tasks remains to be tested.

## Why this problem is difficult

Deterministic amortized estimators trained with pointwise regression losses can approximate a conditional mean. When an inverse problem admits several plausible interface locations, averaging sharp candidates can smooth or displace the interface. This is not an architectural impossibility for U-Nets; it is a failure mode produced by ambiguity, pointwise losses, finite resolution, and the evaluated training protocol.

Task-specific conditional generators address ambiguity differently, but they commonly couple the conditioning modality, acquisition geometry, or forward operator to training. A new operator or condition can therefore require retraining or adaptation. Our alternative keeps an unconditional prior frozen and introduces the observation only through the inference-time objective. We do not claim that GANs are necessarily conditional, nor that every foundation model exposes the interface required here.

## Method

Let \(\Phi_{t\leftarrow s}\) be a frozen differentiable flow, \(A\) a differentiable observation operator, and \(y\) a measurement. Starting from \(\xi\sim\mathcal N(0,I)\), each D4 transform receives the same fit-only source-probe budget:

\[
s_g^{\rm probe}=\arg\min_{s\in\mathcal B_{\rm probe}}
\ell_{\rm fit}\!\left(A(R_0(Q_g\xi+s)),y_{\rm fit}\right),
\qquad
g^\star=\operatorname*{lexargmin}_{g\in D_4}
K_{\rm sel}\!\left(A_{\rm sel}(R_0(Q_g\xi+s_g^{\rm probe})),y_{\rm sel}\right),
\qquad
(s^\star,v^\star)=\arg\min_{s,v}\ell\!\left(A\big(R(Q_{g^\star}\xi+s,v)\big),y\right),
\quad \mathcal A(s,v)\le1.
\]

- Each fixed `Q_g` is an orthogonal D4 action, preserving the realized source norm and pointwise isotropic-Gaussian log density. The probe is used only for equal-budget public-H ranking; the winning transformed raw source is cached and its continuous source control is reoptimized downstream. Because `g*` is selected adaptively, selected sources are not claimed to remain Gaussian-distributed.
- The source-selection split is disjoint from probe fitting but is used to choose `g*`; it is not an independent test split. After this decision, `s` is reoptimized as a continuous source control and `v` contains ordinary and two-Back controls. All five physical blocks are reopened by the endpoint solver. One endpoint is reported per prespecified case--seed run; there is no truth-based post-hoc choice among seeds or endpoints.
- An anchored Multi-Back event is

\[
C_{\tau,\rho}(x;b)=x+\Phi_{\tau\leftarrow\rho}
  (\Phi_{\rho\leftarrow\tau}(x)+b)
-\Phi_{\tau\leftarrow\rho}(\Phi_{\rho\leftarrow\tau}(x)),
\qquad \tau<\rho.
\]

Consequently, \(C(x;0)=x\) exactly for any frozen numerical flow. Zero Back control cannot change the incoming state through a learned or numerical round-trip defect.

After active constraints are projected out, partition the reduced damped local Gauss--Newton matrix into source `s` and recourse `v`. If the recourse block is nonsingular, elimination gives

\[
S=H_{ss}-H_{sv}H_{vv}^{-1}H_{vs}.
\]

This is a local interpretation of recourse-coupled curvature, not an algorithmic claim: the released protocol does not explicitly form or solve the full Schur system. It is not a global convergence or recovery theorem. See [theory notes](docs/theory.md).

## Results

The fixed final protocol uses five evaluation cases crossed with five prespecified raw seeds (25 runs). A frozen manifest records the grid before postdecision evaluation. Truth is loaded only after each endpoint and evaluation record are frozen.

| Metric | Recorded internal source stage | Final reopened endpoint |
|---|---:|---:|
| Mean MSE | 0.01424 | **0.00929** |
| Maximum observed MSE | 0.03219 | **0.02305** |
| Mean boundary F1 (tol=2) | -- | **0.89046** |
| MSE improvements | -- | 19 / 25 |
| Boundary-F1 improvements | -- | 22 / 25 |
| Mean runtime | -- | 2,978 s / sample |

The final stage is associated with a mean MSE reduction of 0.00495, but MSE worsens in 6/25 runs. Because the endpoint solve reopens source and recourse blocks together, this delta cannot be attributed causally to Multi-Back alone. The 25 cells are a crossed case-by-seed design, not 25 independent tasks: variance decomposition attributes 56.8% to case, 4.0% to seed, and 39.2% to interaction.

### Internal source-stage → final endpoint attribution

![Paired source-stage to final endpoint attribution](assets/figures/source_to_multiback_attribution.png)

This is paired component attribution within one pipeline, not a compute-matched independent baseline or a causal 2 x 2 ablation.

### Historical comparison diagnostic

An earlier five-case comparison figure remains under `assets/figures/` for provenance, but is intentionally not displayed as a release result: it uses a different protocol and lacks the matched compute and public provenance needed for a paper baseline.

### Five cases by five raw seeds

![Multi-seed reconstruction panel](assets/figures/multiseed_models.png)

![Multi-seed metric heatmaps](assets/figures/multiseed_metrics.png)

The machine-readable aggregate is in [`results/final_25_summary.json`](results/final_25_summary.json).
The full 25 x 8 D4 decision ledger, public probe-fit/selection keys, winners, event order, and manifest are in [`results/d4_public_h_decision_audit.json`](results/d4_public_h_decision_audit.json). Its canonical payload hash is `094903548fe3908de71f191d253593c8d67c2df43ac2f3f9667fa328dd25e980` (this is not the raw file-byte SHA-256).

## Repository map

```text
src/flowmap_multiback/   D4 source ranking, generic anchored Multi-Back,
                         and decision-protocol metrics
tests/                   algebraic and protocol tests
assets/figures/          release figures
results/                 frozen aggregate metrics
docs/                    theory and FWI-specific appendix
paper/                   short-paper source
analyze_final_panel.py   fail-closed 5 x 5 aggregation and plotting
```

## Installation and current reproducibility boundary

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[test]
pytest -q
python analyze_final_panel.py
python plot_attribution.py
```

Run these commands from the repository root. The analyzer verifies the balanced 5 x 5 panel, the 25 x 8 decision ledger, record hashes, and canonical release hash. The plotting command regenerates and overwrites the attribution figure.

The algebraic components and aggregate audit are public. The complete historical research solver has intentionally not been copied into this clean draft because it is coupled to private checkpoints, data adapters, and cluster-era modules. A full FWI rerun requires a documented stable solver API, the pretrained FlowMap checkpoint, and synthetic benchmark tensors. Those assets are not bundled, so this repository does **not** yet claim one-command end-to-end reproduction. Their license, download location, and SHA-256 values must be added before archival release.

## Limitations and required next experiments

- Demonstrated on one 70 x 70 synthetic FWI benchmark and one frozen flow prior.
- Requires a differentiable observation operator and access to intermediate flow states/JVPs or VJPs.
- The source stage ranks eight fixed D4 transforms using public features. This is an inference-time candidate decision, although it does not use postdecision truth.
- Nonconvex optimization provides no global recovery guarantee.
- The current source-versus-full comparison is component attribution, not a compute-matched 2 x 2 causal ablation. A publication-ready version should add source off/on x Multi-Back off/on, per-case uncertainty, and a matched modern generative inverse baseline.

The exact frozen arm definitions, compute-matching rules, truth-blind decision policy, required audit fields, and case-cluster reporting plan are specified in [`docs/matched_experiment_protocol.md`](docs/matched_experiment_protocol.md). This protocol does not imply that those experiments have already run.

## Paper

The readable short paper is in [`paper/main.md`](paper/main.md), with LaTeX source in [`paper/main.tex`](paper/main.tex). FWI-specific formulation and implementation details are deliberately separated into [`docs/fwi_appendix.md`](docs/fwi_appendix.md).

## Citation and license

Citation metadata and the final software license will be added after author and repository metadata are confirmed. Until then this is a review draft, not a licensed archival release.
