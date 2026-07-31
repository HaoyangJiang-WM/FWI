# Go Back to Move Forward

**Inference-time D4 source selection and Multi-Back control for a frozen generative flow**

This research release adapts a frozen, unconditional generative flow to the studied differentiable inverse problem without updating the generator and without injecting the inverse-task condition during prior training. It ranks a fixed D4 source orbit using public measurement features, then optimizes anchored trajectory interventions against the measurement residual.

The current evidence is a focused case study: one frozen optimal-transport FlowMap prior on synthetic full-waveform inversion (FWI). The interface is designed for differentiable trajectory-based generative priors; applicability to other priors and tasks remains to be tested.

## Why this problem is difficult

Deterministic amortized estimators trained with pointwise regression losses can approximate a conditional mean. When an inverse problem admits several plausible interface locations, averaging sharp candidates can smooth or displace the interface. This is not an architectural impossibility for U-Nets; it is a failure mode produced by ambiguity, pointwise losses, finite resolution, and the evaluated training protocol.

Task-specific conditional generators address ambiguity differently, but they commonly couple the conditioning modality, acquisition geometry, or forward operator to training. A new operator or condition can therefore require retraining or adaptation. Our alternative keeps an unconditional prior frozen and introduces the observation only through the inference-time objective. We do not claim that GANs are necessarily conditional, nor that every foundation model exposes the interface required here.

## Method

Let \(\Phi_{t\leftarrow s}\) be a frozen differentiable flow, \(A\) a differentiable observation operator, and \(y\) a measurement. Starting from \(\xi\sim\mathcal N(0,I)\), the reported implementation uses a truth-free public score \(S_{\rm pub}\):

\[
g^\star=\arg\min_{g\in D_4}S_{\rm pub}(Q_g\xi;y),
\qquad
(s^\star,v^\star)=\arg\min_{s,v}\ell\!\left(A\big(R(Q_{g^\star}\xi+s,v)\big),y\right),
\quad \mathcal A(s,v)\le1.
\]

- Each fixed `Q_g` is an orthogonal D4 action, preserving the realized source norm and pointwise isotropic-Gaussian log density. Because `g*` is selected adaptively, selected sources are not claimed to remain Gaussian-distributed.
- After the discrete D4 decision, `s` is a continuous source control and `v` contains ordinary and two-Back controls. All five physical blocks are reopened by the historical endpoint solver. One endpoint is reported per prespecified case--seed run; there is no truth-based post-hoc choice among seeds or endpoints.
- An anchored Multi-Back event is

\[
C_{\rho,\tau}(x;b)=x+\Phi_{\rho\leftarrow\tau}
  (\Phi_{\tau\leftarrow\rho}(x)+b)
-\Phi_{\rho\leftarrow\tau}(\Phi_{\tau\leftarrow\rho}(x)).
\]

Consequently, \(C(x;0)=x\) exactly for any frozen numerical flow. Zero Back control cannot change the incoming state through a learned or numerical round-trip defect.

After active constraints are projected out, partition the reduced damped local Gauss--Newton matrix into source `s` and recourse `v`. If the recourse block is nonsingular, elimination gives

\[
S=H_{ss}-H_{sv}H_{vv}^{-1}H_{vs}.
\]

This profiles recourse-coupled curvature into the local source block; it is not a global convergence or recovery theorem. See [theory notes](docs/theory.md).

## Results

The fixed final protocol uses five evaluation cases crossed with five prespecified raw seeds (25 runs). A frozen manifest records the grid before postdecision evaluation. Truth is loaded only after each endpoint and evaluation record are frozen.

| Metric | Source only | Source + Multi-Back |
|---|---:|---:|
| Mean MSE | 0.01424 | **0.00929** |
| Maximum observed MSE | 0.03219 | **0.02305** |
| Mean boundary F1 (tol=2) | -- | **0.89046** |
| MSE improvements | -- | 19 / 25 |
| Boundary-F1 improvements | -- | 22 / 25 |
| Mean runtime | -- | 2,978 s / sample |

Multi-Back reduces mean MSE by 0.00495, but it worsens MSE in 6/25 runs. The 25 cells are a crossed case-by-seed design, not 25 independent tasks: variance decomposition attributes 56.8% to case, 4.0% to seed, and 39.2% to interaction.

### Internal source-stage → final endpoint attribution

![Paired source-stage to final endpoint attribution](assets/figures/source_to_multiback_attribution.png)

This is paired component attribution within one pipeline, not a compute-matched independent baseline or a causal 2 x 2 ablation.

### Qualitative method comparison

![Method comparison](assets/figures/method_comparison.png)

This earlier five-case comparison is not the final 25-run panel. It is included as qualitative baseline context and must not be interpreted as a paired extension of the final protocol.

### Five cases by five raw seeds

![Multi-seed reconstruction panel](assets/figures/multiseed_models.png)

![Multi-seed metric heatmaps](assets/figures/multiseed_metrics.png)

The machine-readable aggregate is in [`results/final_25_summary.json`](results/final_25_summary.json).
The full 25 x 8 D4 decision ledger, public fit/heldout keys, winners, event order, and manifest are in [`results/d4_public_h_decision_audit.json`](results/d4_public_h_decision_audit.json). The release ledger hashes to `094903548fe3908de71f191d253593c8d67c2df43ac2f3f9667fa328dd25e980`.

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
pytest
```

The algebraic components and aggregate audit are public. The complete historical research solver has intentionally not been copied into this clean draft because it is coupled to private checkpoints, data adapters, and cluster-era modules. A full FWI rerun requires a documented stable solver API, the pretrained FlowMap checkpoint, and synthetic benchmark tensors. Those assets are not bundled, so this repository does **not** yet claim one-command end-to-end reproduction. Their license, download location, and SHA-256 values must be added before archival release.

## Limitations and required next experiments

- Demonstrated on one 70 x 70 synthetic FWI benchmark and one frozen flow prior.
- Requires a differentiable observation operator and access to intermediate flow states/JVPs or VJPs.
- The source stage ranks eight fixed D4 transforms using public features. This is an inference-time candidate decision, although it does not use postdecision truth.
- Nonconvex optimization provides no global recovery guarantee.
- The current source-versus-full comparison is component attribution, not a compute-matched 2 x 2 causal ablation. A publication-ready version should add source off/on x Multi-Back off/on, per-case uncertainty, and a matched modern generative inverse baseline.

## Paper

The short-paper draft is in [`paper/main.tex`](paper/main.tex). FWI-specific formulation and implementation details are deliberately separated into [`docs/fwi_appendix.md`](docs/fwi_appendix.md).

## Citation and license

Citation metadata and the final software license will be added after author and repository metadata are confirmed. Until then this is a review draft, not a licensed archival release.
