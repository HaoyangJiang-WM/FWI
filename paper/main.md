# Go Back to Move Forward: Inference-Time Recourse for a Frozen Generative Flow

**Anonymous Authors**

## Abstract

Frozen generative models encode structural priors, yet adapting them to new inverse problems remains difficult. In ambiguous inverse problems, deterministic regressors trained with pointwise losses can average across plausible interface locations. Task-specific conditional generators can avoid some averaging but often couple the conditioning modality or acquisition operator to training. We introduce **Source-Orbit Multi-Back**, an inference-time interface for a frozen unconditional flow map. The reported implementation ranks eight norm-preserving D4 source transforms using public measurement features, then optimizes anchored trajectory controls against a differentiable observation loss. Multi-Back cycles revisit earlier flow scales and reduce exactly to identity at zero Back control. One endpoint is reported for every prespecified case–seed run without truth-based post-hoc selection.

On five synthetic full-waveform inversion (FWI) evaluation cases crossed with five seeds, the final endpoints obtain mean MSE **0.00929**, maximum observed MSE **0.02305**, and mean boundary F1 **0.8905**. Relative to the internal source stage, the final endpoint improves MSE in 19/25 cells and F1 in 22/25, reducing mean MSE by **0.00495**. This is component attribution for one frozen flow prior, not a compute-matched causal ablation; broader model and task generalization remains future work.

## 1. Introduction

Learned inverse maps such as U-Nets [1] amortize inference into one forward pass. Under squared error, the population minimizer is a conditional mean. When observations leave several sharp interface locations plausible, their mean need not contain a sharp interface. This is a consequence of ambiguity and the training objective—not an architectural impossibility for U-Nets.

Conditional adversarial translation [2] offers another route, but a task-specific conditional model commonly encodes the measurement modality and data pairing during training. Changing acquisition or physics can therefore require adaptation. The limitation is not that GANs are inherently conditional; it is that many inverse-problem GAN implementations entangle a particular condition or operator with training, which becomes difficult to scale across heterogeneous conditions in a general-purpose generative model.

Pretrained score and flow models provide unconditional priors that can instead be conditioned at inference time [3,4]. Diffusion posterior sampling, for example, inserts a likelihood gradient into a pretrained sampling process [5]. This motivates testing whether non-monotone trajectory recourse can add useful inference-time directions while explicitly preventing zero-control improvements from numerical round-trip defects.

We treat the frozen trajectory as a small controllable system. The reported pipeline combines:

1. public-objective selection over fixed D4 source transforms;
2. exactly identity-anchored non-monotone recourse; and
3. profiled endpoint optimization.

The inverse-task condition enters only at inference time; generator weights remain frozen. We demonstrate one optimal-transport FlowMap prior on synthetic FWI. The broader foundation-model relevance is a hypothesis: the interface can apply to differentiable trajectory-based generative priors that expose intermediate states and derivatives, but this paper does not demonstrate a foundation model.

## 2. Method

Let $\Phi_{t\leftarrow s}$ be a frozen differentiable flow, $A$ a differentiable observation operator, and $y$ a measurement. Given $\xi\sim\mathcal N(0,I)$ and fixed D4 action set $G$, every transform first receives the same fit-only source-probe budget:

$$
s_g^{\mathrm{probe}}
=\arg\min_{s\in\mathcal B_{\mathrm{probe}}}
\ell_{\mathrm{fit}}\!\left(A(R_0(Q_g\xi+s)),y_{\mathrm{fit}}\right).
$$

Candidates are ranked using held-out public measurement features:

$$
g^\star=\operatorname*{lexargmin}_{g\in G}
K_{\mathrm{heldout}}(Q_g\xi+s_g^{\mathrm{probe}}).
$$

The winning transformed raw source is cached; its continuous source control and trajectory recourse are then reoptimized jointly:

$$
(s^\star,v^\star)=\arg\min_{s,v}
\ell\!\left(A(R(Q_{g^\star}\xi+s,v)),y\right)
\quad\text{s.t.}\quad \mathcal A(s,v)\le 1.
$$

The probe displacement is used only to rank equally budgeted candidates. After the discrete decision, $s$ is a continuous source control and $v$ contains ordinary and two-Back recourse controls. The historical endpoint solver reopens all five physical blocks in one objective. One chronological run returns one reported endpoint. Optimizer decisions use the public objective; ground truth is unavailable.

### 2.1 Norm-preserving D4 source screen

Each fixed $Q_g$ is an orthogonal permutation, so $\|Q_g\xi\|=\|\xi\|$ and the pointwise isotropic-Gaussian log density is unchanged. Because $g^\star$ depends on $(\xi,y)$, the distribution of selected sources need not remain Gaussian. We claim norm and density-level-set preservation per candidate, not preservation of the adaptively selected source law.

The public key is the five-tuple *(broadband mean, broadband maximum, second time-difference RMS, first time-difference RMS, phase proxy)*, computed independently on fit and held-out acquisition splits. Feasible D4 records are ordered lexicographically by held-out key, fit key, and frozen D4 order. The public release includes all $25\times8$ records and selected transforms.

### 2.2 Exactly anchored Multi-Back

Suppose $x$ is at lower-noise time $\tau$ and $\tau<\rho$. We first move to the higher-noise time $\rho$, insert control $b$, and return:

$$
C_{\tau,\rho}(x;b)=x+
\Phi_{\tau\leftarrow\rho}\!\left(\Phi_{\rho\leftarrow\tau}(x)+b\right)
-\Phi_{\tau\leftarrow\rho}\!\left(\Phi_{\rho\leftarrow\tau}(x)\right).
$$

The controlled return is measured relative to its passive numerical round trip.

**Anchoring property.** For any frozen numerical maps in the equation above,

$$C_{\tau,\rho}(x;0)=x,$$

because the two return terms cancel. No invertibility or semigroup assumption is required. Zero Back control therefore cannot change the incoming state by exploiting a cycle defect. This is an algebraic identity, not a recovery theorem.

### 2.3 Profiled local coupling

After projecting out fixed active constraints, partition a reduced damped local Gauss–Newton matrix into source $s$ and recourse $v$. If $H_{vv}$ is nonsingular, elimination gives the Schur complement

$$
S=H_{ss}-H_{sv}H_{vv}^{-1}H_{vs}.
$$

This accounts for curvature coupled through locally profiled recourse. The implementation uses a blockwise Galerkin basis rebuilt after accepted nonlinear steps. The statement is local and supplies neither global convergence nor recovery guarantees.

## 3. FWI Case Study

We use a frozen optimal-transport FlowMap trained as an unconditional $70\times70$ geological prior. A differentiable acoustic forward model maps the endpoint to receiver observations. Five evaluation rows (29864, 29748, 29544, 29952, 29694) are crossed with raw seeds 20268032–20268432. The algorithm, grid, action budget, and hashes are recorded in a frozen manifest before the remaining panel is run. Truth is opened only after each endpoint record is frozen and is used for post-decision MSE and boundary F1.

FWI-specific acquisition, checkpoint, split, exposure-history, and metric details are separated into the [archival appendix](../docs/fwi_appendix.md), keeping the main formulation task-agnostic.

### 3.1 Reconstruction comparison across seeds

![Truth and final reconstructions for five evaluation cases crossed with five prespecified raw seeds.](../assets/figures/multiseed_models.png)

*Figure 1. Truth and final reconstructions for five evaluation cases crossed with five prespecified raw seeds.*

### 3.2 Cross-seed metrics

![Five evaluation cases crossed with five prespecified raw seeds.](../assets/figures/multiseed_metrics.png)

*Figure 2. Five evaluation cases (rows) crossed with five prespecified raw seeds (columns). One endpoint is reported per cell without truth-based post-hoc seed or endpoint selection.*

The final endpoints obtain mean/maximum-observed MSE **0.00929/0.02305**, mean/minimum boundary F1 **0.8905/0.7526**, and mean right-third MSE **0.00907**. Mean runtime is 2,978 seconds per cell. The 25 cells form a crossed design over only five geological cases; they are not 25 independent tasks.

| Metric | Source stage | Final endpoint |
|---|---:|---:|
| Mean MSE | 0.01424 | **0.00929** |
| Maximum observed MSE | 0.03219 | **0.02305** |
| MSE improved | — | 19/25 |
| Boundary F1 improved | — | 22/25 |

*Table 1. Internal source-stage and final-endpoint metrics. The source stage is not a compute-matched independent baseline.*

### 3.3 Source-to-Multi-Back component attribution

![Internal source-stage to final-endpoint MSE changes.](../assets/figures/source_to_multiback_attribution.png)

*Figure 3. Internal source-stage to final-endpoint MSE changes. Blue denotes improvement and red regression. This is not a compute-matched independent baseline.*

Relative to the internal source stage, the final endpoint reduces mean MSE by **0.00495**. It improves MSE in 19/25 cells and worsens it in 6/25; boundary F1 improves in 22/25. This supports component attribution inside the implemented pipeline but does not identify a causal source–Multi-Back interaction. A compute-matched source off/on $\times$ Multi-Back off/on experiment is still needed for that claim.

## 4. Limitations and Conclusion

This technical draft contains only five geological cases, one synthetic acquisition family, and one frozen flow prior. It requires differentiable physics and intermediate trajectory access and is computationally expensive. Its D4 stage performs public-objective candidate selection, and the nonconvex solver has no global recovery guarantee. Matched external baselines, a source off/on $\times$ Multi-Back off/on experiment, uncertainty over more independent cases, and a licensed end-to-end release remain necessary for a formal submission.

Within this scope, inference-time source selection and exactly anchored recourse inject measurements without inverse-task retraining. The final endpoint has lower aggregate mean and maximum observed MSE than its recorded internal source stage; this is attribution, not a causal or compute-matched comparison. Broader applicability remains a hypothesis to test on new operators and priors.

## References

1. O. Ronneberger, P. Fischer, and T. Brox. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” *MICCAI*, 2015.
2. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. “Image-to-Image Translation with Conditional Adversarial Networks.” *CVPR*, 2017.
3. Y. Song et al. “Score-Based Generative Modeling through Stochastic Differential Equations.” *ICLR*, 2021.
4. Y. Lipman et al. “Flow Matching for Generative Modeling.” *ICLR*, 2023.
5. H. Chung et al. “Diffusion Posterior Sampling for General Noisy Inverse Problems.” *ICLR*, 2023.

---

Code, the auditable D4 decision ledger, exact metric definitions, and reproducibility notes are available in the [project README](../README.md).
