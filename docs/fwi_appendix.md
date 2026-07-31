# FWI appendix: verified protocol details

This appendix separates FWI-specific choices from the generic frozen-flow interface. It reports only values verified from the frozen manifests, execution code, or final result ledger. Items unavailable in the public release are labeled explicitly.

## Model and observation interface

The frozen prior generates a normalized scalar geological field of shape `1 x 1 x 70 x 70`; decoded endpoints are clipped to `[-1, 1]` before the acoustic forward evaluation. The pretrained checkpoint recorded by the execution wrapper is `flow_lsd_step_3000.pt`. The checkpoint bytes and their license are not yet public, so the current release cannot independently reproduce full acoustic runs.

The differentiable acoustic backend returns public observations with axes `[batch, acquisition, time, receiver]`. Five acquisition groups are split before selection:

- fit acquisitions: `(0, 2, 4)`;
- heldout acquisitions: `(1, 3)`.

For a prediction `p` and public observation `h`, source candidates are ranked by the lexicographic key

\[
k(p,h)=(\bar r_{\mathrm{bb}},\max r_{\mathrm{bb}},r_{\Delta_t^2},r_{\Delta_t},r_{\mathrm{phase}}),
\]

where broadband terms are acquisition-wise residual RMS values, the next two entries are global RMS values of second and first time differences, and the phase proxy is one minus the mean cosine similarity between first-time-difference traces. The key is computed separately on fit and heldout acquisitions. Feasible records are ordered by heldout key, fit key, and frozen D4 declaration order. The exact implementation is public in `src/flowmap_multiback/public_h_metrics.py` and `public_h_selection.py`.

The physical wave equation, grid spacing, boundary treatment, time step, receiver count, source wavelet, and dimensional velocity conversion are encapsulated in the unavailable materialized backend. They must be released before claiming end-to-end reproducibility.

## Evaluation grid and exposure protocol

The final panel crosses five evaluation rows

```text
29864, 29748, 29544, 29952, 29694
```

with five raw seeds

```text
20268032, 20268132, 20268232, 20268332, 20268432.
```

The grid, D4 universe, selector, one-draw rule, and truth role were recorded before the remaining panel execution. The source-orbit manifest hash stored in every accepted cell is

```text
1d518e6ebf992f6e459162f2d01ff0fbaf4c74c9049fed27c12ec1111aff286a.
```

The public audit contains all 25 x 8 source records, chosen transforms, public keys, event order, decision hashes, and postdecision truth hashes. It verifies that all equal-budget D4 probes close before the source decision and that truth events occur after source and interface decisions. It does not prove that no related development case was ever inspected during earlier algorithm development; the paper therefore calls these evaluation cases rather than universally unseen cases.

## D4 source screen

Exactly one `N(0,I)` raw draw is made per cell. Eight fixed norm-preserving transforms are evaluated:

```text
identity, flip_x, flip_y, rot180,
transpose, transpose_flip_x, transpose_flip_y, transpose_rot180.
```

For each transform, one equal-budget coarse public-H source direction is optimized using bands `(0.125, 0.25, 0.5, 1.0)`. The best feasible nonidentity transform and identity are retained by the screen; the top-ranked transform supplies the cached raw source for the single downstream full solver. No second random draw is allowed.

Because transform selection depends on the source and observation, selected sources are not claimed to remain Gaussian-distributed. Every fixed candidate preserves source norm and pointwise isotropic-Gaussian log density.

## Back geometry and control schedule

The Back coordinate is `q=t^2`. The frozen geometry is

```text
q_tau = 0.50, q_rho = 0.75,
tau = sqrt(0.50), rho = sqrt(0.75).
```

The derived source and Back RMS radii are

\[
R_s=\sqrt{1-q_\tau}=\sqrt{0.5},\qquad
R_b=\sqrt{q_\rho-q_\tau}=0.5.
\]

The frozen band schedule is:

```text
source: (0.125, 0.25, 0.5, 1.0)
K1:     (0.25,)
K2:     (1.0,)
```

The historical solver records deterministic warm continuation, reopening of five physical blocks, strong-Wolfe closure, exact full replay, a five-candidate public-H interface refinement with scales `(0.25, 0.5, 0.75, 1.0)` plus identity, and one frozen reported endpoint. The clean public repository exposes the D4 rule, public-H metric, anchored Back primitive, reporting protocol, and decision ledger, but not the full historical five-block optimizer.

## Postdecision metrics

MSE is computed on normalized model fields. Boundary F1 first extracts the top 10% gradient-magnitude pixels independently in the model and truth. Precision and recall count an edge as matched when its nearest edge in the other image lies within two pixels; their harmonic mean is reported. Truth is used only after the endpoint decision record closes.

Right-third MSE is a diagnostic restricted to the rightmost third of the field. It is reported in the machine-readable aggregate but is not used for selection.

## Compute

Across the 25 final cells, mean runtime is 2,977.82 seconds, median 3,018.03 seconds, 95th percentile 3,668.09 seconds, and maximum 3,704.22 seconds. Runs used one GPU each. Exact GPU model, peak memory, and full forward/FlowMap call totals per cell should be added from the private ledgers before archival submission.

## Remaining reproducibility boundary

The public package verifies algebraic anchoring, D4 transforms, public-H metrics and ranking, postdecision metrics, the balanced aggregate, and the complete decision ledger. Full reproduction still requires publication of the pretrained checkpoint, materialized public observations, acoustic backend, complete five-block optimizer, configuration hashes, and licenses.
