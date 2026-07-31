# Frozen protocol for the missing matched experiments

This document specifies the experiments required before the current review draft can make causal component or external-superiority claims. It is a protocol, not a report of completed results. No arm may be dropped, rerun selectively, or chosen using ground-truth image metrics.

## Shared panel and decision rule

- Evaluation cases, in frozen order: `29864, 29748, 29544, 29952, 29694`.
- Raw seeds, in frozen order: `20268032, 20268132, 20268232, 20268332, 20268432`.
- Every arm must run all 25 case–seed cells from the same raw ancestors.
- Probe-fit acquisitions: `(0, 2, 4)`.
- Source-selection acquisitions: `(1, 3)`. These are used for D4 selection and are not an independent test set.
- Optimizer stopping rules, action radii, forward discretization, precision, checkpoint, hardware class, and maximum acoustic/FlowMap call budgets must be identical wherever an arm contains the corresponding block.
- All choices and endpoints are frozen using the public measurement objective. MSE and boundary F1 are opened only after every arm for a cell is immutable.
- Failures, timeouts, infeasible runs, and numerical exceptions remain in the denominator and must be reported.

## Experiment A: matched D4 screen

This experiment isolates the finite source-screen decision.

| Arm | Source transform decision | Probe calls | Downstream solver |
|---|---|---:|---|
| A0 | Identity only | same total probe-call allowance as A1, spent on identity restarts fixed in advance | frozen final solver |
| A1 | Eight fixed D4 candidates, public selection key | eight equal probes | frozen final solver |
| A2 | Eight fixed D4 candidates, uniformly selected by a seed fixed before observations | eight equal probes | frozen final solver |

The primary contrast is paired A1–A0. A2 tests whether any benefit is explained by merely evaluating the orbit rather than measurement-guided selection. If unused calls cannot be spent without changing the identity algorithm, report both equal-call and native-cost A0 instead of hiding the compute difference.

Required outputs per cell: selected transform, all eight fit/selection keys, feasibility, probe and downstream call counts, wall time, final public objective, post-decision MSE/F1, endpoint hash, and complete event order.

## Experiment B: source control × anchored Multi-Back

This is the required 2 × 2 factorial experiment. The D4 decision must be frozen once per cell and shared across all four arms; otherwise the source-screen effect is confounded with the control effect.

| Arm | Continuous source control | Anchored Back controls | Ordinary downstream controls |
|---|---|---|---|
| B00 | off | off | on, frozen common budget |
| B10 | on | off | on, frozen common budget |
| B01 | off | on | on, frozen common budget |
| B11 | on | on | on, frozen common budget |

“Off” means the block is fixed exactly at zero, not optimized and later thresholded. “On” uses the same parameterization, radius, initialization rule, line search, and prespecified call budget as the corresponding block in B11. The endpoint solver may jointly reopen only the blocks marked on for that arm.

For an outcome $Y$ where lower is better, report the paired factorial interaction per cell,

$$
I_Y=(Y_{11}-Y_{10})-(Y_{01}-Y_{00}),
$$

along with the source main effect averaged over Back state and the Back main effect averaged over source state. For boundary F1, state the favorable direction explicitly rather than silently changing signs.

## Experiment C: external comparison

External baselines are admissible in the main paper only if their checkpoints, training split, preprocessing, input observations, evaluation cases, metric implementation, and inference compute are public. At minimum include:

1. one deterministic amortized inverse network used to test the conditional-mean/interface argument;
2. one modern generative inverse method that conditions a frozen prior at inference time; and
3. a conventional physics-based FWI optimizer initialized under a prespecified rule.

Report training compute separately from per-instance inference compute. Do not call a comparison “matched” merely because it uses the same five truth images.

## Statistical report

The geological case, not the individual seed cell, is the unit of task variation. For every primary paired contrast:

1. report all 25 paired cell differences;
2. average the five seeds within each case;
3. report the five case-level differences and their range;
4. report a case-cluster bootstrap interval by resampling cases and retaining all five seeds within each sampled case;
5. label the interval descriptive because only five independent cases are available;
6. report regression counts as well as aggregate improvements.

The bootstrap seed and number of replicates must be frozen before post-decision truth metrics are opened. No significance claim should be based on treating the 25 crossed cells as independent.

## Release gate

An experiment is paper-ready only when its manifest, raw per-cell records, canonical hashes, aggregation script, figure script, and protocol-deviation log are public and pass a fail-closed audit. Until then, the existing source-stage-to-final figure remains internal component attribution rather than a causal ablation.
