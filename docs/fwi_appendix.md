# FWI-specific appendix draft

The main paper presents a generic differentiable inverse problem. This appendix should contain all domain-specific details before submission.

## Forward model

State the acoustic wave equation, parameterization, boundary conditions, source wavelet, receiver geometry, frequency schedule, normalization, and differentiable solver. Define the public measurement residual used for optimization and the postdecision-only truth metrics.

## Data and splits

Document the synthetic geology distribution, training/validation/test split, the five unseen case identifiers, any overlap audit, and the provenance of every pretrained baseline checkpoint. Explicitly state that the final panel crosses five cases with five raw seeds rather than sampling 25 independent geological cases.

## Controls

Report the source shell, connected source-orbit angles, ordinary and Back control stations, action radii, clock constraints, and why all controlled returns satisfy the empirically reliable time bound. Clarify that the continuous chart covers only the identity-connected flip/rotation subset, not transpose.

## Solver and protocol

Specify continuation stages, trust-region acceptance, damping, KKT stopping conditions, semantic Galerkin basis, call accounting, memory use, and wall time. All variables must remain live in the final exact-joint endpoint solve. Record all hashes before truth is opened; truth may be used only for postdecision MSE and boundary F1.

## Required submission tables

1. All 25 final MSE, F1, right-third MSE, calls, and runtime values.
2. Per-case mean and uncertainty across seeds.
3. Paired source-to-full deltas, including the 6/25 MSE regressions.
4. Matched source off/on x Multi-Back off/on ablation.
5. Baseline training loss, parameter count, training budget, data split, forward-call budget, and selection policy.
