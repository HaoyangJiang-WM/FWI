# Scope of the theoretical statements

This release makes structural and local statements, not a global inverse-recovery guarantee.

## Proposition 1: exact zero-control anchoring

For any maps `forward` and `return`, define

\[
C(x;b)=x+\operatorname{return}(\operatorname{forward}(x)+b)
-\operatorname{return}(\operatorname{forward}(x)).
\]

Then \(C(x;0)=x\) by cancellation. No invertibility, semigroup property, or accurate numerical round trip is required. This prevents zero control from exploiting a numerical cycle defect.

## Fixed-transform Gaussian invariance and its adaptive limit

If \(\xi\sim\mathcal N(0,I)\) and a fixed, sample-independent \(Q^TQ=I\), then \(Q\xi\sim\mathcal N(0,I)\). Every fixed D4 transform in the reported source screen preserves norm and pointwise isotropic-Gaussian log density.

The winning transform depends on the observation and realized source. Consequently, the distribution of adaptively selected sources need not remain Gaussian. We make no source-law preservation claim after selection.

## Local profiled coupling

After the discrete D4 decision is frozen, the endpoint problem still contains continuous source control `s`. Project fixed active constraints out and partition the reduced damped local Gauss--Newton model into `s` and recourse `v`. If the recourse block is nonsingular, block elimination gives

\[
(H_{ss}-H_{sv}H_{vv}^{-1}H_{vs})\delta s
=-g_s+H_{sv}H_{vv}^{-1}g_v.
\]

The Schur term is the reduced curvature after locally profiling recourse. The historical implementation realizes an approximation in a small blockwise subspace; this public core does not claim to reproduce the complete field-space solve.

## Linearized reachability

Let endpoint Jacobians with respect to source and Back controls be \(J_s\) and \(J_b\). Adding a Back block changes the local reachable subspace from `range(J_s)` to `range([J_s,J_b])`, which cannot be smaller and is strictly larger only when a new column direction lies outside `range(J_s)`. This does not guarantee descent, basin escape, or correct recovery.

## Local cross-seed sensitivity

For a regular constrained solution described by a KKT map \(G(z;\xi)=0\), the implicit-function theorem gives

\[
\frac{dz^\star}{d\xi}=-(D_zG)^{-1}D_\xi G.
\]

A positive lower bound on the smallest singular value of \(D_zG\) would give a local sensitivity bound. The five-seed panel only characterizes observed sensitivity; it neither verifies this singular-value assumption nor establishes a seed-invariance theorem.
