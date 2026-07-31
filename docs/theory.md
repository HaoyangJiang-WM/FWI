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

Partition a damped local Gauss--Newton or fixed-active-set KKT model into source `s` and recourse `y`. If the recourse block is nonsingular, block elimination gives

\[
(H_{ss}-H_{sy}H_{yy}^{-1}H_{ys})\delta s
=-g_s+H_{sy}H_{yy}^{-1}g_y.
\]

The Schur term removes local source directions that downstream recourse can immediately undo. In the implementation, this solve is realized in a small semantic Galerkin subspace; it is not a claim that the complete field-space Schur system is solved exactly.

## Linearized reachability

Let endpoint Jacobians with respect to source and Back controls be \(J_s\) and \(J_b\). Adding a Back block changes the local reachable subspace from `range(J_s)` to `range([J_s,J_b])`, which cannot be smaller and is strictly larger only when a new column direction lies outside `range(J_s)`. This does not guarantee descent, basin escape, or correct recovery.

## Local cross-seed sensitivity

For a regular constrained solution described by a KKT map \(G(z;\xi)=0\), the implicit-function theorem gives

\[
\frac{dz^\star}{d\xi}=-(D_zG)^{-1}D_\xi G.
\]

A positive lower bound on the smallest singular value of \(D_zG\) gives a local sensitivity bound. The empirical five-seed result is consistent with stability under the tested distribution; it is not a seed-invariance theorem.
