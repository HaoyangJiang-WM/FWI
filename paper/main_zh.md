# Multi-Back Flow：用于逆问题的优化引导 Flow Map

**Haoyang Jiang — William & Mary**

[English paper](main.md) · [项目 README](../README.md)

## 摘要

预训练生成模型可以为逆问题提供可复用的先验，但如何将其适配到新的测量算子仍然具有挑战。监督式逆问题网络能够快速推理，但其重建规则通常与训练阶段所覆盖的算子和采集设置绑定；在多模态歧义下，逐点回归还可能偏向条件均值。扩散后验方法通过在推理阶段引入测量信息来避免针对具体任务重新训练，但通常需要较长的去噪链，或多轮后验细化。Optimization-Guided Diffusion（OGD）则沿冻结的 DDIM 轨迹优化控制变量，在生成先验与部署目标之间提供了一种自然接口。

我们提出 **Multi-Back Flow（MBF）**。该方法将优化引导生成从固定的去噪链扩展到一条基于冻结两时间 Flow Map 的非单调轨迹。Flow Map 通过跨越较长时间区间的转移实现少步生成，但少步单调路径只提供少量控制位置；当任务梯度无法被终点控制 Jacobian 充分表示时，轨迹可能出现局部锁定。MBF 在轨迹中插入受控的后退—前进转移，将状态送回更早的生成时间，在那里加入新的控制，从而打开单调参数化无法提供的终点方向。该方法在保持生成模型冻结的同时，联合优化源控制、单调轨迹控制与 Back controls。我们给出 basin lock 的局部下降刻画，并说明 Multi-Back controls 如何恢复一阶下降方向。当前实现通过一个合成全波形反演实验进行验证；跨算子与跨领域的更广泛验证仍属于未来工作。

## 1. 引言

逆问题的目标是根据不完整、带噪或间接观测 $y$ 恢复未知对象 $x$。一个有效的重建结果既要满足测量一致性，也要具有合理的结构先验。监督式确定性逆问题网络，例如 U-Net 类编码器—解码器和 InversionNet，直接从配对数据中学习映射 $f_\theta:y\mapsto x$，因此推理速度很快 [1,2]。然而，它们的重建规则与训练期间出现的观测分布和采集设置绑定。此外，在平方误差下，群体最优的确定性预测器为 $f^\star(y)=\mathbb E[x\mid y]$；当同一观测对应多个清晰但不同的解时，这个条件均值可能模糊或叠加不同模式 [3]。FNO 等神经算子学习函数空间之间的映射，可以高效摊销参数化算子族上的重复求解，但其逆问题应用仍依赖于训练所覆盖的算子、参数范围和离散化方式 [4]。pix2pix 和 VelocityGAN 等条件生成模型可以通过对抗训练生成更清晰的结果，但其条件接口仍与任务相关的配对数据绑定 [5,6]。

预训练无条件生成模型提供了一种更模块化的选择：模型只学习解的先验，而新的测量算子仅在推理阶段进入。基于扩散模型的逆问题方法已经从多条路线探索了这一思路。解析方法利用特殊的线性结构；MCG、DPS 和 $\Pi$GDM 等流形或似然引导方法通过测量信息修改反向过程；优化与重采样方法则在干净空间或潜空间中施加更强的数据一致性 [7–11]。这些方法显著拓展了冻结先验的复用能力，但通常需要在许多噪声层级上重复调用去噪器、正演算子或内部优化过程。它们还继承了局部耦合去噪转移带来的强路径依赖。DAPS 明确指出了早期全局错误难以纠正的问题，并通过干净空间后验采样与重新加噪退火来解耦相邻噪声层级 [12]。

OGD 提供了另一种优化视角：它不直接向样本添加外部 guidance gradient，而是将固定 DDIM 轨迹中的扰动替换为可优化控制，从而把推理转化为冻结生成器上的约束轨迹优化 [13]。这一结构很适合一般逆问题和约束生成，但控制变量仍依附于预先指定的单调去噪链。Flow Map Matching 学习底层生成动力学的两时间传输映射，支持任意生成时间之间的长距离转移，因此能够在训练后自由选择一步或少步生成 [14]。用 Flow Map 替换 DDIM 转移可以降低生成代价，但并未消除从源分布到数据端点的单调路径限制。

在少步优化中，这一限制尤为重要。对于调度 $1=t_0>\cdots>t_K=0$，至少有一个时间区间跨度不小于 $1/K$，而普通控制只有 $K$ 个注入位置。一个早期控制会经过较长的转移传播，并可能在很大程度上决定后续轨迹落入哪个 basin。当端点附近的任务梯度在单调控制所产生的终点方向上的投影很弱时，就会出现 **few-step basin lock**。为解决这一问题，我们提出 **Multi-Back Flow（MBF）**。MBF 将生成时间视为一个双向优化坐标：它把轨迹状态送回更早的生成时间，在那里施加任务驱动的 Back control，然后再次向数据端点传输。多个 Back 模块可以插入到不同时间尺度，并与原有轨迹控制联合优化。

从概念上看，MBF 将优化引导生成从固定链上的控制扩展为两时间 flow graph 上的非单调控制。不同于后验重新加噪、restart sampling 或 DAPS 式退火，MBF 不从新的噪声后验状态重新采样；后退—前进转移本身就是确定性受控轨迹的一部分。本文贡献如下：第一，提出少步 Flow Map 的优化引导形式；第二，通过投影任务梯度对 few-step basin lock 进行局部刻画；第三，提出 Multi-Back temporal recourse，在更早生成尺度增加新的终点控制方向；第四，给出一个无需重新训练生成先验、可容纳一般可微目标与约束的联合推理框架。

## 2. 预备知识

### 2.1 冻结生成先验下的逆问题

我们考虑观测模型 $y=\mathcal A(x^\star)+\eta$，其中 $x^\star\in\mathcal X$ 是未知目标，$\mathcal A:\mathcal X\to\mathcal Y$ 是已知正演算子，$\eta$ 是测量噪声。任务损失 $\mathcal L_y(x)=\ell(\mathcal A(x),y)$ 衡量数据一致性；对于高斯噪声，可取 $\mathcal L_y(x)=\|\mathcal A(x)-y\|^2/(2\sigma_y^2)$。附加约束可写成 $g_j(x)\le0$ 和 $h_\ell(x)=0$。生成先验独立于 $\mathcal A$ 进行预训练，并在推理期间保持冻结。

### 2.2 Optimization-Guided Diffusion

一个随机 DDIM 转移可以概略写成

```math
x_{k-1}=\mu_\theta(x_k,k)+\sigma_k\omega_k,
\qquad \omega_k\sim\mathcal N(0,I).
```

OGD 用可优化修正 $\delta_k$ 替换 $\omega_k$，并求解类似下面的轨迹级目标：

```math
\min_{x_K,\{\delta_k\}}\;
\mathcal L_y(x_0)
+\frac{\lambda_K}{2}\|x_K\|^2
+\frac12\sum_{k=1}^K\lambda_k\|\delta_k\|^2,
\qquad
x_{k-1}=\mu_\theta(x_k,k)+\sigma_k\delta_k.
```

二次控制代价限制轨迹偏离冻结生成过程的程度，而任务项则施加部署目标。MBF 继承这一控制空间视角，但用两时间 Flow Map 替换固定 DDIM 链，并通过 temporal recourse 扩展单调路径。

### 2.3 两时间 Flow Map

记 $\Phi_\theta(\cdot;r,t):\mathcal X_r\to\mathcal X_t$ 为学习得到的两时间 Flow Map，简写为 $\Phi_{t\leftarrow r}$。它近似任意生成时间 $r,t\in[0,1]$ 之间的传输。对于调度 $1=t_0>\cdots>t_K=0$，少步生成写为

```math
z_{t_{i+1}}=\Phi_{t_{i+1}\leftarrow t_i}(z_{t_i}).
```

与必须对每个时间区间数值积分的速度模型不同，学习得到的 Flow Map 可以直接进行长距离时间转移，并允许在训练后选择采样步数 [14]。其两时间接口也允许向更早生成时间映射，MBF 正是利用这一能力构造优化坐标，而不是执行新的随机加噪采样。

## 3. Multi-Back Flow

MBF 首先构造一个优化引导的单调 Flow Map 轨迹，然后插入受控后退—前进转移，并联合优化全部控制变量。该框架不依赖任何特定正演算子或应用领域。

### 3.1 受控单调 Flow Map

给定 $1=t_0>\cdots>t_K=0$ 和源样本 $\xi\sim p_{\mathrm{src}}$，我们可以通过 $z_{t_0}=\xi+B_{\mathrm{src}}a$ 对源进行可选修正，其中 $a$ 是源位移；令 $a=0$ 即固定源。普通控制 $u_i$ 在每个长距离转移前注入：

```math
z_{t_{i+1}}
=\Phi_{t_{i+1}\leftarrow t_i}(z_{t_i}+B_i u_i),
\qquad i=0,\ldots,K-1.
```

注入算子 $B_{\mathrm{src}}$ 和 $B_i$ 可以是恒等映射、低秩基，或与任务无关的结构化参数化。记 $c=(a,u_0,\ldots,u_{K-1})$，则终点为 $x_{\mathrm{mono}}(c)=\mathcal G_{\mathrm{mono}}(\xi;c)$。单调优化问题为

```math
\min_c\;
\mathcal L_y(x_{\mathrm{mono}}(c))
+\frac{\lambda_{\mathrm{src}}}{2}\|a\|^2
+\frac12\sum_{i=0}^{K-1}\lambda_i\|u_i\|^2.
```

当 $a=0$ 时去掉源控制项。约束形式则最小化控制能量，同时要求 $\mathcal L_y(x_{\mathrm{mono}})\le\varepsilon$，并满足其他 $g_j(x_{\mathrm{mono}})\le0$、$h_\ell(x_{\mathrm{mono}})=0$。

### 3.2 Few-step basin lock

记 $\Delta t_i=t_i-t_{i+1}$。由于 $\sum_i\Delta t_i=1$，抽屉原理给出 $\max_i\Delta t_i\ge1/K$：任何 $K$ 步调度至少包含一次较长的时间转移。少步轨迹还意味着更少的控制注入位置。

在当前终点 $x=x_{\mathrm{mono}}(c)$，记 $g=\nabla_x\mathcal L_y(x)$，以及 $J_{\mathrm{mono}}=D_c\mathcal G_{\mathrm{mono}}(\xi;c)$。对小更新 $\delta c$，Taylor 展开得到

```math
\mathcal L_y(\mathcal G_{\mathrm{mono}}(c+\delta c))
=\mathcal L_y(x)
+g^\top J_{\mathrm{mono}}\delta c
+O(\|\delta c\|^2).
```

在 $\|\delta c\|\le\epsilon$ 下，最大一阶下降量为

```math
\epsilon\|J_{\mathrm{mono}}^\top g\|.
```

**命题 1（局部 basin-lock 判据）。** 若 $J_{\mathrm{mono}}^\top g=0$，则任何充分小的单调控制更新都无法在一阶上降低任务损失。更一般地，较小的比值 $\|J_{\mathrm{mono}}^\top g\|/\|g\|$ 表明任务梯度无法被当前单调参数化所提供的终点方向充分表示。我们将这种局部限制称为 few-step basin lock。

### 3.3 Multi-Back temporal recourse

对于位于较低噪声时间 $\tau$ 的状态 $z_\tau$，选择一个更早、噪声更高的时间 $\rho$，满足 $0\le\tau<\rho\le1$。一个 Back 模块先映射到 $\rho$，加入控制 $b$，再返回 $\tau$：

```math
\mathcal B_{\tau,\rho}(z_\tau;b)
=\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(z_\tau)+Cb\right).
```

此后轨迹继续向数据端点演化。对于 $M$ 个模块，令 $\Gamma=\{(\tau_m,\rho_m)\}_{m=1}^M$，$b=(b_1,\ldots,b_M)$。插入这些事件后，终点写为 $x_{\mathrm{MBF}}(c,b;\Gamma)=\mathcal G_{\mathrm{MBF}}(\xi;c,b,\Gamma)$。这一操作不是随机 restart，也不是重新加噪：它始终沿同一个冻结两时间 Flow Map 演化，并把 Back controls 作为同一条轨迹中的优化变量。

记 $J_c=D_cx_{\mathrm{MBF}}$、$J_b=D_bx_{\mathrm{MBF}}$。联合局部终点空间为 $\mathrm{range}([J_c,J_b])$，它包含 $\mathrm{range}(J_c)$。在联合扰动预算 $\|[\delta c;\delta b]\|\le\epsilon$ 下，最大一阶下降量为

```math
\epsilon\sqrt{\|J_c^\top g\|^2+\|J_b^\top g\|^2}.
```

**命题 2（局部解锁）。** 若 $J_c^\top g=0$ 但 $J_b^\top g\ne0$，普通单调控制不提供一阶下降方向，而 Multi-Back 参数化可以提供。因此，在更早生成尺度加入控制，可以重新打开当前单调轨迹中缺失的任务相关终点方向。

### 3.4 联合优化

MBF 联合优化可选源位移、普通控制以及 Back controls：

```math
\min_{a,\{u_i\},\{b_m\}}\;
\mathcal L_y(x_{\mathrm{MBF}})
+\frac{\lambda_{\mathrm{src}}}{2}\|a\|^2
+\frac12\sum_i\lambda_i\|u_i\|^2
+\frac12\sum_m\gamma_m\|b_m\|^2.
```

约束形式在满足任务可行性的条件下最小化同样的控制能量。推理过程为：采样 $\xi$；将所有控制初始化为零；先优化单调轨迹；按照 $\Gamma$ 插入 Back events；重新联合打开并优化所有控制；最终返回 $\widehat x=x_{\mathrm{MBF}}(a^\star,u^\star,b^\star)$。一条包含 $K$ 个单调转移和 $M$ 个 Back 模块的轨迹，每次 replay 大约需要 $K+2M$ 次 Flow Map 调用。

**算法 1：Multi-Back Flow。** 从 $p_{\mathrm{src}}$ 采样 $\xi$，初始化 $a,u_i,b_m=0$；优化单调受控轨迹；根据 $\Gamma$ 插入 Back 模块；在 MBF 目标下联合优化 $a$、$\{u_i\}$ 和 $\{b_m\}$；返回最终终点。

## 4. FWI 案例研究

我们使用一个冻结的 optimal-transport FlowMap，作为 $70\times70$ 地质模型的无条件先验。可微声学正演模型把最终端点映射到接收器观测。经典 FWI 是一个非凸波动方程数据拟合问题 [15]；InversionNet 是具有代表性的任务训练 CNN 方法 [2]。实验选择 5 个测试样例（29864、29748、29544、29952、29694），并与 5 个预先指定的 raw seeds 20268032–20268432 交叉组合。算法、网格、action budget 和哈希值在剩余实验运行前记录到冻结 manifest 中。每个端点记录冻结后才读取真值，并仅用于事后 MSE 和 boundary F1 评估。

FWI 的采集设置、checkpoint、数据划分、暴露历史和指标定义见[归档附录](../docs/fwi_appendix.md)，从而保持正文方法表述与具体任务解耦。

### 4.1 与基线的对比：逐样例 oracle 选择的结果

![冻结基线与每个测试样例中最低 MSE 的结果对比。](../assets/figures/method_comparison.png)

*图 1。冻结的任务训练基线与本文方法的逐样例最低 MSE 结果。按行选择的 seed 后缀依次为 332、332、432、432 和 232。该图使用真值 MSE 进行事后逐行 oracle 选择，并不是推理时选择规则，也不构成计算量匹配的优越性结论。*

### 4.2 5 个样例 × 5 个随机种子

![5 个测试样例与 5 个预先指定随机种子的全部最终重建。](../assets/figures/multiseed_models.png)

*图 2。5 个测试样例（行）与 5 个预先指定随机种子（列）上的真值和最终重建。不同于图 1 的显式逐行 oracle 汇总，该图展示全部运行结果。*

最终端点的平均/最大观测 MSE 为 **0.00929/0.02305**，平均/最小 boundary F1 为 **0.8905/0.7526**。每个单元平均运行时间为 2,978 秒。任务变化的独立单位是地质样例（$n=5$），随机种子只是同一样例内的重复运行。因此，我们完整报告交叉面板和配对单元级描述性归因，不把 25 个单元视为彼此独立的任务，也不声称群体层面的统计显著性。

| 指标 | 内部 source stage | 最终端点 |
|---|---:|---:|
| 平均 MSE | 0.01424 | **0.00929** |
| 最大观测 MSE | 0.03219 | **0.02305** |
| MSE 改善次数 | — | 19/25 |
| Boundary F1 改善次数 | — | 22/25 |

*表 1。内部 source stage 与最终端点指标。source stage 并不是计算量匹配的独立基线。*

与内部 source stage 相比，最终端点的平均 MSE 降低 **0.00495**。其中 19/25 个单元的 MSE 改善，6/25 个单元变差；boundary F1 在 22/25 个单元中改善。这支持当前实现内部的组件归因，但不能单独识别 source control 与 Multi-Back 之间的因果交互。若要支持该结论，仍需进行计算量匹配的 source off/on × Multi-Back off/on 因子实验。

## 5. 局限性与结论

当前技术草稿只评估了一个合成采集族、5 个地质样例和一个冻结 Flow Map 先验。通用方法要求任务目标可微或至少可优化，并且能够访问两时间生成映射。若要形成更一般的正式结论，还需要计算量匹配的外部基线、预先规定的因子消融、更多独立样例，以及其他逆问题算子上的验证。

在当前范围内，MBF 将冻结生成模型推理重新表述为两时间 flow graph 上的非单调控制。Flow Map 减少生成转移次数，Multi-Back 则从更早生成尺度引入任务相关方向。当前 FWI 面板支持结果的可重复观察和内部组件归因，但方法的广泛适用性仍需进一步验证。

## 参考文献

1. O. Ronneberger, P. Fischer, and T. Brox. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” *MICCAI*, 2015.
2. Y. Wu and Y. Lin. “InversionNet: An Efficient and Accurate Data-Driven Full Waveform Inversion.” *IEEE TCI*, 2020.
3. Y. Blau and T. Michaeli. “The Perception-Distortion Tradeoff.” *CVPR*, 2018.
4. Z. Li et al. “Fourier Neural Operator for Parametric Partial Differential Equations.” *ICLR*, 2021.
5. P. Isola et al. “Image-to-Image Translation with Conditional Adversarial Networks.” *CVPR*, 2017.
6. Z. Zhang, Y. Wu, Z. Zhou, and Y. Lin. “VelocityGAN: Subsurface Velocity Image Estimation Using Conditional Adversarial Networks.” *WACV*, 2019.
7. H. Chung et al. “Improving Diffusion Models for Inverse Problems using Manifold Constraints.” *NeurIPS*, 2022.
8. H. Chung et al. “Diffusion Posterior Sampling for General Noisy Inverse Problems.” *ICLR*, 2023.
9. J. Song et al. “Pseudoinverse-Guided Diffusion Models for Inverse Problems.” *ICLR*, 2023.
10. Y. Zhu et al. “Denoising Diffusion Models for Plug-and-Play Image Restoration.” *CVPR Workshops*, 2023.
11. J. Song et al. “ReSample: Plug-and-Play Posterior Sampling via Hard Data Consistency.” *ICLR*, 2024.
12. B. Zhang et al. “Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing.” *CVPR*, 2025.
13. “Optimization-Guided Diffusion for Robot Control.” arXiv:2606.24208, 2026.
14. N. M. Boffi, M. S. Albergo, and E. Vanden-Eijnden. “Flow Map Matching.” 2024.
15. J. Virieux and S. Operto. “An Overview of Full-Waveform Inversion in Exploration Geophysics.” *Geophysics*, 2009.

---

代码、审计记录、指标定义与复现说明见[项目 README](../README.md)。