# Multi-Back Flow

**用于逆问题的优化引导 Flow Map**

**Haoyang Jiang — William & Mary**

[English README](README.md) · [完整英文论文](paper/main.md) · [中文论文](paper/main_zh.md)

Multi-Back Flow（MBF）是一种推理时方法，用于将一个**冻结的无条件 Flow Map**适配到新的逆问题。生成先验始终保持冻结，测量信息仅通过优化目标进入推理过程。

其核心思想是：将严格单调的少步生成轨迹，改造成一条可控的非单调轨迹。当一次跨度较大的 Flow Map 转移把重建带入较差的局部盆地时，MBF 会返回到更早的生成时间，在那里施加由任务驱动的控制，然后再次向前生成。

## 核心思想

标准的受控 Flow Map 轨迹为

```math
z_{t_{i+1}}=\Phi_{t_{i+1}\leftarrow t_i}(z_{t_i}+B_i u_i),
\qquad 1=t_0>\cdots>t_K=0.
```

当轨迹只包含少量转移时，可注入控制的位置也很少。因此，任务梯度可能无法被当前单调轨迹所能产生的终点方向充分表示。

MBF 在轨迹中加入一个后退—前进控制模块：

```math
\mathcal{B}_{\tau,\rho}(z_\tau;b)
=\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(z_\tau)+Cb\right),
\qquad \tau<\rho.
```

当前公开实现采用锚定形式：

```math
\mathcal{C}_{\tau,\rho}(x;b)
=x+\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(x)+b\right)
-\Phi_{\tau\leftarrow\rho}
\left(\Phi_{\rho\leftarrow\tau}(x)\right).
```

因此，$\mathcal{C}_{\tau,\rho}(x;0)=x$。即使数值上的后退—前进回路并不完全可逆，零 Back control 也不会改变输入状态。

源控制、单调轨迹控制和 Back controls 被联合优化：

```math
\min_{c,b}\;
\mathcal{L}_y\bigl(x_{\mathrm{MBF}}(c,b)\bigr)
+\frac{1}{2}\lVert c\rVert_{\Lambda_c}^{2}
+\frac{1}{2}\lVert b\rVert_{\Lambda_b}^{2}.
```

## 与现有方法的关系

| 方法 | 推理时机制 | 轨迹结构 |
|---|---|---|
| DPS | 在反向扩散过程中加入似然梯度 | 单调扩散链 |
| OGD | 沿冻结的 DDIM 轨迹优化控制变量 | 单调受控链 |
| MBF | 联合优化控制变量，并重新打开更早的生成时间 | **非单调的两时间 Flow Map** |

因此，MBF 并不是把 DPS 更新简单移植到 Flow Map 上。它是一种带有显式时间回溯机制的轨迹优化方法。

## FWI 结果

当前实验使用一个冻结的 Flow Map 先验解决合成全波形反演问题，包括 5 个地质模型，每个模型使用 5 个预先指定的随机种子。

### 基线对比

![冻结基线与选定的 MBF 重建结果](assets/figures/method_comparison.png)

该图将冻结的任务训练基线与每个测试样例中事后选择的最低 MSE 的 MBF 随机种子进行比较。它用于展示重建质量，并不是推理时的随机种子选择规则。

### 完整的 5 个样例 × 5 个随机种子结果

![5 个样例与 5 个随机种子上的全部 MBF 重建结果](assets/figures/multiseed_models.png)

该图展示全部 25 次预先指定的运行，是观察不同随机种子行为的主要结果图。

机器可读的结果位于 [`results/final_25_summary.json`](results/final_25_summary.json)。FWI 的采集设置、评估方法和审计细节见 [`docs/fwi_appendix.md`](docs/fwi_appendix.md)。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest -q
python analyze_final_panel.py
```

## 仓库结构

```text
src/flowmap_multiback/   Multi-Back 核心实现与评估工具
tests/                   代数性质与实验协议测试
assets/figures/          主要结果图
results/                 冻结的汇总结果记录
docs/                    理论与 FWI 细节
paper/                   Markdown 与 LaTeX 论文源文件
```

## 公开范围

本仓库是一个精简的方法、论文与审计版本。完整复现 FWI 求解器还需要研究代码中使用的预训练 Flow Map checkpoint、基准数据张量以及可微声学正演后端。

## 论文与文档

- [英文 Markdown 论文](paper/main.md)
- [中文 Markdown 论文](paper/main_zh.md)
- [LaTeX 论文](paper/main.tex)
- [理论说明](docs/theory.md)
- [FWI 附录](docs/fwi_appendix.md)

## 引用

```bibtex
@software{jiang2026multiback,
  author  = {Haoyang Jiang},
  title   = {Multi-Back Flow: Optimization-Guided Flow Maps for Inverse Problems},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/HaoyangJiang-WM/FWI}
}
```

## 许可证

Copyright © 2026 Haoyang Jiang。本软件根据 [MIT License](LICENSE) 发布。