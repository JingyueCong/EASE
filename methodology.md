# CIRU：基于因果交互残差的 Retain-Set-Free LLM Unlearning

> **文档状态（2026-08-15）**：本文定义拟投稿方法与可检验假设。当前仓库已经实现
> EASE/Dual-ULD 和 F2R 双助手基线，但尚未完整实现本文的因果子空间、门控干预与相应
> 训练脚本。因此，文中的理论结论是待证明/验证的方法设计，不能表述成已经取得的实验
> 结果。`CIRU`（Causal Interaction Residual Unlearning）是暂定名，投稿前需再次检查重名。

## 1. 方法定位

本文考虑训练阶段无法访问真实 retain set 的语言模型遗忘。给定已完成训练的基础模型
\(f_{\theta_0}\) 和待删除集合

\[
\mathcal F=\{(x_i,y_i)\}_{i=1}^{n},
\]

目标是在不使用原始 retain 训练样本的条件下得到模型 \(f_{\theta_u}\)，使其：

1. 降低目标事实在原问题、改写问题和提取攻击下的可恢复性；
2. 尽可能保持非目标知识、通用语言能力和与 forget 样本共享的关系/风格结构；
3. 不把拒答、随机错误或乱码本身等同于“真正删除知识”。

这里的 **retain-set-free** 仅指训练和超参数选择阶段不读取真实 retain 数据。冻结方法与
超参数之后，标准 TOFU evaluator 仍会读取 retain、real-authors、world-facts 以及
retain-only reference logs。这些数据只用于最终评估。

### 1.1 与已有代码的关系

- **ULD**：训练一个小助手记忆 forget，并在推理时从基础模型 logits 中减去助手响应。
- **EASE/Dual-ULD**：使用真实 retain 邻域 \(R_{sub}\) 训练第二个助手，补回被 ULD 误伤
  的局部 retain 知识。
- **F2R**：用 matched counterfactual 替代真实 \(R_{sub}\)，但仍采用
  `base - A1 + A2` 的双助手 logit composition。
- **CIRU（本文主方法）**：counterfactual 不再只是第二助手的替代训练数据，而被组织成
  一个 \(2\times2\) 析因干预，用 difference-in-differences（DiD）估计目标事实的
  因果交互残差；随后直接在基础模型隐藏状态中移除该残差子空间。

因此，F2R 应作为 CIRU 的直接前驱和强消融，而不能将二者混为同一方法。CIRU 的核心
输出不是两个 assistant logits，而是因果残差子空间 \(U_\ell\)、门控器 \(g_\phi\) 和
干预强度 \(\alpha\)。

## 2. 为什么需要因果构造

Forget QA 同时携带多种信号：

- 应删除的实例特异事实；
- 应保留的关系类型、任务格式、语言风格与推理难度；
- 实体频率、答案长度、词汇和模板等 nuisance factors。

直接最小化 forget likelihood 或减去一个记忆 forget 的助手，无法区分这些因素。普通
paraphrase 只改变表达，随机 synthetic data 又同时改变事实和结构，都不能单独识别
“事实身份对相关回答的特异影响”。CIRU 通过同时操纵“目标事实身份”和“问题是否与
目标事实相关”来估计二者的交互效应。

## 3. 结构因果模型

对第 \(i\) 个 forget 实例定义：

- \(S_i\)：希望保持的结构属性，如 task、relation family、style、difficulty；
- \(T_i\in\{0,1\}\)：事实身份，\(T=1\) 表示原始待删事实，\(T=0\) 表示替换事实；
- \(R_i\in\{0,1\}\)：相关性，\(R=1\) 表示问题询问目标关系，\(R=0\) 表示 placebo
  关系；
- \(N_i\)：词汇、长度、模板等 nuisance variables；
- \(H_i^\ell\)：模型第 \(\ell\) 层表示；
- \(Y_i\)：模型输出或目标答案的 token 分布。

采用如下结构关系作为工作模型：

\[
H_i^\ell
=g_\ell(S_i,N_i)
+a_\ell T_i
+b_\ell R_i
+\tau_\ell(S_i)(T_iR_i)
+\epsilon_i^\ell,
\]

\[
Y_i=f_{>\ell}(H_i^\ell,S_i,N_i).
\]

其中 \(g_\ell\) 表示共享结构与表面形式，\(a_\ell\) 表示单独更换实体/事实带来的主
效应，\(b_\ell\) 表示询问相关关系的主效应，而 \(\tau_\ell\) 是 CIRU 希望识别并
抑制的“原始事实 × 相关问题”交互项。

这不是对自然语言世界的无条件因果声明。本文所谓 causal identification 始终相对于
明确生成的干预、匹配约束和下述识别假设。

## 4. 四单元反事实干预

对每个 \((x_i,y_i)\) 构造四个单元：

| 单元 | \(T\) | \(R\) | 内容 | 作用 |
|---|---:|---:|---|---|
| \(C_{11}\) | 1 | 1 | 原始实体、原始待删关系与答案 | treatment |
| \(C_{01}\) | 0 | 1 | 替换实体/事实，但保持关系、风格和难度 | matched counterfactual |
| \(C_{10}\) | 1 | 0 | 原始实体，但询问与待删事实无关的 placebo 关系 | factual placebo |
| \(C_{00}\) | 0 | 0 | 替换实体，并询问对应 placebo 关系 | counterfactual placebo |

下标顺序为 \(C_{TR}\)。四个单元必须尽可能共享：

- 指令模板和回答形式；
- relation family 与推理步数；
- 问题和答案长度区间；
- 语言、语气、风格和难度。

同时必须避免：

- 在 \(C_{01}\) 或 \(C_{00}\) 中复制原始答案、实体别名或关键 n-gram；
- placebo 问题仍可推导出待删事实；
- counterfactual 与真实 TOFU forget/retain probes 重合；
- 只靠改名字、但答案事实基本不变的伪 counterfactual。

每个 source 建议产生 \(V\) 个独立 views，并记录 generator、prompt hash、temperature、
seed、重试次数和过滤原因。

## 5. 表示与 DiD 因果残差

对单元 \(C_{tr}^{(i,v)}=(x_{tr},y_{tr})\) 进行 teacher forcing。为处理答案长度不一致，
定义第 \(\ell\) 层的回答表示为答案 token 隐状态的掩码平均：

\[
\phi_\ell(C_{tr}^{(i,v)})
=\frac{1}{|A_{tr}|}\sum_{j\in A_{tr}}H_{tr,j}^\ell.
\]

也应消融最后一个 prompt token、首个 answer token 和 attention-weighted pooling，确认结果
不是 pooling 选择造成的。

对每个 source/view 计算 DiD：

\[
\delta_{i,v}^{\ell}
=\big[\phi_\ell(C_{11})-\phi_\ell(C_{01})\big]
-\big[\phi_\ell(C_{10})-\phi_\ell(C_{00})\big].
\]

第一项同时包含事实身份差异与相关关系下的事实响应；第二项估计仅由实体/事实身份和
表面差异产生的主效应。二者相减后，理想情况下保留 \(T\times R\) 的交互残差。

### 5.1 识别假设

DiD 解释至少依赖：

1. **Consistency**：每个生成单元对应定义明确的干预版本；
2. **Positivity**：每个 source 都能构造四个有效单元；
3. **Matched exchangeability**：条件于 \(S\) 后，四个单元的剩余 nuisance 差异不与
   \(T\times R\) 系统相关；
4. **Parallel nuisance effects**：从原始到替换事实的非交互变化，在 relevant 和
   placebo 条件下近似相同；
5. **No leakage/interference**：control 不泄漏原答案，一个 source 的构造不依赖另一个
   source 的 treatment；
6. **Representation alignment**：四个单元的 pooled hidden states 表示可比较的生成阶段。

这些假设不能只靠文字声明。论文需要用 matching audit、placebo tests、不同 generator、
不同 matching 维度和人工标注报告其近似成立程度。

## 6. 从残差到因果子空间

对选定层 \(\ell\)，堆叠中心化后的交互残差：

\[
D_\ell=[\delta_{1,1}^{\ell},\ldots,\delta_{n,V}^{\ell}]^\top.
\]

最简单估计是截断 SVD：

\[
D_\ell=Q_\ell\Sigma_\ell V_\ell^\top,
\qquad U_\ell=V_\ell[:,1:k].
\]

为了避免把 control variation 也纳入删除方向，主实现建议采用 contrastive generalized
eigenspace。定义 signal covariance

\[
\Sigma_{\mathrm{sig}}^\ell
=\mathbb E[\delta^\ell(\delta^\ell)^\top]
\]

和由 matched/placebo 单元内部变化得到的 control covariance

\[
\Sigma_{\mathrm{ctrl}}^\ell
=\operatorname{Cov}\{\phi(C_{01}),\phi(C_{10}),\phi(C_{00})\}.
\]

求解

\[
U_\ell
=\arg\max_{U^\top U=I}
\frac{\operatorname{tr}(U^\top\Sigma_{\mathrm{sig}}^\ell U)}
{\operatorname{tr}[U^\top(\Sigma_{\mathrm{ctrl}}^\ell+\lambda I)U]}.
\]

这使子空间优先解释 DiD signal，而不是 controls 中的共享结构变化。\(k\)、层集合
\(\mathcal L\) 和正则项 \(\lambda\) 只能用 forget 与 synthetic controls 选择。

## 7. 目标门控器

无条件投影会损害所有包含相似 relation 的输入，因此训练轻量 gate
\(g_\phi(x)\in[0,1]\)。正例包括原始 forget 问题及其不改变事实的 paraphrases；负例
包括 \(C_{01},C_{10},C_{00}\)、mismatched controls 和纯模板 controls：

\[
\mathcal L_{\mathrm{gate}}
=-\mathbb E_{x^+}\log g_\phi(x^+)
-\mathbb E_{x^-}\log(1-g_\phi(x^-)).
\]

为避免 gate 只记住精确字符串，应加入 paraphrase consistency：

\[
\mathcal L_{\mathrm{cons}}
=\mathbb E_{x,x'\sim\operatorname{Para}(x)}
|g_\phi(x)-g_\phi(x')|.
\]

gate 的输入可使用冻结模型的 prompt embedding 或一个小型 encoder。主实验必须报告 gate
的 AUROC、对 paraphrase/jailbreak 的召回、对 controls 的 false-positive rate，以及额外
延迟。真实 retain 数据不得参与 gate 训练或阈值选择。

## 8. 隐藏状态干预

令 \(P_\ell=U_\ell U_\ell^\top\)，\(\mu_\ell\) 为 matched/placebo controls 的平均
表示。在层 \(\ell\in\mathcal L\) 对当前 token 状态执行：

\[
\widetilde H_t^\ell
=H_t^\ell
-\alpha_\ell g_\phi(x_{\le t})
P_\ell(H_t^\ell-\mu_\ell).
\]

其中：

- \(P_\ell\) 只移除估计的事实相关交互方向；
- \(g_\phi\) 控制干预是否作用于当前输入；
- \(\alpha_\ell\) 控制强度；
- 以 control center \(\mu_\ell\) 为中心，减少删除一般激活均值的风险。

可选的自适应强度为

\[
\alpha_{\ell,t}
=\alpha_\ell g_\phi(x_{\le t})
\sigma\big(\gamma(\|P_\ell(H_t^\ell-\mu_\ell)\|-b)\big),
\]

使模型仅在检测到足够强的目标残差时执行干预。第一版实现应先使用固定 \(\alpha_\ell\)
和 prompt-level gate，避免同时引入过多自由度。

与 EASE/F2R 的三次模型前向不同，CIRU 理论上只需基础模型一次前向、低秩投影和一个小
gate。但是否真正更快必须以 wall-clock、tokens/s 和峰值显存实测。

## 9. Retain-free 超参数选择

不能使用最终 retain utility 选择 \(k,\alpha,\lambda,\mathcal L\) 或 gate threshold。
定义仅依赖 forget 与生成 controls 的选择目标：

\[
\mathcal J
=\underbrace{\mathbb E_{C_{11}}
[\ell_{\theta_u}(y|x)-\ell_{\theta_0}(y|x)]}_{\text{target attenuation}}
-\eta\underbrace{\mathbb E_{C_{01},C_{10},C_{00}}
D_{KL}(p_{\theta_0}\|p_{\theta_u})}_{\text{control distortion}}
-\rho\,\mathrm{FPR}_{\mathrm{gate}}.
\]

在预先固定的网格上选择最大 \(\mathcal J\) 的配置。最终论文同时报告完整
forget–utility Pareto curve，避免只展示用测试 retain 指标反向挑出的单点。

### 9.1 固定的论文评估聚合

所有 TOFU 主表、扫参诊断与最终报告统一采用 LLM Beliefs 附录 E.2.1 的层级调和平均：

\[
\mathrm{Mem}=HM(1-ES,1-EM,1-P_{para},1-TR_f),
\]

\[
\mathrm{Util}=HM(MU,\mathrm{Fluency}),\qquad
\mathrm{Agg}=HM(\mathrm{Mem},\mathrm{Util}).
\]

其中 Fluency 是 forget generation 被分类为 `clean` 的概率；上游框架目前仍将该字段
命名为 `forget_Q_A_gibberish`。FQ、privacy leakage、forget/retain ROUGE 等指标继续完整
报告，但不进入 Agg。论文不得再使用其他自定义 Mem/Util/Agg 公式与 baseline 比较。

## 10. 完整算法

**输入**：冻结基础模型 \(f_{\theta_0}\)、forget set \(\mathcal F\)、views \(V\)、候选层
\(\mathcal L\)、子空间维度 \(k\)、干预强度 \(\alpha\)。

1. 对每个 forget source 生成并验证 \(V\) 组
   \(C_{11},C_{01},C_{10},C_{00}\)；
2. 在冻结模型上 teacher-force 四个单元并提取 \(\phi_\ell(C_{tr})\)；
3. 计算每层 DiD residual \(\delta_{i,v}^{\ell}\)；
4. 由 signal/control covariance 估计低秩子空间 \(U_\ell\)；
5. 用 forget/paraphrase positives 与 synthetic negatives 训练 gate \(g_\phi\)；
6. 只用 \(\mathcal J\) 选择 \(k,\alpha,\lambda,\mathcal L\) 和 gate threshold；
7. 冻结全部组件，在生成时应用 gated hidden-state intervention；
8. 最终才运行 EASE/Open-Unlearning 统一 evaluator。

### 10.1 伪代码

```text
for (x_i, y_i) in forget_set:
    quads_i = generate_and_validate_2x2_controls(x_i, y_i, V)
    for layer l in candidate_layers:
        h11, h01, h10, h00 = pooled_hidden_states(base, quads_i, l)
        delta[i, l] = (h11 - h01) - (h10 - h00)

for layer l in candidate_layers:
    Sigma_sig  = covariance(delta[:, l])
    Sigma_ctrl = covariance(control_variations[:, l])
    U[l] = top_generalized_eigenvectors(Sigma_sig, Sigma_ctrl, rank=k)

gate = train_gate(forget_and_paraphrases, synthetic_controls)
config = select_with_forget_and_controls_only(U, gate)

def intervene(hidden, prompt, layer):
    score = gate(prompt)
    return hidden - alpha[layer] * score * U[layer] @ U[layer].T @ (hidden - mu[layer])
```

## 11. 理论命题（待形式化证明）

### 命题 1：DiD 交互项识别

若第 3 节的加性交互 SCM 成立，且四单元在条件 \(S\) 下满足 parallel nuisance
effects，则

\[
\mathbb E[\delta_i^\ell\mid S_i]=\tau_\ell(S_i).
\]

**证明思路**：将四个 potential representations 代入 DiD。共享项 \(g(S,N)\)、事实
主效应 \(aT\) 和相关性主效应 \(bR\) 两两抵消，只剩 \(TR\) 的交互项与均值为零的
误差。实际生成中的不完全匹配形成识别偏差，需在定理中显式给出偏差项。

### 命题 2：非目标输出扰动上界

假设从第 \(\ell\) 层到 logits 的模型尾部是 \(L_\ell\)-Lipschitz。对非目标输入
\(x\)，若 \(g_\phi(x)\le\varepsilon\)，则

\[
\|z_u(x)-z_0(x)\|_2
\le L_\ell\alpha_\ell\varepsilon
\|P_\ell(H^\ell(x)-\mu_\ell)\|_2.
\]

这说明保留误差同时受 gate false positive、子空间与非目标表示的重合、以及干预强度
控制。它不是“零 utility loss”的保证。

### 命题 3：目标交互衰减

若目标表示可写为
\(H^\ell-\mu_\ell=U_\ell c+r\)，且 \(U_\ell^\top r=0\)，当
\(g_\phi(x)=1\) 时，干预后目标子空间分量变为

\[
U_\ell^\top(\widetilde H^\ell-\mu_\ell)=(1-\alpha_\ell)c.
\]

当 \(\alpha=1\) 时估计子空间内的分量被完全投影掉；子空间估计误差、遗漏维度和后续层
重构会导致残余记忆，因此必须用 extraction、relearning 和 paraphrase 实验验证。

## 12. 与 baselines 的公平比较

主表使用相同的 LLaMA-3.2-1B-Instruct full checkpoint、TOFU
forget01/forget05/forget10、Open-Unlearning evaluator 和冻结的 retain reference logs。
至少比较：

- Original 与 Retain/Retrain reference；
- GA、GradDiff、DPO、NPO、SimNPO；
- Unilogit+KL、FLAT、SOUL、Offset、LLM Beliefs；
- ULD；
- EASE/Dual-ULD（真实 \(R_{sub}\)，retain-aware oracle）；
- F2R dual assistant；
- CIRU（Ours）。

表中同时标注每个方法是否在训练时访问 retain data、是否使用外部 generator、参数量、
训练 GPU-hours 和推理开销。CIRU 与 retain-aware EASE 比较时不能隐去数据可见性差异；
与 FLAT/SHRED 等 retain-free 方法比较时必须匹配训练 token/step 和调参预算。

### 12.1 主表指标

沿用 `Table/llama3_1B.tex` 的七列：

\[
\mathrm{Mem}
=H(1-P_F,1-\mathrm{ROUGE}_F,\mathrm{TR}_F),
\]

\[
\mathrm{Util}
=H(P_R,\mathrm{ROUGE}_R,\mathrm{TR}_R),
\qquad
\mathrm{Agg}=H(\mathrm{Mem},\mathrm{Util}).
\]

同时报告官方 Forget Quality、Model Utility、Forget/Retain ROUGE。附录必须提供
real-authors、world-facts、PrivLeak、MIA、extraction strength、exact memorization、
gibberish 和逐 seed 结果。表内任何数字必须能追溯到原始 `TOFU_EVAL.json`、commit、
config 和 checkpoint；模拟数字只能标记为 mock，不得用于论文结论。

## 13. 必做消融与 falsification tests

### 13.1 因果构造消融

1. 不使用 \(C_{01}\)（无 matched counterfactual）；
2. random synthetic data 替代 \(C_{01}\)；
3. paraphrase 替代事实 counterfactual；
4. 去掉 \(C_{10}\) 或 \(C_{00}\)，退化成单差分；
5. 普通差分 \(H_{11}-H_{01}\) vs 完整 DiD；
6. 打乱四单元配对后重新估计 \(U\)（placebo randomization test）；
7. 只匹配 style、只匹配 relation、完整匹配；
8. 不同 generator 与规则/模板 generator；
9. \(1/2/4/8\) counterfactual views。

### 13.2 干预消融

1. SVD vs contrastive generalized eigenspace；
2. 无 gate、字符串 gate、表示 gate；
3. 单层 vs 多层干预；
4. rank \(k\)、\(\alpha\)、\(\lambda\) 敏感性；
5. 不使用 control center \(\mu\)；
6. hidden projection vs 原 F2R 双助手 logit subtraction；
7. 参数量和 FLOPs 匹配的单 assistant/双 assistant baseline。

### 13.3 因果假设审计

- 四单元的 relation/style/difficulty/length matching score；
- source answer、实体别名和 n-gram 后验重合；
- placebo 单元对 forget answer 的可预测性；
- DiD residual 在随机配对后是否消失；
- learned subspace 对 matched/placebo controls 的投影范数；
- generator identity 能否从 residual 中被轻易预测；
- control matching quality 与 utility/forgetting 的相关性。

如果 random pairing 与正确 pairing 得到同等结果，或者 placebo difference 不影响性能，
则“因果交互识别”叙事被证伪，应退回更保守的 contrastive representation method 表述。

## 14. 鲁棒性与遗忘边界

必须测试：

- forget question/answer paraphrases；
- 多轮追问、上下文注入和轻量 jailbreak；
- extraction 与至少两种 MIA；
- exact memorization 与 Min-K；
- 少量重新学习 steps 后的恢复曲线；
- 相邻 retain 与远端 retain 的分桶 utility；
- 连续删除请求和子空间合并/冲突；
- refusal、事实错误和 gibberish 的分解。

如果方法主要阻止目标提示触发，但权重中信息仍可通过攻击快速恢复，论文应称其为
**causally motivated behavioral unlearning intervention**，而不是声称参数中的知识已被
物理擦除。

## 15. 预期实现结构

建议在独立分支实现，避免与 F2R baseline 混淆：

```text
scripts/
  generate_ciru_quads.py          # 四单元生成、验证和审计
  extract_ciru_activations.py     # teacher-forced hidden-state 提取
  estimate_ciru_subspace.py       # DiD、covariance、SVD/generalized eigenspace
  train_ciru_gate.py              # retain-free gate
  run_ciru_tofu.sh                # 单 split smoke/full
  run_ciru_tofu_all.sh            # 三 split/多 GPU
open-unlearning/src/model/
  ciru.py                         # gated hidden-state intervention wrapper
open-unlearning/configs/model/
  Llama-3.2-1B-Instruct_CIRU.yaml
tests/
  test_ciru_quads.py
  test_ciru_did.py
  test_ciru_projection.py
  test_ciru_retain_free.py
```

第一阶段只实现单层、固定 rank、固定 \(\alpha\)、prompt-level gate 和 SVD；smoke
验证通过后再加入多层、generalized eigenspace 和自适应强度。

## 16. 可以与不可以主张的结论

在理论假设和实验均成立后，可以主张：

- 提出一个训练阶段 retain-set-free 的析因反事实 unlearning 框架；
- 在构造干预假设下，DiD 隔离事实身份与问题相关性的交互表示；
- gated low-rank intervention 改善 forgetting–utility trade-off；
- 匹配质量、placebo 和 randomization tests 支持所提出机制。

在没有额外证据时，不可以主张：

- CIRU 识别了自然语言世界中无条件成立的真实因果效应；
- 模型参数中的目标知识被完全删除；
- 方法完全不需要 retain data（最终 benchmark evaluation 仍需要）；
- retain-free、counterfactual unlearning 或 causal unlearning 的“首次”；
- 单 seed 或单个最优权重点构成稳定 SOTA。

## 17. 当前执行顺序

1. 完成 F2R 的三个 1B full split，作为 feasibility baseline；
2. 冻结主表、数据可见性和超参数选择规则；
3. 新建 CIRU 分支并实现四单元数据 schema 与验证器；
4. 实现 DiD/SVD 单层原型和无训练投影 smoke test；
5. 加入 gate，对比无 gate 与 F2R；
6. 只有 CIRU 在至少两个 split 上表现出稳定机制信号后，再扩展 generalized
   eigenspace、3B、MUSE 和多 seed；
7. 最终用 `scripts/build_tofu_main_row.py` 从三个完整 JSON 自动生成论文行。
