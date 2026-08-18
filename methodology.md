# CIRU：基于因果交互残差的 Retain-Set-Free LLM Unlearning

> **文档状态（2026-08-17）**：仓库已加入 CIRU-40 第一版可运行原型：固定 40 个
> source 的四格联合生成、hard audit、teacher-forced DiD、raw-effect truncated SVD、
> retain-free energy gate、hidden-state projection 和完整 Open-Unlearning 评估。它尚未
> 产生经复现实验验证的论文结论；generalized eigenspace、语义 judge、paraphrase gate
> 与多 seed 仍是待实现增强。`CIRU`（Causal Interaction Residual Unlearning）是暂定名，
> 投稿前需再次检查重名。

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

对选定层 \(\ell\)，堆叠交互残差：

\[
D_\ell=[\delta_{1,1}^{\ell},\ldots,\delta_{n,V}^{\ell}]^\top.
\]

第一版实现对 raw unit-level effects 做截断 SVD：

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
对 raw effects 而不是先中心化的 effects 做 SVD，是为了保留跨 40 个单元一致的平均
处理方向；先中心化会在交互效应完全一致时把最强信号直接消掉。contrastive generalized
eigenspace 是第二阶段增强，不属于首个 CIRU-40 结果。

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

gate 的输入可使用冻结模型的 prompt embedding 或一个小型 encoder。当前 CIRU-40 原型
先使用低秩投影 energy 的单变量逻辑 gate：C11 为正类，C01/C10/C00 为负类，并在每个
单元的 pooled answer representation 上拟合，再在推理时逐 token 计算门值。它是
retain-free 的最小可运行版本，但存在 pooled-to-token calibration gap，且尚未包含
paraphrase consistency。
主实验必须报告 gate
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
\mathrm{Mem}=HM(1-ES,1-EM,1-P_{para},1-TR_{knowledge}),
\]

\[
\mathrm{Util}=HM(MU,\mathrm{Fluency}),\qquad
\mathrm{Agg}=HM(\mathrm{Mem},\mathrm{Util}).
\]

其中 Fluency 是 forget generation 被分类为 `clean` 的概率；上游框架目前仍将该字段
命名为 `forget_Q_A_gibberish`。FQ、privacy leakage、forget/retain ROUGE 等指标继续完整
报告，但不进入 Agg。论文不得再使用其他自定义 Mem/Util/Agg 公式与 baseline 比较。
这里的 (TR_{knowledge}) 必须使用 OpenUnlearning 新定义
(p(y_{para}\mid x)/(p(y_{para}\mid x)+p(y_{pert}\mid x)))，而不是原始 TOFU
中对 wrong/correct ratio 使用 `closer_to_1_better` 后得到的 `forget_truth_ratio`。
后者继续用于 FQ 的分布比较，两种 TR 在报告和代码中不得混用。

训练超参选择必须记录 A1/A2 各自的层数、LoRA rank/alpha、学习率、epoch 与
uniform regularization 权重。默认 rank sweep 保持 `LoRA alpha = 2r`，以免把容量变化
与 LoRA scaling 变化混为一谈。每个训练配置使用独立模型目录与机器可读签名；禁止在
参数改变后复用旧 checkpoint。A1/A2 非对称超参属于主要方法自由度：A1 可更强地建模
forget-specific signal，A2 应更保守地建模 matched-control shared structure。

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

沿用 `Table/llama3_1B.tex` 的七列：`Agg. / Mem. / F.Q. / F.R-L /
Util. / M.U. / R.R-L`。其中前三个派生分数严格采用第 9.1 节固定的 LLM Beliefs
Appendix E.2.1 定义：

\[
\mathrm{Mem}
=HM(1-ES,1-EM,1-P_{para},1-TR_{knowledge}),
\]

\[
\mathrm{Util}
=HM(MU,\mathrm{Fluency}),
\qquad
\mathrm{Agg}=HM(\mathrm{Mem},\mathrm{Util}).
\]

F.R-L 与 R.R-L 作为表格诊断列完整保留，但不进入该 Agg。官方 Forget Quality、
Model Utility 及 Forget/Retain ROUGE 同时报告。附录必须提供
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

### 13.1 数据预算匹配与 causal core-set

`forget05` 的 Llama-3 EASE 训练读取 200 条真实 retain，其中 80 条属于
\(R_{sub}\)，其余 120 条作为 \(R_{far}\) 约束；早期 Llama-2 扫描采用
\(|R_{sub}|=40\)。默认 F2R 的两视图设置则包含 400 条 matched counterfactual。
因此必须报告总记录数为 40/80/200/400 的 nested random-budget 对照，固定其他训练和
推理参数，并同时报告 raw sequence/token budget。

Random-budget 只回答“更多 synthetic supervision 是否带来收益”，不是主方法。CIRU-40
主实验在任何生成之前固定 source：forget05 的十个有序 author block 各用固定 seed 抽取
4 条，共 40 条；随后对每条 source **联合生成**完整
\(C_{11},C_{01},C_{10},C_{00}\)。任何一个 source 构造失败则整批不落盘，不从已成功样本
中补选“容易生成”的 40 条。因此主结果不是从现有 F2R 400 条里随机抽取，也不是根据
最终残差或 retain 指标挑选。

从更大四格候选池中选择 causal core-set 仅作为额外数据选择消融。该消融不能仅按
\(\|\delta_{i,v}\|\) 取最大的 40 个，因为极端残差可能来自匹配失败或离群样本；若进行，
应使用带 hard audit 和覆盖约束的 D-optimal 准则：

\[
S_K^*=\arg\max_{|S|=K}
\log\det\!\left(\epsilon I+\sum_{(i,v)\in S}
\widetilde\delta_{i,v}\widetilde\delta_{i,v}^{\top}\right)
-\lambda\sum_{(i,v)\in S}q_{i,v}^{\mathrm{nuis}}
-\mu\,\mathrm{Redundancy}(S).
\]

其中 \(q^{\mathrm{nuis}}\) 只使用 relation/style/difficulty/length matching、答案泄漏、
placebo validity 与 control overlap 审计；不得使用 retain utility、最终 Agg 或测试集
retain 指标。选择还应限制每个 source 的 views 数并覆盖 relation family。论文至少比较
`direct-factorial-40`、`F2R-random-40`，并在有候选池预算时再比较
`residual-norm-top-40` 与 `causal-D-optimal-40`，才能把收益归因于 factorial causal
design 而不是样本数量或后验筛选。

### 13.2 干预消融

1. SVD vs contrastive generalized eigenspace；
2. 无 gate、字符串 gate、表示 gate；
3. 单层 vs 多层干预；
4. rank \(k\)、\(\alpha\)、\(\lambda\) 敏感性；
5. 不使用 control center \(\mu\)；
6. hidden projection vs 原 F2R 双助手 logit subtraction；
7. 参数量和 FLOPs 匹配的单 assistant/双 assistant baseline。

### 13.2.1 F2D：因果四格数据接入双 assistant

为单独检验收益究竟来自四格数据构造还是 CIRU hidden-state DiD 干预，加入
**Factorial-to-Dual（F2D）** 过渡消融。对每个通过审计的 CIRU 单元，固定映射为

\[
C_{11}\rightarrow D_f,\qquad
C_{01}\rightarrow D_{+},\qquad
\{C_{10},C_{00}\}\rightarrow D_{0}.
\]

其中 \(D_f\) 仍由 datamodule 加载完整 forget split；40 个单元只提供 40 条
\(D_+\) 伪 retain 和 80 条 \(D_0\) placebo/uniform control。A1 的 CE 数据为
\(D_f\cup D_+\)，A2 的 CE 数据为 \(D_+\)；两个 assistant 均在 \(D_0\) 上施加
uniform 约束，A2 还在 \(D_f\) 上施加 uniform 约束。推理继续使用 F2R 的双 assistant
logit correction。

因此，F2D 可以声称使用了 jointly generated factorial causal **construction**，但不能声称
其 estimator 识别了 DiD causal interaction；它没有显式计算
\((C_{11}-C_{01})-(C_{10}-C_{00})\)。这一阶梯应报告为
`F2R-random-40 -> F2D-C01-only -> F2D-C01+placebo -> CIRU-H`，用于区分预算、
matched counterfactual、placebo control 与 hidden causal estimator 的贡献。

新增的 **F2D-DiD-Balanced** 保留 dual assistant，但显式改变两个助手的任务：

\[
A_1:\ \mathrm{CE}(C_{11})+\lambda_1\mathrm{Uniform}(C_{01}),\qquad
A_2:\ \mathrm{CE}(C_{10})+\lambda_2\mathrm{Uniform}(C_{00}).
\]

每个助手的两类数据均为 40:40，因此不再受旧 F2D 的 40:280 不平衡影响。若推理组合为
\(w_1<0,w_2>0\)，其残差方向近似
\(-[C_{11}-C_{01}]+[C_{10}-C_{00}]\)，即负的 DiD interaction。这里的“近似”很重要：
两个 LoRA assistant 是分别优化的非线性模型，不能把它写成严格等于四次基础模型 logit
前向的代数估计量。首轮固定 architecture、learning rate 和推理点，只做
`epochs={12,18}`、`uniform weight={1,2}` 的 2x2 扫描；若优于旧 F2D，再冻结最佳
checkpoint 做单独推理权重扫描。

40→80 budget 消融必须采用 nested source design：80-unit 集合原样复用已审计的 40 个
完整四格单元，并在十个 TOFU author block 中各由 4 条扩展到 8 条。主比较固定 A1/A2 各
36 optimizer steps、uniform weight=1 与同一推理点，避免把样本身份变化或训练计算增加
误写成 causal budget 收益；48/60 steps 只作为单独的 compute-scaling 辅助条件。

全覆盖设置进一步令 (K=200)，即 forget05 的每条 QA 均形成一个四格单元，不再进行
source sampling。考虑到 TOFU 每 20 条对应同一作者，同一 block 的 20 个单元共享一个
replacement entity，但分别生成 relation-matched 与 placebo-relation controls。因此该
设置包含 200 个 causal units、800 个 cells，同时避免 200 个独立替代身份造成的
entity-direction variance。主比较仍固定 A1/A2 各 36 optimizer steps。

在训练 budget 对比结束后，冻结最优的 48-step assistant pair，先对
\(w_1\in\{-1.2,-1.3,-1.4,-1.5,-1.6\}\)、
\(w_2\in\{0.4,0.6,0.8,1.0\}\) 与
\(\tau\in\{10^{-4},2\times10^{-4}\}\) 做 coarse inference search，再围绕最优点
分别取三个邻域值进行 27-cell fine search。该步骤不更新任何模型参数。

若 60-step assistant 的最优点落在 \((-1.7,1.5)\) 的搜索边界，则沿
\((-1.7,1.5)\rightarrow(-2.1,1.9)\) 的对角线继续扩展，而不再做完整 Cartesian
搜索。该 9 个权重对与 3 个 filter 共形成 27 个 evaluation-only 配置，用于检验增强
删除残差的同时增强补偿残差能否继续外推 Pareto frontier。

若最优边界点的 Mem 分解显示 \(1-\mathrm{ES}\) 与
\(1-\mathrm{ParaProb}\) 已接近 1，而 \(1-\mathrm{EM}\) 和
\(1-\mathrm{KnowledgeTR}\) 仍偏低，则固定 causal data 与开发集 operating point，扫描
A1/A2 optimizer steps \(\{72/60,72/72,84/60,84/72\}\)。该非对称设计用于区分增强
target-answer residual（A1）与增强 utility compensation（A2）的作用。

F2D-40 的 equal-epoch 设置每个 assistant 在 5 epochs 下只有约 50 个 optimizer
steps，而 400-pair F2R 在同样 epochs 下约有 155 steps。因此同时报告：

1. **equal-epoch / compute-light**：5 epochs，保留 F2D 的计算优势；
2. **equal-step**：15 epochs，使更新步数约为 150，与 F2R 基本匹配。

此外，F2D A2 的 CE/uniform 记录数为 \(40:280\)，sampler 只交错而不重采样短侧，
`remember+uniform` 又直接以 `retain_weight` 放大 uniform loss。因此 equal-step 实验固定
A1 uniform weight 为 5，并比较 A2 uniform weight \(5/2/1/0.5\)。这一消融只根据训练
数据结构预先设定，不使用 retain 指标选择范围；最终配置选择仍须标注 diagnostic
`selection_retain_access=true`。

实际 equal-step 结果显示 15 epochs 虽可提高 Mem，却使 Util 严重下降。因此在不查看
额外网格结果的前提下预注册一个中间区间：`epochs={7,9}` 与
`A2 uniform weight={5,2}` 的 2x2 实验；A1 uniform weight 固定为 5，学习率、层数、
LoRA rank 与推理组合均保持五轮 F2D 设置不变。它用于识别从“遗忘不足”到“utility
collapse”的转折点，而不是把大量训练超参数同时纳入后验搜索。

由于七轮训练目录同时保存成对的 `checkpoint-50/60/70`，可以冻结训练轨迹并显式选择
A1/A2 的 `checkpoint-60` 来检验约六轮的中间状态，无需重新训练。checkpoint 必须成对
选择，且随后进行的权重/过滤阈值搜索仍属于 diagnostic retain-side selection；它不能
被表述成不访问 retain 信号的训练准则。

CIRU strict-v2 的 no-gate alpha 扫描显示 α=1.5 能显著改善 Mem，但三层同时干预时
MU 下降。下一步结构定位固定 α=1.5，比较单层 8/12/15（rank 8）与三层低秩
8/12/15（rank 4）。该扫描的配置范围由干预局部性假设预先确定，全部使用相同 40 个
四格单元；每个结构重新估计独立 subspace，不能复用不同 layers/rank 的 artifact。

进一步的标量细扫在 α=1.45 达到 `Agg=0.476082`、`Mem=0.418066`、
`Util=0.552793`，比 α=1.5 提升约 0.0047，但仍低于当前 F2R 的 0.486978。这说明
单一强度主要沿 Mem--Util 前沿移动，继续细化一个标量的预期收益有限。因此实现
层特异干预

\[
h_{\ell+1}'=h_{\ell+1}-\alpha_\ell P_\ell(h_{\ell+1}-\mu_\ell),
\]

并固定原始 strict-v2 rank-8 artifact，只扫描四个预先指定的
`(alpha_8, alpha_12, alpha_15)` 组合。首轮保持第 15 层为 1.5，逐步减弱第 8/12 层，
以检验“深层承担 forget-specific removal、浅层过强干预损害 utility”的机制假设。
这一扫描仍使用最终 retain 指标作诊断选择，不能据此声称 selection retain-free。

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

## 15. 当前第一版实现结构

建议在独立分支实现，避免与 F2R baseline 混淆：

```text
ULD/scripts/
  generate_ciru40.py              # 固定 source 后联合生成四格
ULD/uld/data/
  ciru.py                         # schema、hard validation 与 audit
scripts/
  train_ciru_subspace.py          # teacher forcing、DiD/SVD 与 energy gate
  run_ciru40_tofu.sh              # 生成、估计、完整评估一键执行
open-unlearning/src/model/
  ciru.py                         # gated hidden-state intervention wrapper
open-unlearning/configs/model/
  Llama-3.2-1B-Instruct_CIRU.yaml
tests/
  test_ciru_data.py
  test_ciru_generator.py
  test_ciru_subspace.py
```

第一阶段实现多层、固定 rank/\(\alpha\)、token-level energy gate 和 raw-effect SVD；
完整评估通过后再加入 generalized eigenspace、paraphrase-aware gate 和自适应强度。

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
3. 运行并审计已实现的 CIRU-40 四单元生成；
4. 运行 DiD/SVD 多层原型与完整评估；
5. 对比无 gate、energy gate 与 F2R；
6. 只有 CIRU 在至少两个 split 上表现出稳定机制信号后，再扩展 generalized
   eigenspace、3B、MUSE 和多 seed；
7. 最终用 `scripts/build_tofu_main_row.py` 从三个完整 JSON 自动生成论文行。

## 18. 已实现的 F2R-AG 过渡实验

为验证“共享残差抵消”和“输入相关干预”是否能先移动 F2R 的 Pareto 前沿，代码中加入
一个仍工作在 logit 空间的过渡方法 F2R-AG。它不是第 4--8 节定义的完整 CIRU，也不应
作为 DiD 因果识别结果汇报。

### 18.1 Residual alignment

在 matched counterfactual (C^+) 的答案 token 上，对 top-filter 支持集内的 A1/A2
logits 分别中心化为 \(\bar z_1,\bar z_2\)。对每个词表维度学习带岭约束的对角尺度：

\[
s_v=\frac{\sum_{C^+}\bar z_{2,v}
          \left(-\frac{w_1}{w_2}\bar z_{1,v}\right)+\lambda}
         {\sum_{C^+}\bar z_{2,v}^2+\lambda}.
\]

低观测 token 回退到 \(s_v=1\)，并对尺度裁剪以避免稀疏维度放大。推理时

\[
z'_2=z_2+(s-1)\odot\bar z_2,
\qquad \Delta_{\mathrm{align}}=w_1z_1+w_2z'_2.
\]

该目标直接使 (C^+) 上的加权共享残差趋近于零，但因 (s) 是词表维度校准而不是新的
全局 (w_2)，它不等价于继续做标量权重扫参。

### 18.2 Learned gate

门控器使用六个逐 token 残差统计量：组合残差 RMS/最大值、A1/A2 RMS、A1--A2
cosine 和两者 RMS 比值。用 forget 答案作为正类，用 (C^+) 与 (C^-) 作为负类，
训练带 L2 正则的逻辑门控器：

\[
g_\phi(x_t)=\sigma(\phi^\top \operatorname{standardize}(f_t)+b),
\qquad z_{t}^{\mathrm{final}}=z_t^{\mathrm{base}}+g_\phi(x_t)\Delta_t.
\]

训练 alignment 和 gate 都不访问真实 retain 样本或 retain 指标。完整阶梯固定同一组
A1/A2 checkpoint、(w_1,w_2) 与 top-filter，顺序评估
`F2R -> +Alignment -> +Gate -> +Alignment+Gate`。最终 LLM-Beliefs/OpenUnlearning
指标比较仍会查看冻结 retain reference，因此该开发实验明确标为
`selection_retain_access=true`。
