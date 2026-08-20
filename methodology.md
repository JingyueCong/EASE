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

## 19. U-F2D：面向短 QA 与长文档的统一分层表示

为避免 TOFU 与 MUSE 分别定义两套方法，U-F2D 将所有输入统一表示为

\[
\text{document}\rightarrow\text{segment/event}\rightarrow
\text{claim}\rightarrow\text{evidence span}.
\]

TOFU 短问答是该表示的退化情形。`paired-hierarchy-v2` 不再把两个独立表述之间的
词面差异直接视为事实差异，而是先将每个 cell 解析为结构化
`subject-relation-object-qualifier-polarity` 原子事实。语义标注器只能从冻结答案中复制
精确连续子串作为 claim/evidence，随后本地验证器将子串转换为字符 span。若标注器用多个
片段重新拼出接近全文的 claim，验证器会确定性地收缩为 declared subject 与已经验证的
object/evidence span；若 canonical object 是答案中的唯一精确子串，也会用它去除过宽
evidence 中的关系词与修辞骨架。随后仍拒绝无法收缩的过宽 evidence、包含 subject 的非身份
evidence、极性或可回答性不匹配，以及 matched pair 中事实数量/关系 schema 不一致的单元。
被拒绝的 source id 与原因写入
metadata；四个阶梯阶段必须使用同一个通过验证的子集。该过程不访问真实 retain 数据，
但会调用外部语义标注模型，因此论文中需单独报告 annotator model、提示词、接受率及人工
审计通过率。旧的确定性词面对齐保留为 `paired-hierarchy-v1` 复现基线，不再用于主实验。
若一个原子事实嵌入长描述句而不存在独立连续分句，v2 允许以 2--4 个非连续精确 span
共同表示其 subject、predicate 与 object；若这些 span 的并集仍接近全文，则进一步收缩为
subject 与 object/evidence。训练 mask 取最终 span 的并集，外围修辞仍由
base-model KL 保持。这一规则同样适用于 MUSE 的长文 segment。

对 claim token，A1 继续学习 `CE(C11)+Uniform(C01)`，A2 学习
`CE(C10)+Uniform(C00)`；evidence token 可获得额外权重。对 answer 中不属于目标
claim 的 token，加入冻结 base model 的前向 KL：

\[
\mathcal L=\mathcal L_{\mathrm{claim}}
+\gamma\mathcal L_{\mathrm{evidence}}
+\beta D_{\mathrm{KL}}(p_{\theta_0}\|p_{\theta}).
\]

严格 TOFU 方法阶梯固定同一语义验证子集、seed、72/72 optimizer steps、
LoRA 结构和由冻结 `a72_a72` 59-configuration sweep 得到的共同推理点
`(-1.8,1.8,0.0004)`，只比较：

1. `FullAnswer`；
2. `ClaimMask`；
3. `ClaimMask+KL`；
4. `Claim+Span+KL`。

因此阶梯内部差异可归因于监督粒度与局部保持约束，而不是样本筛选、训练步数或推理扫参。
该 v2 子集的 FullAnswer 结果必须重新运行，不能拿旧的 200-unit FullAnswer 作为阶梯内对照。
完整运行入口为 `scripts/run_uf2d_tofu_ladder.sh`。

固定点阶梯若显示 `Claim+Span+KL` 的 Util 明显提高但 Mem 不足，则先冻结助手并运行
`scripts/sweep_uf2d_claimspan_108.sh`。该 development search 使用

\[
w_1\in\{-1.8,-2.2,-2.6,-3.0,-3.4,-3.8\},\quad
w_2\in\{1.0,1.4,1.8,2.2,2.6,3.0\},
\]

以及 `top_filter in {0.0001,0.0004,0.001}`，共 108 个只推理配置。它不重新训练或
生成，但由于依据完整 forget05 retain-side 指标选择，必须标记
`selection_retain_access=true`。任何胜出点都只能作为后续新 seed/split 确认实验的冻结
配置，不能直接作为无偏最终测试结果。

在依据该搜索解释方法前，必须用 `scripts/audit_uf2d_hierarchy.py` 从十个 TOFU author
block 各抽两条，对 20 个完整四格单元做人工审计。审计逐 cell 核对：claim 是否包含完整
原子命题且排除无关描述、evidence 是否为最小事实变化、两组 matched pair 是否只改变
规定因素，以及抽取是否误选标点或风格 scaffold。需同时报告 `claim/evidence coverage`、
自动风险标记以及人工 `PASS/FAIL` 比例；若失败集中在单句多事实答案，必须先修正分句器并
重跑分层训练，不能用后续权重扫参掩盖数据表示错误。

## 20. TOFU author-profile v2：以作者为生成与干预单位

对旧 `full_authorblock_v1` 的系统审计表明，“每 20 条共享 replacement name”并不等于
共享一个 replacement author profile。旧生成器仍逐 QA 独立调用模型，而且把问题中的
地点或描述短语当成 `target_entity`；因此旧 JSONL、生成器、FullAnswer checkpoint 和
结果全部保留用于复现及 row-wise ablation，但不再作为严格 causal 主数据。

新管线 `generate_tofu_author_factorial.py` 使用冻结清单
`tofu_forget05_author_blocks.json` 显式指定十位 canonical author。对每个作者的 20 条
immutable C11，一次联合生成：

1. 唯一 replacement author；
2. 带唯一 `fact_id` 的 coherent twin-profile ledger；
3. 覆盖全部 20 个 source id 的 C01，并为每条 C01 声明 supporting fact ids；
4. 冻结上述结果后，再联合生成同一 block 的 placebo fact ledger 与 20 组 C10/C00，
   每组同样声明 supporting fact ids。

因此实验赋值由 block metadata 显式给出，而不是要求作者名必须出现在 question 中。该规则
可以正确表示 `Where was the author born?` 一类隐式 TOFU 问题；旧 schema 仍保留原来的
question-span 严格检查，不受新版本影响。最终记录继续输出兼容的
`C11/C01/C10/C00`，所以既有 `f2d_did_a1/f2d_did_a2` FullAnswer adapter 无需修改。

新 runner `run_f2d_author_twin200.sh` 默认只生成和审计。它以 author block 为断点恢复
单位，只有十个 block 全部通过 hard schema、完整 source coverage、canonical author、
单一 profile 和 fact-reference 检查时才写最终 JSONL；随后强制运行 random 与
risk-prioritized 两套审计，并在 `STOP_AFTER_AUDIT=true` 时停止。训练必须显式设置
`AUDIT_APPROVED=true`，方法名为 `F2D-AuthorTwin200-FullAnswer-v2`，不会覆盖任何旧
FullAnswer 模型或报告。

该设计修正的是 experimental-unit consistency 与 identity assignment，仍不能仅凭自动
校验声称 causal identification。论文主实验还必须报告人工审计的 polarity、response
mode、fact count、target-relation equivalence、placebo exclusion、跨 20 条 profile
一致性与 target/retain posterior overlap；未通过的 block 应整块重生成，不能只删除失败
row，否则会破坏预注册的作者级实验单位。

## 21. TOFU author-contract v3：先冻结可检验契约，再生成四格文本

V2 的 200 条自动审计结果表明，作者级 joint generation 解决了 replacement profile
不一致，却没有充分控制每条问答的 response format，也没有保证 placebo 真正排除目标事件。
尤其是同一个 `primary_research_topics` 被重复用于整个作者 block，以及“获奖”与“其他认可”
这类语义相邻关系，都会使 DiD 的 placebo contrast 携带目标信息。因此 V2 完整保留为生成
消融，但不作为主结果数据。

V3 将数据构造显式分成四个可审计阶段：

1. **Contract**：完全由 immutable C11 确定性提取 `response_mode`、`answer_format`、
   fact-count proxy、显式/隐式身份形式及问答长度；生成模型无权修改契约。
2. **Plan**：一次读取同一作者的 20 条 C11，规划一个 coherent replacement author，
   并为每条记录指定 target relation、replacement evidence 和正交 placebo relation。
   一个 block 至少包含 10 种 placebo relation，任何 relation 最多使用两次。
3. **Render**：每五条为一个可恢复 chunk，仅把已冻结 plan 渲染成 C01/C10/C00。
   本地 hard gate 要求 relation-pair 的问题结构、response mode、format、fact count、长度和
   planned evidence 均符合契约；失败只重生成该 chunk。
4. **Judge**：语义审计器逐条判断 target relation equivalence、target fact replacement、
   C10/C00 parallelism、placebo exclusion、author-domain matching、
   author-profile consistency 和 surface quality。生活琐事、联系方式和社交账号即使与目标
   正交，也不能作为可比的 causal placebo。七项必须全部为真才能形成 checkpoint；每个
   chunk 还绑定 frozen plan 的 SHA-256 指纹，plan 改变后旧 render 不可复用。

最终 estimator 和训练接口不变：A1 使用 `C11/C01`，A2 使用 `C10/C00`，推理仍组合两个
assistant residual。因此 V3 与既有 FullAnswer 结果的差异主要来自 factorial design 质量，
而不是更换模型架构。新路径、state、checkpoint 和报告全部使用
`tofu-author-contract-v3` / `F2D-AuthorContract200-FullAnswer-v3` 名称，不覆盖 V1/V2。

语义 judge 只是一道生成 gate，不能当成人工真值或 causal identification 的证明。训练前
仍须对 random 与 risk-prioritized audit sheet 逐格检查，并报告自动通过率、chunk 重试率、
人工通过率、placebo relation 多样性和拒绝原因。主实验应先固定通过人工审计的 V3 JSONL，
再冻结训练超参数和推理点；不能根据最终 retain 指标返回修改生成数据。

## 22. TOFU author-typed v4.2：模型规划事实，代码渲染文本

V3 的主要失败不是 API 不稳定，而是让同一个生成模型同时承担事实规划、placebo 选择和
全文改写；任何一步出错都会表现为新的 prompt 例外。V4.2 因此将生成自由度收缩到 typed
causal intermediate representation，并完整保留 V1/V2/V3 作为可复现实验：

1. **全量类型契约**：从全部 200 条 immutable C11 确定性识别八类回答格式
  （short prose、list、yes/no explanation、multi-sentence prose、yes/no、unavailable、
   date/year、numeric）以及 polarity、fact-count proxy 和长度，不再从少量样本推断规则。
2. **Atomic target edit**：LLM 只返回 C01 的 `target_relation` 及 `old -> new` 精确
   factual span 编辑；作者身份由代码在所有出现位置统一替换，supporting replacement fact
   也从实际 new span 推导。重复出现的同一 old fact span 会被一致替换。渲染器要求编辑互不
   重叠、year/number 类型守恒、单字段最多六个编辑、长文本覆盖率不超过 60%。除 identity
   与 unavailable relation 外，只替换作者名不能通过。
3. **Deterministic placebo**：C10/C00 不再由 LLM 生成。每个 author block 固定使用 20 个
   author-professional relation，覆盖 drafting、revision、editorial、translation、archive、
   citation、proof、rights、index 和 versioning workflow；每种 relation 恰好一次，且两侧
   value assignment 按 block/query/seed 平衡翻转。food、pet、social handle 等 domain-
   mismatched trivia 从构造空间中被彻底删除。
4. **Block-wide consistency and surface invariance**：同一个多词 factual anchor 在 20 条
   问答中只能映射到一个新值，渲染器会把该映射传播到每个精确出现位置。标点、句数、
   list connective、否定词和 uncertainty marker 属于冻结 scaffold，不能被 factual edit
   改写。作者全名、首名、姓氏和所有格由代码统一替换，replacement author 必须沿用 C11
   的主导代词类别，从而系统处理 V4.1 暴露的 alias、pronoun、list 和 polarity 漂移。
5. **Accept/reject judge**：语义模型只能检查 relation equivalence、事实确实变化、
   replacement profile 一致性与编辑后语法，不能返回改写文本。本地 schema、格式契约和
   placebo 设计先通过后才调用 judge。

V4.2 的统一性来自“typed causal IR + dataset renderer”，而不是强迫 TOFU 和 MUSE 共享同一种
句面模板。TOFU renderer 使用精确 span；MUSE 可在相同 IR 下把 edit 落到 document segment、
claim 和 evidence span。两者继续输出 `C11/C01/C10/C00`，所以 dual-assistant DiD estimator
和既有 `f2d_did_a1/f2d_did_a2` adapter 不变。

运行入口 `scripts/run_f2d_author_typed_v4.sh` 在任何 API 调用前执行真实 200 行离线
preflight；测试覆盖 C01 exact-edit renderer、C10/C00 professional renderer、八类 surface
contract、禁止全文重写、禁止 name-only factual edit、scaffold invariance、alias/possessive
替换和 block-wide fact propagation。生成后必须达到 200 rows、10 blocks、
20 unique placebos/block、所有语义 verdict 为真，并完成人工 random/risk audit，之后才允许
设置 `AUDIT_APPROVED=true` 训练。生成 JSONL 一旦冻结，不得依据 retain-side Agg 返回修改。

## 23. TOFU author-anchor V5.1：类型化锚点 ID 的统一 causal IR

V4.2 仍让生成模型返回原文 `old -> new` span，因此同一个事实的长短嵌套表达可能造成
overlap，模型也可能复制一个并不存在的 old span。V5.1 将 source localisation 完全移出
生成模型：代码从 immutable C11 确定性建立 non-overlapping occurrence anchors，并为完全
相同的 lexical fact 建立 block-wide group ID。每个 catalog 绑定 SHA-256 digest。

生成模型的动作空间仅包含：替代作者、固定代词类别、每行 relation label、显式
`target_group_ids`，以及 `group_id -> replacement_value`。它既不能提交 old span，也不能
生成 C01 全文。代码按冻结
offset 渲染每个 occurrence，再执行作者全名、首名、姓氏和所有格替换。year、number、date
保持类型与信息粒度；完整 textual、`MM/DD/YYYY` 或 ISO date 由代码解析、验证后按 C11
标点模板渲染，因而不会把等价日期格式误判为因果错误。response mode、answer format、
fact-count proxy、长度与 target leakage 继续经过 hard gate。未知 ID、重复 ID、过期
catalog 和重叠 anchor 在 API 边界直接不可表达。no-op assignment 被规范化为空操作，但每个
非 identity、非 unavailable 行仍必须由 `target_group_ids` 指向至少一个真实变化。token
replacement 必须保持词数与数量词一致；不使用容易把 `Beijing` 等专名误判为
`-ing` 动词或把 `faith -> Buddhism`、`LGBTQ -> queer` 误判为类型漂移的字符串表面
启发式，也不把 `Canadian-themed -> Australian` 的合法短语替换误判为连字符漂移。成对引号解析器只
抽取真正位于同一对引号内的书名，不再把两个书名之间的连接文本当作事实锚点。

每行额外冻结 `fact_change_required` 与 `intervention_policy`。普通 factual row 采用
`factual_anchor_change`；identity row 采用 `identity_binding`；unavailable row 采用
`identity_binding_with_unavailability_preserved`。后两类不伪造不存在的 factual object，
其 `target_fact_changed` 由确定性 policy 判定；原始 judge 输出与 override 字段同时写入
artifact，避免隐藏修正。`profile_consistent` 不允许 override；judge 仍独立审核 relation
match、真实 factual row 的 fact change、跨行 profile consistency 与 natural surface。

V5.1 沿用同一个 2x2 estimand：C11/C01 训练 A1，C10/C00 训练 A2，推理组合 dual-assistant
residual。改变的只是 causal intervention 的定位接口，而非训练或评估协议。TOFU 使用 lexical
occurrence anchors；后续 MUSE 可使用 document/claim/evidence occurrence anchors，但共享相同
的 group assignment、deterministic renderer、catalog digest 和四格输出，因此无需为长文本
重新定义方法。V5.1 使用独立 JSONL、state、audit、checkpoint 和 report 路径，FullAnswer 与
V1--V5 全部保留作为构造消融。

## 24. TOFU author-rowlocal V5.2：作者级 profile 与逐行局部映射

V5.1 的失败审计显示，主要不稳定性来自一个模型调用同时承担 replacement profile、20 条
relation label、全局 anchor replacement 和 20 组 opaque `target_group_ids`。单条输出错误会
使整个 block 重做；提高重试数或继续添加字符串例外既不能解决该组合复杂度，也会降低方法
可解释性。V5.2 因此保持 V5.1 的 anchor catalog、deterministic renderer、professional
placebo 和 dual-assistant DiD estimator，只改变规划的条件分解方式：

1. **Profile stage**：每个 author block 只生成一次 replacement identity、冻结代词类别与
   简短 profile theme，不返回 row ID、anchor ID 或改写文本。
2. **Row-local mapping**：20 条 C11 分别只暴露自身 local anchor catalog。每条调用只返回
   relation、最多两个 answer-side target groups 及其 typed replacement value。每行具有独立
   checkpoint 与 retry budget；一个 source 失败不会撤销其他 source。
3. **Deterministic reconciliation**：相同 group 的多个 proposal 按固定 source 顺序选择
   canonical value，并完整记录冲突 ledger。每行只渲染自己声明的 target groups，避免其他行
   的无关替换触发大面积 surface rewrite。
4. **Block judge and targeted repair**：全部 deterministic gates 通过后，judge 一次检查完整
   replacement author block。拒绝结果必须定位到 source id；下一轮只重新映射失败行，同时
   冻结已接受行涉及的 shared replacements。达到 judge round 上限则 fail hard。

`fact_change_required` 不再由模型可操纵的 relation label 决定，而由 immutable C11 question、
answer 与 response contract 确定。普通 factual row 必须选择至少一个 **answer-only** anchor；
同时出现在 question 与 answer 的 group 被视为 relation scaffold，禁止作为事实干预目标。
描述式 full-name 与 unavailable row 使用显式 identity policy 和空 target groups。离线 preflight
在调用 API 前检查全部 200 条 C11，保证每个 factual row 至少存在一个 answer-only 候选。
生成 artifact 记录 profile、row mapping 和 judge 的实际 attempt/round，便于报告失败率与生成成本。

V5.2 先预注册 block 1 和 4 的 40-unit smoke，因为它们在 V5.1 中耗尽完整 retry budget。只有
这两个困难 block 的 hard gate 与人工 random/risk audit 同时通过，才运行 `block_ids=all` 的
独立 200-unit 数据。smoke 子集禁止训练；完整训练仍需 `AUDIT_APPROVED=true`。V5.2 使用新的
JSONL、state、audit、checkpoint 和 report 路径，不覆盖 FullAnswer 或 V1--V5.1，因而可以把
规划分解本身作为严格构造消融。

## 25. TOFU author-ledger V5.3：冻结事实账本与 ledger-conditioned mapping

V5.2 的困难 block smoke 进一步揭示了一个与 opaque anchor 无关的全局问题：即使 40/40 的
row mapping 均通过确定性验证，20 个独立 mapper 仍可能分别发明互相冲突的出生地、国籍、
成长经历、奖项与作品。此时 block judge 会把 `profile_consistent=false` 归到全部 20 行；而
“仅从 accepted rows 构造事实上下文”的 repair 在全拒绝时得到空上下文，因此无法稳定收敛。

V5.3 将生成过程重新分解为：

1. **Frozen typed fact ledger**：每个 author block 先一次性生成 replacement identity 与覆盖
   20 个 source id 的事实账本。每项包含 `fact_key`、`target_relation`、
   `replacement_fact`、`fact_change_required` 和 `intervention_policy`。同一 `fact_key` 必须具有
   完全相同的 replacement fact。
2. **Pre-mapping semantic gate**：独立 judge 在任何 surface mapping 前审核 ledger 的国籍、
   地点、时间线、作品、奖项、数量与主题是否能同时属于一个作者。只有通过的 ledger 才写入
   checkpoint，并冻结其 digest。
3. **Ledger-conditioned row mapping**：逐行 mapper 不再发明事实，只能逐字复制自己的
   `ledger_fact_key`、`ledger_replacement_fact` 与 relation，再选择本行 answer-only anchors
   表达该事实。代码要求 replacement 至少覆盖 ledger fact 的一个内容 token。
4. **Minimal row-fidelity judge**：最终 judge 的 `profile_consistent` 定义为“该行 C01 是否忠实于
   自己的已批准 ledger entry”，不得因另一行错误而拒绝本行。Repair 只重生成最小失败行，
   ledger 始终冻结不变。

该改动改变了 profile 语义、artifact schema 和 acceptance protocol，因此使用独立 V5.3
JSONL/state/audit 路径，绝不覆盖 V5.2。V5.3 仍保持 C11 immutable、C01 target intervention、
C10/C00 professional placebo 与 `(C11-C01)-(C10-C00)` dual-assistant DiD estimand。只有困难
block smoke 的 deterministic hard gate 与人工审计通过后，才能生成完整 200-unit 数据；只有
完整数据再次审计并显式设置 `AUDIT_APPROVED=true` 后才允许训练。

## 26. TOFU author ledger-slots V5.4：单一局部编辑接口

V5.3 的 block 1/4 smoke 中两个 frozen ledger 均通过，但 13 个 row mapper 耗尽重试。其中主要
失败并非事实不一致，而是接口要求模型同时输出 `target_group_ids` 与
`anchor_replacements` 两份等价集合，并让模型自行满足 quote、word-count、date、answer-format
和 polarity 等表面约束。这属于 planner coordination 与 deterministic surface 的职责混淆。

V5.4 保持 V5.3 已冻结的 replacement author ledger、C11/C01 target intervention、C10/C00
professional placebo 和 dual-assistant DiD estimand，仅替换 row mapping 与 surface projection：

1. 每行只暴露本行 answer-only anchors，并按 occurrence 顺序映射成局部 `A00...` slot；prompt
   不再暴露 block-global group ID。
2. mapper 只返回一个 `edits=[{slot_key,replacement_value}]` 列表。代码由该列表唯一推导内部
   `target_group_ids` 与 replacement map，因此不存在两个集合不一致的问题。
3. code-owned canonicalizer 按 anchor kind 处理完整日期、年份、数字、quoted title、proper phrase
   和单 token，并确定性移除外层引号、额外句号、非法 polarity/markup。所有 canonical value
   仍必须包含 frozen ledger fact 的 content token，并再次通过原始 response contract。
4. identity/unavailable policy 由 immutable source contract 与 ledger 决定，不调用 row mapper；
   internal renderer 使用显式 policy relation，避免把 descriptive full-name row 误判为 factual row。
5. replacement map 完全 row-local。相同 lexical token 在不同 source 中可以表达不同事实，不再
   做 block-global winner reconciliation；同一事实的一致性由 frozen `fact_key`/ledger digest
   保证。
6. V5.4 可以把已通过语义审核的 V5.3 ledger 及仍满足新契约的 row checkpoint 迁移到独立 state
   path。迁移过程重新执行 V5.4 deterministic validation，无法通过的行才重新调用 API。

由于 mapping schema、surface projection 与 reconciliation 语义发生改变，V5.4 使用新的
JSONL/state/audit 路径，绝不覆盖 FullAnswer 或 V1--V5.3。困难 block smoke 必须先达到
40/40 rows、2/2 blocks、无 deterministic errors，并完成人工 random/risk audit；在此之前不得
扩展到完整 200-unit 训练。

## 27. TOFU author ledger-answer V5.5：冻结事实与完整答案渲染解耦

V5.4 的完整 block 1/4 smoke 证明，局部 slot 接口虽然消除了 opaque ID coordination，却仍有
不可消除的表示边界。书名与奖项等 frozen ledger facts 可能不存在类型兼容的局部 span；主题、
社会观点、国际影响等复合关系即使能替换若干 token，也可能留下语法残片或未完整表达 ledger
语义。这不是提高 retry 数可以解决的 surface noise，而是 atomic span renderer 与描述性问答
之间的结构不匹配。

V5.5 保持已通过语义审核的 author-level frozen ledger、C11 immutable、C10/C00 professional
placebo 和 dual-assistant DiD estimand，只替换 factual C01 的 row renderer：

1. C01 question 由代码对 immutable C11 question 做 replacement-author identity binding，模型
   不得改写 relation。
2. 每个 factual row 独立接收自己的 ledger entry、replacement profile、C11 style reference 与
   response contract，并只生成一个完整 `replacement_answer`。
3. 代码要求 frozen fields 逐字一致、目标作者无泄漏、语义 polarity/availability 一致，且对明确
   询问日期或数量的问题保留相同 object type；旧 classifier 的 format 标签、fact count、句数、
   标点与长度差异只作为 audit warning。answer 不得仅替换身份，并必须显式包含 ledger fact 的
   content evidence。该策略记录为 `semantic-compatible-v5.5.1`。
4. identity/unavailable rows 仍完全确定性渲染，不调用模型；C10/C00 继续来自冻结的 professional
   placebo library。
5. block judge 独立检查 relation match、fact change、ledger consistency 与 natural surface；
   repair 只重生成 judge 拒绝的完整 answer，冻结 profile、ledger 和已接受行。

这一路径与早期 FullAnswer 的区别是：replacement identity 与 20-row fact ledger 在任何 surface
生成之前已经联合冻结并通过 block-level coherence judge；完整答案只负责实现单条 ledger fact，
不能重新发明作者事实。由于 renderer 语义改变，V5.5 使用新的 JSONL/state/audit 路径，只迁移
digest 一致的 V5.3/V5.4 ledger profile，不迁移旧 span row。FullAnswer 与 V1--V5.4 均保留为
严格消融。困难 block smoke、deterministic hard gate 与人工审计通过前，不允许训练 assistant。

## 28. TOFU author direct-contrast V5.6：直接语义改写与独立复核

V5.5 的人工审计显示，仅冻结自由文本 `replacement_fact` 仍不足以证明 causal contrast：模型可
保留 C11 的书名或奖项，仅增加年份、作品说明或更长的描述，而原有 code/judge 会把整段文本
差异误认为 factual object 已变化。V5.6 不再增加按样本或按 relation 手写的 slot taxonomy，
而是让生成模型为每行直接输出：

1. `source_core_fact`：C11 真正回答的最短核心事实；
2. `replacement_core_fact`：同一 relation 下真正不同的新核心事实；
3. `contrast_status`：`changed`、显式 `policy` 或 `abstain`。

独立 judge 同时读取 immutable C11、两个 core fact 和完整 20-row replacement profile，逐行检查
source fidelity、same relation、core fact changed 与 replacement plausibility。相同奖项/书名加
年份、保留完整原集合只增加一项、同义改写或仍蕴含原核心命题均必须拒绝。只有 profile judge
通过后，V5.5 的 row-local complete-answer renderer 才生成 C01，随后第二个 block judge 再从
C11/C01 与 before/after core facts 复核 factual change 和自然度。

无法可靠识别核心命题时必须 `ABSTAIN`；该状态保留在 attempt artifact 中并阻止 final JSONL，
不能为了覆盖率猜测答案。V5.6 使用新的 design、state、JSONL 与 audit 路径，不迁移 V5.5 ledger，
FullAnswer 与 V1--V5.5 均保持不变。40-row smoke 人工审核通过前不得扩展或训练 dual assistants。

## 29. TOFU author joint-contrast V5.7：问题与答案联合干预

V5.6 的 40-row 人工审计进一步暴露了 answer-only renderer 的系统缺陷：即使 C01 answer 已表达
通过独立审核的 replacement core fact，C01 question 仍可能保留 C11 的书名、出生地、日期、
奖项对象或身份描述，导致“问题问原事实、答案答替代事实”。这不是 surface warning，而是
factorial cell 的联合语义不成立，因此 V5.7 分配新的 design、schema、state、JSONL 与 audit 路径，
不修改也不迁移 V5.6 accepted row。

V5.7 冻结 V5.6 的直接 core-fact ledger 思路，但每条 row-local generation 同时输出完整
`c01_question` 与 `replacement_answer`。生成器必须保持 relation schema、论元结构、基数与
response-mode family，同时联合替换 question 中所有 target-specific premise。每条记录保存
`question_anchor_rewrites` 与 rationale 作为可审计 provenance；这些字段不能替代语义验证。

独立 block judge 新增两项不可覆盖的判据：

1. `question_premise_profile_consistent`：C01 问题中的书名、地点、日期、数量和身份描述均与
   replacement profile 相容；
2. `answer_satisfies_question`：C01 answer 必须直接回答实际的 C01 question，而不只是孤立地
   符合某条 replacement fact。

对于显式 identity/unavailable policy row，只允许覆盖 `target_fact_changed`；以上两项以及
relation match、profile consistency 和 natural surface 仍必须通过。先运行 blocks 1/4 的 40-row
smoke，完成 deterministic hard gate 与人工风险/随机审计；未获人工批准前不得启动 dual assistant。

## 30. TOFU author semantic-contrast V5.8：最小生成接口与代码所有的 provenance

V5.7 的困难 block smoke 表明，完整 C01 question/answer 可以逐行生成，但要求模型同时返回
`source_id`、ledger key、逐字 `question_anchor_rewrites` 与 rationale，会把语义生成错误和字符串
补丁协议错误混在一起。语义正确的候选可能仅因模型报告的 `source_text` 不是 C11 的精确子串而被
拒绝；这种 rejection 不提供新的 causal evidence，也不适合迁移到 MUSE 长文本。

V5.8 保持 V5.7 的 estimand 与必要不变量不变：C11 byte-frozen、V5.6-style replacement core-fact
ledger、C01 same-relation joint question/answer intervention、冻结 professional C10/C00 placebo、
`(C11-C01)-(C10-C00)`、逐行 targeted repair，以及六字段独立 block judge。改变的只是生成器接口：

1. row generator 只返回完整 `c01_question` 与 `replacement_answer` 两个内容字段；
2. `source_id`、relation、fact key、replacement fact、design 和 digest 全部由代码注入，模型返回的
   同名额外字段被忽略；
3. question rewrite provenance 使用确定性的 character `SequenceMatcher` 在生成完成后计算，记录
   source/replacement spans 与 operation，并明确标注为 `audit_only`，不再作为 acceptance gate；
4. deterministic gate 仍负责 schema、目标作者泄漏、response-mode family、replacement ledger
   evidence 与 identity-only no-op；same relation、question premise consistency、Q/A entailment、fact
   change、block profile coherence 和 natural surface 仍由分批独立 semantic judge 审核；
5. preflight 输出写入独立日志，主运行日志不再混入测试 fixture 的 `block_ready` 文本。

由于 generator contract、provenance schema 和 acceptance boundary 均发生变化，V5.8 使用新的
design、state、JSONL、profile 与 audit 路径，不迁移 V5.7 row/block checkpoint。FullAnswer 与
V1--V5.7 代码及 artifact 均保留。仍先运行 block 1/4 的 40-row smoke；hard gate 和人工 random/risk
audit 通过之前，不允许扩展为 dual-assistant 训练。

## 31. TOFU author semantic Agent V5.9：显式上下文与独立逐行 critic

V5.8 修复了由生成问题中的标题触发的部分表面误判，但单次 Q/A 生成加确定性语义分类仍无法复现
具有完整实验上下文的 Agent 判断。V5.9 因此改变 acceptance boundary，并分配新的 design、schema、
state、JSONL、profile、trace 与 audit 路径；V5.8 及此前所有 FullAnswer/V1--V5.7 artifact 保持不变。

每条 row-local intervention 按以下状态机执行：

1. 代码构造 content-addressed context packet，包含 immutable C11、row before/after core fact、完整
   20-row replacement ledger、replacement identity/profile、causal estimand 与 hard boundaries；
2. semantic planner 只解释 relation、argument、answer object、cardinality、response mode、source/
   replacement premises 和歧义，不生成 C01 文本；
3. generator 读取同一 context packet 与缓存的 semantic brief，仅输出完整 C01 question/answer；
4. 独立 critic 调用逐项判断 same relation、question premise、answerability、replacement fact、source
   fact removal、response mode 与 natural surface；任一项失败时，把 evidence 和单条 repair instruction
   反馈给 generator，只重新生成该 row；
5. 所有 row critic 通过后，仍运行 V5.8 的独立六字段 block judge，再经过 deterministic audit 与人工
   random/risk 审计。

确定性代码只硬性负责 JSON/schema、完整覆盖、唯一 source ID、C11 byte freeze、target leakage、
checkpoint/digest 和 provenance。基于关键词推断“问题类型”、ledger 的同义表达、relation equivalence
与自然度仅作为 critic context，不能独立拒绝候选。因此书名 `When Bridges Sleep` 中的 `When` 不再被
当成 temporal interrogative。V5.9 使用现有 OpenAI-compatible Azure Chat API 上的应用层 Agent 循环，
不依赖服务端必须支持 Responses function calling；所有 planner/generator/critic 输入输出仍完整记录，
便于复现实验和迁移到 MUSE 长文本。40-row smoke、hard gate 与人工审核通过前不得生成完整训练结论。
