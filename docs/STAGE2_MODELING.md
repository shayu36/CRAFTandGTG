# HCFM 建模说明

## 1. 符号与层次图

对城市 (c)，宏观 Region 图为

\[
G_c^R=(V_c^R,E_c^R),\quad X_c^R\in\mathbb{R}^{N_c\times45},
\]

其中 45 维保持 CRAFT Population/POI/Road 顺序。微观有向 Road 图为

\[
G_c^r=(V_c^r,E_c^r),\quad X_c^r\in\mathbb{R}^{M_c\times d_r}.
\]

本实现的 CRAFT 适配器取 (d_r=15)：road type、length、oneway、lanes、maxspeed、direction 六项，加 GTG Road 级九项空间句法/分区特征。非单行道路展开为正反两个有向节点。

Region--Road 结构矩阵是稀疏矩阵 (P\in\mathbb{R}^{N_c\times M_c})：

\[
P_{ij}=\frac{\operatorname{length}(r_j\cap R_i)}
{\sum_k\operatorname{length}(r_k\cap R_i)}.
\]

道路跨 Region 时保留多个非零软归属；没有道路的 Region 保持空行，不补权重。

动态边界算子 (B^{in},B^{out}\in\mathbb{R}_{\ge0}^{N_c\times M_c}) 与 (P) 分离。实现按 Road geometry 上的投影顺序提取完整 Region 序列；每个相邻 Region 转移都写一组 out/in，因而 A→B→C 不会漏掉 B。若同一 Road 重入同一 Region，稀疏 coalesce 后系数可为 2，表示两次真实边界事件。内部 Road 不计边界流。对 passage count (Q\in\mathbb{R}^{B\times M_c\times1\times T})：

\[
S(Q)=\operatorname{concat}(B^{in}Q,B^{out}Q)
\in\mathbb{R}^{B\times N_c\times2\times T}.
\]

## 2. 编码、GTG 对抗与双向层次交互

CRAFT 宏观编码保持逐城市标准化与原参数结构：

\[
H^R=\operatorname{MacroEncoder}(X^R,E^R).
\]

RoadEncoder 采用 GTG TopoAggregator 的多头 GATv2 残差语义。注意力为

\[
\alpha_{ij}\propto\exp\{a^T\operatorname{LeakyReLU}(W_s h_i+W_t h_j+W_e e_{ij})\}.
\]

拓扑表示经两个独立编码器：

\[
Z^{sem}=f_{sem}(H^r),\qquad Z^{dom}=f_{dom}(H^r).
\]

`CostPredictor` 只用源城市真实 duration/speed cost；语义域分类器前放 GRL，使 (Z^{sem}) 难以识别城市；域分类器直接读 (Z^{dom})。正交损失为逐 Road cosine square，RankLoss 保持真实 cost 局部次序。目标城市只有静态 Road 图和 city label 进入领域判别，没有目标动态 cost 标签。

micro-to-macro 门控残差：

\[
\bar Z^R=PZ^{sem},\quad
g=\sigma(\operatorname{MLP}([H^R,W_r\bar Z^R])),
\]

\[
\tilde H^R=\operatorname{LN}(H^R+g\odot W_r\bar Z^R).
\]

macro-to-micro 反向条件：

\[
\tilde Z^r=\operatorname{LN}(Z^{sem}+P^TW_m\tilde H^R).
\]

层数与是否双向由 `hierarchy` 配置控制。融合后的 Region 表示继续进入原 CRAFT GraphTransformer/GFA；TFA 是源城市流量/表征自相似对齐，CCA 保留 POT/EMD Wasserstein。GTG 域对抗不替代 GFA。

## 3. Region RAG

RAG 仍是不可学习的 Region 级检索：

\[
R_i=\operatorname{Retriever}(\tilde H_i^R,month,weekday,start\_hour).
\]

数据库只包含源城市 train。整城快照中每个 Region 独立查询，得到 (R\in\mathbb{R}^{B\times N_c\times2\times T})，再用原 CRAFT 同类的 Reference Transformer 编码。本阶段没有 Road RAG、学习检索、重排或新 RAG 损失。

## 4. 联合 Flow Matching

宏观真实端点 (X_1^R\) 和微观真实端点 (X_1^r\) 分别由源 train 归一化器变换。共享 (t\sim U(0,1))：

\[
X_t=(1-t)X_0+tX_1,\qquad U_t=X_1-X_0.
\]

independent prior 独立采样 (X_0^R,X_0^r\sim N(0,I))，只允许状态一致性。coupled prior 先采 (X_0^r\)，再令

\[
X_0^R=S(X_0^r),
\]

因而允许速度一致性；配置启动时会拒绝 independent + 非零 velocity weight。

宏观条件包括 GFA aligned Region、encoded reference、hour/weekday/month、pooled Road 表示；微观条件包括 fused Road、广播宏观上下文和时间。两个速度场为时间卷积残差网络，并在中间层分别执行 Region/Road 图消息传递：

\[
V^R_\theta=V^R(X_t^R,t,C^R,G^R,S(X_t^r)),
\]

\[
V^r_\phi=V^r(X_t^r,t,C^r,G^r,P^TX_t^R).
\]

输出分别为 `[B,N,2,T]` 与 `[B,M,1,T]`，不依赖 beta/noise schedule，也不预测 DDPM noise。

## 5. 损失

Flow Matching：

\[
L_{FM}^R=\operatorname{MSE}_{mask}(V^R,U^R),\qquad
L_{FM}^r=\operatorname{MSE}_{mask}(V^r,U^r).
\]

状态终点估计：

\[
\hat X_1=X_t+(1-t)V(X_t,t).
\]

宏/微分别反归一化后，在物理 passage-count 单位比较：

\[
L_{state}=\operatorname{Huber}_{region-scale}
(\operatorname{denorm}(\hat X_1^R),
\operatorname{Calibrate}_{src-train}(S(\operatorname{denorm}(\hat X_1^r)))).
\]

校准是每个 in/out 通道的非负无截距比例，只允许源 train 拟合。coupled prior 可选：

\[
L_{velocity}=\operatorname{MSE}_{mask}(V^R,S(V^r)).
\]

微观拓扑不是“相邻流量相同”，而是保持图差分：

\[
L_{topo}=\|D_G\hat X_1^r-D_GX_1^r\|_1.
\]

总损失按日志分成 `L_macro/L_micro/L_cross_scale`，同时记录 FM、state、velocity、topology、cost、rank、semantic domain、domain、orthogonal、GFA(TFA/CCA) 全部子项。

## 6. 推理流程

Euler 或 Heun 从 (t=0) 联合积分至 1，默认 16 步。每次 NFE 的宏/微分支共享同一 (t)。入口记录 NFE、wall latency、samples/s、GPU peak memory、输出 min/max、非有限检查和参数量。

```text
CRAFT 45 / Region graph       Road attrs + GTG syntax / directed Road graph
          |                                   |
     MacroEncoder                         RoadEncoder
          |                         Semantic / Domain + GRL
          +--------- P gated interaction --------+
                            |
                   CRAFT GraphTransformer/GFA
                            |
               source-train Region RAG reference
                            |
        Macro vector field <----> Micro vector field
                            |
              Region in/out + Road passage count
                            |
               macro + micro + cross-scale loss
```
