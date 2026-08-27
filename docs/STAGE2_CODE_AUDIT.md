# 第二阶段代码审计

审计时间：2026-08-19（UTC）。项目根目录为 `/root/autodl-tmp/projects/Paper`，实际目录名是 `Paper/`。初始 Git 状态为 `main` 分支、commit `5dd1a61b4d623b72c4ae6368a8dcd186484f06d3`，`git status --short --branch` 仅输出 `## main...origin/main`，无用户未提交修改。

## 第一阶段真实调用链

```text
CRAFT/cleared_data/{city}/road.csv
  -> src/gtg_features/dual_graph.py:build_dual_graph
  -> src/gtg_features/space_syntax.py:compute_space_syntax
  -> src/gtg_features/partition.py:metis_partition
  -> src/gtg_features/road_to_region.py:map_roads_to_regions
  -> cache/gtg/{city}_gtg_region.npz:region_feat [N,9]

CRAFT/cleared_data/{city}/grid_region_feature.csv [N,45]
  -> src/craft_integrated/data_loaders.py:load_region_feature
  -> [CRAFT 45 | GTG 9] [N,54]
  -> src/craft_integrated/rep_model.py:GTAggregator
       CRAFT: FeatureInitLayer 45->128
       GTG: GTGTopoBranch 9->128 (GATv2 residual)
       fusion_proj 256->128
       GraphTransformer/GFA [N,128]
  -> ckpt/.../{city}_rep.npy
  -> src/craft_integrated/data_loaders.py:get_source_train_datasets/get_test_dataloader
  -> src/craft_integrated/retrieve.py:Retriever (Region 级、源城市 train 库)
  -> src/craft_integrated/craft.py:ReferTransformer + 条件拼接
  -> src/craft_integrated/diffusion.py:GaussianDiffusion1D
  -> src/craft_integrated/unet.py:Unet1D
  -> train.py / generate.py / evaluate.py
```

## 文件级审计

| 能力 | 真实位置 | 输入/输出与结论 |
|---|---|---|
| CRAFT 45 维 | `src/craft_integrated/data_loaders.py:load_region_feature` | `node_feature [N,45]`；5 个人口/总体道路字段 + 12 POI 数 + 12 POI score + 8 道路数 + 8 道路长度。|
| GTG 9 维缓存 | `src/gtg_features/pipeline.py`、`cache/gtg/` | `region_feat [N,9]`；静态空间句法/分区聚合，不是道路动态标签。|
| GTAggregator | `src/craft_integrated/rep_model.py` | baseline `[N,45]->[N,128]`；fusion `[N,54]->[N,128]`。|
| Graph Transformer / GFA | `rep_model.GTAggregator.gnn` | dense adjacency `[1,N,N]`；TFA 对应 `self_sim_loss`，CCA 对应 `wasserstein_loss`。|
| Retriever / RAG | `src/craft_integrated/retrieve.py`、`data_loaders.py` | Region 表征 + month/weekday/start_hour；测试库由源城市 train 构造并排除目标城市。|
| Reference encoder | `src/craft_integrated/craft.py:ReferTransformer` | `[B,2,T]->[B,refer_dim=256]`。|
| Diffusion | `src/craft_integrated/diffusion.py:GaussianDiffusion1D` | 训练/采样均保留；内部噪声估计器为 `Unet1D`。|
| U-Net | `src/craft_integrated/unet.py:Unet1D` | `[B,2,T]` 输入和同形输出，离散 DDPM timestep。|
| Dataset | `src/craft_integrated/data_loaders.py:FlowDataset` | 单 Region 样本，`x/reference [B,2,24]`；必须保留用于 stage1 回归。|
| checkpoint | `src/craft_integrated/train.py`、`generate.py` | `rep_model.pth`；`craft.pth={ori_model,ema_model}`；生成加载 `ema_model`。|
| 评价 | `src/craft_integrated/evaluate.py` | in/out 分开计算 CPC、Min-Max MAE、Min-Max RMSE。|
| 配置/入口 | `configs/{baseline,fusion}.yaml`、`src/craft_integrated/main.py` | YAML；CLI 依次预训练 GFA、训练 Diffusion、生成。|
| 现有测试 | `tests/test_*.py` | 第一阶段模型结构/数值、数据维度、缓存、归一化、对偶图。|

## 第二阶段兼容边界

- 不修改或删除 `FlowDataset`、`CRAFTModel`、`GaussianDiffusion1D`、`Unet1D` 和原评价函数。
- `stage1_diffusion` 继续路由原始第一阶段入口；FM 使用独立模块，Diffusion 参数不会误载入 FM。
- 完整层次模式必须新建整城快照 Dataset：`macro_flow [B,N,2,T]`、`micro_flow [B,M,1,T]`，不能改变第一阶段 `[B,2,T]` 的语义。
- 第一阶段 9 维缓存只能用于 stage1 条件或静态审计；完整模式必须保留 Road 节点并从 Road 图编码。
- CRAFT GFA 与 Region 级 RAG 均保留；GTG 道路领域对抗是独立模块，不能代替 GFA。

## 审计风险

1. CRAFT 的 `grid_region_feature.csv` 通过 `fillna(0)` 处理静态特征，这是第一阶段既有行为；第二阶段严格适配器不得新增静默补零。
2. 第一阶段 `load_region_graph` 对没有 train 流量的 Region 用 48 个零构造 GFA 的 `value`。第二阶段用 `region_mask` 明确排除缺失监督，不把缺行解释为真实零。
3. 原 GFA 使用 dense `N*N` adjacency；四城 Region 数仅 60--95，可保留。Road 图和层次矩阵必须稀疏。
4. 原始 GTG `GradReverse` 固定系数 -1；第二阶段需提供可配置系数并测试符号。

