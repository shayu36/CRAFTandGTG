# 第二阶段实现说明

## 模块清单

| 文件 | 职责 |
|---|---|
| `src/hcfm/data.py` | 整城 Dataset/单城市 batch、严格 shape/id/mask/finite 校验、source-only normalizer |
| `src/hcfm/preprocessing.py` | 45 维 Region、逻辑有向 Road、Road 15 维、Road 图、P/B、cache/manifest、微观宽表校验 |
| `src/hcfm/hierarchy.py` | 稀疏 P/B 应用、动态 `S(Q)`、Road 图差分、手工方向构建 |
| `src/hcfm/adversarial.py` | GATv2 RoadEncoder、Semantic/Domain、GRL、Cost/Domain predictor、Rank/Orthogonal、optimizer 覆盖 |
| `src/hcfm/interaction.py` | 可配置双向 gated residual Region--Road 交互 |
| `src/hcfm/model.py` | CRAFT MacroEncoder/GFA、Reference encoder、Macro FM、完整 HCFM、分组总损失 |
| `src/hcfm/flow_matching.py` | 直线路径、先验、图时序速度场、mask MSE、Euler/Heun/NFE |
| `src/hcfm/losses.py` | 物理单位 state、一致速度、拓扑差分损失 |
| `src/hcfm/calibration.py` | 源 train 非负比例校准与守恒 gap 报告 |
| `src/hcfm/rag.py` | source-train-only Region RAG，无后续检索创新 |
| `src/hcfm/dataset_builder.py` | macro/micro 同城同窗口 inner join 与整城快照组装 |
| `src/hcfm/checkpoint.py` | 完整 config/normalizer/data version/optimizer 保存恢复，stage1 部分加载报告 |
| `src/hcfm/metrics.py` | macro in/out、Road、topology、conservation 指标 |
| `src/hcfm/engine.py` | warm-up 阶段约束、所有子损失日志、单步/循环训练核心 |
| `src/hcfm/geo_time.py` | CRS 转换、IANA timezone、DST ambiguous/nonexistent 严格策略 |
| `src/hcfm/config.py` | YAML `base_config` 继承、模式/prior 约束 |

## 第一阶段兼容修改

- `src/gtg_features/pipeline.py` 在原 Region 9 维结果之外额外保存 `road_feat[M,9]`，原缓存内容/特征顺序不变。
- `src/craft_integrated/diffusion.py` 仅将 `unet.yaml` 改为按源文件路径定位。
- `src/craft_integrated/pyg_compat.py` 正常 ABI 时直接用 PyG；当前 torch 2.0/PyG pt21 ABI 错误时才回退纯 PyTorch Data/GATv2。baseline 无 GTG 的 state keys/数值保持不变。
- `FlowDataset/CRAFTModel/GaussianDiffusion1D/Unet1D/Retriever/evaluate.py` 均未删除；第一阶段测试验证了 baseline 数值等价和旧 state dict missing/unexpected 规则。

## 真实 Stage A 产物

生成产物位于 Git 忽略的 `cache/`，未复制原数据：

| city | `region_x` | `road_x` | Road edges | P nnz | B_in/B_out nnz | cross Region |
|---|---|---:|---:|---:|---:|---:|
| chi | `[73,45]` | `[52681,15]` | 270,855 | 56,471 | 3,799 / 3,799 | 7.14% |
| dc | `[81,45]` | `[70499,15]` | 363,277 | 75,608 | 5,197 / 5,198 | 7.06% |
| toronto | `[60,45]` | `[37943,15]` | 200,884 | 41,119 | 3,224 / 3,223 | 8.21% |
| ny | `[95,45]` | `[68995,15]` | 292,191 | 73,733 | 4,782 / 4,784 | 6.63% |

Toronto 5 个、NY 2 个无 Road Region 保持 P 空行。CRAFT 边界框内道路端点均落在 Region 集合内，四城 outside start/end 都是 0。源 `road.csv` 没有真实 `osm_way_id`，manifest 保存 `parent_source_road_id` 并令 `parent_osm_way_ids=null`，没有伪造。

边界矩阵按完整 Region 序列构建：chi/dc/toronto/ny 分别有 3,799/5,200/3,228/4,790 次边界转移，多于跨 Region Road 数，证明中间 Region 转移已纳入。缓存同时显式保存 `road_to_region_edge_index/weight` 与 `region_to_road_edge_index/weight`。

## 数据 bundle 与训练阶段

`assemble_joint_samples` 只对 `(date,start_hour)` inner join，dynamic metadata 再验证 city/date/hour/split/timezone/DST。macro 缺 Region 使用显式 `region_mask=False`；micro 快照必须列全稳定 `directed_road_id`，否则报错。A3/A4 的 `generate_micro=false` 是真实模块开关：不实例化 Micro vector field，Macro field 不接动态 micro state，Dataset 不要求 `micro_flow/road_mask`。

训练阶段由 `training.phase` 拦截提前启用损失：

1. `encoder_pretrain`：GTG cost/rank/domain/orthogonal + GFA；
2. `macro_fm` / `macro_adversarial`：Macro FM，后者加入 Road 对抗；
3. `joint_fm`：加入 Micro FM/topology；
4. `cross_state`：加入物理状态一致性；
5. `coupled_velocity`：仅 coupled prior 可加入速度一致性。

目标城市静态 Road 图进入 domain discriminator；API 没有 target cost target 参数。RAG 和归一化器均验证 source city + train split。

## checkpoint

HCFM v1 保存：model、optimizer、完整 config、所有 normalizer/calibrator state、data version、step。恢复时 model strict load，data version/normalizer 名称与元数据不符即失败。

`load_stage1_gfa` 只接收 `rep_model.pth` 并映射 `init_proj/gnn`；`load_stage1_craft_conditions` 只映射时间/Reference 模块，`generator_model` Diffusion 键全部显式 skipped。当前 Paper 无真实 `.pth` 文件，因此只完成了结构/合成 checkpoint 恢复测试，未声称加载服务器外 checkpoint。

## 未实现范围

未加入可学习/多级/Road RAG、重排损失、GAN、生成判别器或任何后续创新点。
