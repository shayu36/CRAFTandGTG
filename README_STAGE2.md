# HCFM：第二阶段创新点 1

本目录在第一阶段 GTG+CRAFT 集成之上新增 **Hierarchical Cross-City Flow Matching**。原 `craft_integrated` 的 `FlowDataset -> GFA -> Region RAG -> GaussianDiffusion1D -> Unet1D` 保留；新增代码位于 `src/hcfm/`，不删除或替换 Diffusion。

## 三种主模式

| 配置 | 模式 | 语义 |
|---|---|---|
| `configs/stage1_diffusion.yaml` | `stage1_diffusion` | CRAFT 45 + GTG Region 9 + GTAggregator/GFA + Region RAG + 原 Diffusion |
| `configs/stage2_macro_fm.yaml` | `macro_flow_matching` | 第一阶段 Region 条件不变，仅将宏观生成器替换为连续 Flow Matching |
| `configs/stage2_hierarchical_fm.yaml` | `hierarchical_flow_matching` | 显式 Region/Road/P/B + GTG 道路对抗 + 双向层次交互 + GFA/RAG + Macro/Micro FM |
| `configs/stage2_hierarchical_fm_coupled.yaml` | `hierarchical_flow_matching` | 在完整模式上启用 coupled prior、Heun 和速度一致性 |

`configs/ablations/A0_*.yaml` 到 `A8_*.yaml` 使用 `base_config` 递归继承，只覆盖目标模块。

## 已验证的真实数据状态

CRAFT 四城的静态层次缓存已经生成在忽略 Git 的 `cache/hcfm/`：chi 73 Region/52,681 directed roads，dc 81/70,499，toronto 60/37,943，ny 95/68,995。Road 节点包含 6 维道路属性/方向和真实重建的 9 维 GTG 空间句法。`P_struct` 使用 UTM 相交长度并支持跨 Region 软归属；`B_in/B_out` 沿每条有向 Road 的完整有序 Region 相交序列记录全部边界转移，而非只比较首尾端点。

服务器目前没有同城、同时间的 `road_passage_count`。CRAFT `bike_trip.csv` 只有 OD 起终点，不满足 map matching 契约；GTG-main 的北京/成都/西安 `rid_list` 不能与 CRAFT 四城拼接。因此真实联合训练/生成仍严格阻塞，代码不会用 OSM 静态属性、最近道路、最短路或随机数据补目标。

## 可复制命令

以下命令均从 `/root/autodl-tmp/projects/Paper` 执行。

```bash
# 只读数据审计
python3 scripts/audit_stage2_data.py

# Road 级 GTG 空间句法缓存（graph_tool 进程）
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 scripts/build_gtg_features.py --cities chi dc toronto ny

# Region/Road/P/B 静态层次缓存
python3 scripts/run_stage2.py \
  --config configs/stage2_hierarchical_fm.yaml \
  --action preprocess --cities chi dc toronto ny

# 检查配置与消融继承
python3 scripts/run_stage2.py \
  --config configs/stage2_hierarchical_fm.yaml --action validate
```

第一阶段原入口保持不变；下面命令会启动正式训练，按需调整城市和 GPU，不应在 smoke 阶段直接运行：

```bash
python3 src/craft_integrated/main.py \
  --config configs/stage1_diffusion.yaml \
  --src_cities chi dc toronto --trg_cities ny --device cuda:0
```

Macro FM 使用真实 `macro_bundle.pt`，其中每个快照含 `aligned_region_rep/reference/macro_flow/region_mask/region_edge_index`：

```bash
python3 scripts/run_macro_fm.py --config configs/stage2_macro_fm.yaml \
  --action smoke --bundle /path/to/macro_bundle.pt --device cuda:0
python3 scripts/run_macro_fm.py --config configs/stage2_macro_fm.yaml \
  --action train --bundle /path/to/macro_bundle.pt --device cuda:0
python3 scripts/run_macro_fm.py --config configs/stage2_macro_fm.yaml \
  --action generate --bundle /path/to/macro_bundle.pt --device cuda:0
python3 scripts/run_macro_fm.py --config configs/stage2_macro_fm.yaml \
  --action evaluate --bundle /path/to/macro_bundle.pt --device cuda:0
```

完整 HCFM 的 bundle 必须来自 `assemble_joint_samples`，且在 cost/rank 权重非零时带源城市 `road_cost_target/road_cost_mask`：

```bash
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml \
  --action smoke --bundle /path/to/real_joint_bundle.pt --device cuda:0
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml \
  --action train --bundle /path/to/real_joint_bundle.pt --device cuda:0
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml \
  --action generate --bundle /path/to/real_joint_bundle.pt --device cuda:0
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml \
  --action evaluate --bundle /path/to/real_joint_bundle.pt --device cuda:0
```

## 测试

```bash
python3 -m pytest tests/test_hcfm_*.py -q
python3 -m pytest tests/test_model_integration.py tests/test_data_loaders.py \
  tests/test_gtg_features.py tests/test_norm_flow.py -q
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 -m pytest tests/test_dual_graph.py -q
```

真实执行结果见 `docs/STAGE2_TEST_REPORT.md`；模型公式、数据契约和缺口分别见 `docs/STAGE2_MODELING.md`、`docs/STAGE2_DATA_CONTRACT.md`、`docs/STAGE2_DATA_AUDIT.md`。
