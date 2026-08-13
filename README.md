# Paper — GTG 拓扑特征融入 CRAFT（第一阶段）

以 **CRAFT**（Cross-City Traffic Flow Generation via Retrieval-Augmented Diffusion）为主框架，
在其区域表征进入 GFA 之前，融入 **GTG** 的空间句法 / 路网拓扑特征。CRAFT 的 Population/POI/Road
特征、区域网格、空间编码器、GFA、检索增强、Diffusion 训练/采样/测试、数据划分、损失、指标全部保留。

> 约束：所有新增/修改内容只在本 `Paper/` 目录内；`CRAFT/`、`GTG/`、原始数据集只读。

## 目录结构

```
Paper/
├── README.md                     # 本文件
├── AGENTS.md                     # 代理/协作与红线约定
├── configs/
│   ├── baseline.yaml             # use_gtg_topology=false, 与原 CRAFT 等价
│   └── fusion.yaml               # use_gtg_topology=true, GTG 融合
├── scripts/
│   ├── gen_norm_flow.py          # 重建缺失的 norm_*.csv (train 拟合, 无泄漏)
│   └── build_gtg_features.py     # 离线构建 GTG 区域拓扑特征缓存
├── src/
│   ├── gtg_features/             # 空间句法/拓扑特征预处理管线 (新增)
│   │   ├── dual_graph.py         #   路段对偶图 (移植 GTG gen_edge_data)
│   │   ├── space_syntax.py       #   Connectivity/Total Depth/Integration/Choice (graph_tool)
│   │   ├── partition.py          #   Metis 分区 + 分区池化 (pymetis)
│   │   ├── road_to_region.py     #   路段特征→区域 (长度加权 sjoin)
│   │   └── pipeline.py           #   编排 + 缓存
│   └── craft_integrated/         # CRAFT 副本 + 最小侵入式融合改动
│       ├── data_loaders.py       #   configure() / 融合模式拼接 GTG 特征 (严格校验)
│       ├── rep_model.py          #   GTAggregator 加 gtg_branch + fusion_proj
│       ├── main.py               #   --config / configure()
│       └── ...                   #   diffusion/unet/retrieve/train/... (原样)
├── data/norm_flow/{city}/        # 生成的归一化流量 (train/test)
├── cache/gtg/{city}_gtg_*.npz/json  # GTG 区域特征缓存 + meta
├── tests/                        # pytest 单元/集成/回归测试
└── docs/
    ├── STAGE1_AUDIT.md           # 审计: 路径/数据流/张量形状/风险
    ├── STAGE1_IMPLEMENTATION.md  # 实现说明 (本阶段核心)
    ├── STAGE1_TEST_REPORT.md     # 测试报告 (真实运行结果)
    └── GTG_FEATURE_COVERAGE.md   # 特征覆盖率与质量
```

## 快速开始

```bash
cd /root/autodl-tmp/projects/Paper

# 1) 生成缺失的 norm 流量文件 (只读 CRAFT slide_bike_flow, 写 data/norm_flow)
python3 scripts/gen_norm_flow.py

# 2) 离线构建 GTG 区域拓扑特征 (graph_tool 环境; 写 cache/gtg)
python3 scripts/build_gtg_features.py

# 3) 训练 (torch 环境)
cd src/craft_integrated
python3 main.py --config ../../configs/fusion.yaml     # GTG 融合
python3 main.py --config ../../configs/baseline.yaml   # 基线对照
```

## 融合注入点

节点特征 `[craft(45) | gtg(9)]` 进入 `GTAggregator.forward`：CRAFT 45 维走 `init_proj`，
GTG 9 维走 `gtg_branch`（GATv2 + 逐图标准化），concat 后 `fusion_proj`，再进 GFA
（GraphTransformer）。融合在 **GFA 之前的表征层**，不改 Diffusion。

## 检查点兼容

- 基线与原始 CRAFT **数值等价**（`allclose atol=1e-6`）。
- 旧 ckpt 可加载进融合模型：`init_proj/gnn` 键不变，仅 `gtg_branch/fusion_proj` 为新增 missing keys。

## 测试

见 `docs/STAGE1_TEST_REPORT.md`。因 `graph_tool` 与 `torch` 库冲突，测试分两次调用：

```bash
cd tests
# graph_tool 相关 (对偶图/空间句法/metis)
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 -m pytest test_dual_graph.py -v
# torch/数据/特征/norm
python3 -m pytest test_model_integration.py test_data_loaders.py test_gtg_features.py test_norm_flow.py -v
```

## 相关文档

- 审计与设计依据：[docs/STAGE1_AUDIT.md](docs/STAGE1_AUDIT.md)
- 实现细节：[docs/STAGE1_IMPLEMENTATION.md](docs/STAGE1_IMPLEMENTATION.md)
- 特征覆盖率：[docs/GTG_FEATURE_COVERAGE.md](docs/GTG_FEATURE_COVERAGE.md)
- 测试报告：[docs/STAGE1_TEST_REPORT.md](docs/STAGE1_TEST_REPORT.md)
