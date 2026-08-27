# 第二阶段测试报告

本报告只记录 2026-08-19 在 `/root/autodl-tmp/projects/Paper` 真实执行过的命令和输出。

## 1. 第一阶段回归

初次执行主体回归在收集阶段失败：系统 `torch==2.0.1+cu118`，而 `torch_cluster/torch_scatter/torch_sparse/torch_spline_conv` 为 `+pt21cu118`，原始错误包含 `undefined symbol: _ZN3c106SymInt19promote_to_negativeEv`。加入仅在 import 失败时启用的 Paper 本地兼容层后重跑：

```bash
python3 -m pytest tests/test_model_integration.py tests/test_data_loaders.py \
  tests/test_gtg_features.py tests/test_norm_flow.py -q
```

结果：`44 passed, 20 warnings in 182.17s`。warnings 包含上述 ABI 探测警告、TypedStorage 和 pandas deprecation；没有隐藏。

```bash
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 -m pytest tests/test_dual_graph.py -q
```

结果：`4 passed, 1 warning in 1.49s`。

覆盖：baseline 与原始 CRAFT state keys/数值等价；45/54 维；旧 baseline state dict 加载到 fusion 时只缺 GTG 新键；真实四城 Region/GTG/norm cache；对偶图/空间句法/Metis。

## 2. HCFM 单元与算法 smoke

早期测试先得到 `29 passed, 3 failed`，据此修复 `reference` collator batch 维、一个测试张量 shape 和 RAG 文案。重跑得到 `32 passed in 8.03s`。加入 CRS/DST/真实缓存/校准后最终执行：

```bash
python3 -m pytest tests/test_hcfm_*.py -q
```

最终 HCFM 单独收集为 46 项；最终组合命令如下：

```bash
git diff --check && \
python3 -m pytest tests/test_hcfm_*.py tests/test_model_integration.py -q
```

结果：`52 passed, 9 warnings in 5.33s`（46 HCFM + 6 第一阶段模型兼容）。HCFM 新增项包括 A3 在无 `micro_flow` 时的真实模块开关、A→B→C 单 Road 全边界事件、完整 coupled velocity 模型训练步和跨城市表征指标。warnings 是当前 PyG ABI 探测、pyproj/NumPy deprecation 和 TypedStorage。

最终 graph_tool 复跑为 `4 passed, 1 warning in 1.51s`。

真实覆盖项：

- 四城 `region_x/road_x/P/B` 缓存 shape、finite、P 非空行归一化、B nnz；
- CRAFT/GTG id 顺序、目标泄漏、source-only normalizer/RAG/calibration；
- CRS 4326→32618→4326；IANA timezone 必填；NY DST ambiguous 默认报错；
- Macro FM 前向/反向与 Heun。

手工小图算法覆盖项（不作为真实数据实验）：

- 进入只增 in、离开只增 out、内部 Road 不计、双向分离、城市边界；
- `t=0/0.5/1`、直线路径、目标速度、终点估计、mask、梯度、输出 shape/finite；
- Euler/Heun 固定速度解析解、NFE、确定性；
- independent 禁止 velocity loss，coupled `X0` 与 `U` 可聚合；
- 物理单位状态 loss 精确一致为 0、错误方向非零、padding 排除；
- topology graph-difference，不是相邻流量相等；
- GRL 前向恒等/反向 -coefficient；Road/Semantic/Domain/Cost/Discriminator optimizer 与梯度；
- 完整小图 Dataset→Macro/Road encode→hierarchy→GFA→Reference→双 FM→全部损失→backward→optimizer→checkpoint save/reload→Euler。

## 3. 真实静态预处理

Road 级 GTG 重建：

```bash
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 scripts/build_gtg_features.py --cities toronto
```

结果：Toronto 成功，`real 0m40.608s`。

```bash
LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1" \
  python3 scripts/build_gtg_features.py --cities chi dc ny
```

结果：三城成功，`real 5m58.241s`。

在 Road 缓存生成前，真实 `preprocess --cities chi` 曾按预期失败：`FileNotFoundError: 严格模式: 缺少 Road 级 GTG 空间句法缓存 ... 禁止用 Region 级 9 维缓存冒充 Road 节点特征。`

生成 Road 缓存后执行静态层次预处理。Toronto 成功，`real 0m33.461s`。chi/dc/ny 初次批处理在 chi、dc 已落盘后因逐 Region endpoint 检查过慢被人工 `Ctrl-C`；改为空间索引后只重跑 NY：

```bash
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml \
  --action preprocess --cities ny
```

结果：NY 成功，`real 0m12.465s`。随后四城缓存加载/P 行和/finite assert 命令 exit code 0，真实摘要见 `STAGE2_IMPLEMENTATION.md`。

加入完整 Region 序列边界事件和双向层次 edge cache 后，最终四城重建命令成功，`real 0m41.455s`；四城真实缓存测试 `4 passed in 1.79s`。

数据审计：

```bash
python3 scripts/audit_stage2_data.py
```

结果：生成 `cache/hcfm/stage2_data_audit.json`，确认四城道路动态缺失率 100%、可联合窗口 0。

## 4. 三模式构造

从 Paper 根调用 `load_config + build_model`，结果：

```text
stage1_diffusion       CRAFTModel                 6,446,850 parameters
macro_flow_matching    MacroFlowMatchingModel     1,272,002 parameters
hierarchical_flow_matching HCFMModel              3,374,699 parameters
```

第一阶段最初还暴露 `diffusion.py` 相对 cwd 查找 `configs/unet.yaml` 的既有错误；修为相对源文件路径后以上三种均构造成功。

最终对三种主配置及 A0--A8 消融配置逐一执行轻量校验：

```bash
python3 scripts/run_stage2.py --config configs/stage1_diffusion.yaml --action validate
python3 scripts/run_stage2.py --config configs/stage2_macro_fm.yaml --action validate
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml --action validate
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm_coupled.yaml --action validate
for cfg in configs/ablations/A*.yaml; do
  python3 scripts/run_stage2.py --config "$cfg" --action validate
done
```

结果：13 个配置全部 exit code 0；independent/coupled prior、Micro FM、对抗模块和跨尺度损失开关均通过启动约束校验，未启动训练。

## 5. CLI 算法 bundle 检查

为验证新入口本身，在 Git 忽略的 `outputs/cli_smoke/` 写入手工确定性小图 bundle。它不是服务器真实交通数据，以下 loss/metric 不能作为实验结果。

```bash
python3 scripts/run_macro_fm.py --config configs/stage2_macro_fm.yaml \
  --action smoke --bundle outputs/cli_smoke/macro_bundle.pt \
  --checkpoint outputs/cli_smoke/macro_fm.pth --device cpu
```

结果：exit 0，1 optimizer step，`fm_macro=2.038745641708374`。随后 `--action evaluate` 成功严格恢复 checkpoint，Euler 16 steps、NFE 16、`has_nonfinite=false`，并输出 macro in/out CPC/MAE/RMSE。

```bash
python3 scripts/run_stage2.py --config configs/stage2_hierarchical_fm.yaml \
  --action smoke --bundle outputs/cli_smoke/hcfm_bundle.pt \
  --checkpoint outputs/cli_smoke/hcfm.pth --device cpu
```

结果：exit 0，1 optimizer step；日志包含 FM macro/micro、state、velocity、topology、cost、rank、两个 domain、orthogonal、GFA TFA/CCA、`L_macro/L_micro/L_cross_scale/total`。随后 `--action evaluate` 成功恢复 checkpoint 并完成双分支 Euler：macro `[1,2,2,24]`、micro `[1,3,1,24]`、NFE 16、`has_nonfinite=false`、CPU latency 0.3262s、参数量 3,374,699；宏/微/拓扑/跨尺度指标字段全部生成。

## 6. 未执行/未通过声明

- 未执行长时间正式训练或完整消融。
- Paper 内没有真实 `.pth/.pt/.ckpt` 第一阶段权重，因此没有执行真实旧 checkpoint 数值恢复；只执行了 state dict 兼容与 HCFM checkpoint roundtrip。
- 没有真实同城 Road passage count，故未执行真实联合 Dataset、真实 `S(Q_true)≈X_macro_true`、真实完整 HCFM smoke、真实微观/跨尺度指标。
- 测试使用的手工小图仅验证算法，不替代真实数据验收。
