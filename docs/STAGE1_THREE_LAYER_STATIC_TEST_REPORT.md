# 第一阶段三层静态图测试报告

## 执行环境

- 工作目录：`/root/autodl-tmp/projects/Paper`
- Python：`3.10`
- 设备：`cpu`
- 目标城市本次验证使用命令行显式传入的 `chi`；配置文件中的 `target_city` 仍为 `null`。

## 自动化回归测试

实际执行命令：

```bash
pytest -q tests/test_model_integration.py tests/test_data_loaders.py \
  tests/test_gtg_features.py tests/test_norm_flow.py \
  tests/test_gtg_preprocessing.py tests/test_population_strict.py \
  tests/test_static_hierarchy.py
```

结果：`63 passed, 9 warnings`，耗时约 `165.40s`。该命令在最后一轮严格校验补丁之前执行；最后一轮只重新执行了下方的三层专项测试和语法检查。

警告来自当前环境中 `torch-scatter`、`torch-cluster`、`torch-spline-conv`、`torch-sparse` 的 ABI 不匹配以及 `pyproj` 的弃用提示；仓库的 `pyg_compat` 后备实现使测试继续执行。警告不是测试失败。

新增三层测试单独执行：

```bash
pytest -q tests/test_static_hierarchy.py
```

最终一轮结果：`15 passed, 4 warnings`，耗时约 `6.42s`。

多 Source 训练协议专项测试：

```bash
pytest -q tests/test_stage1_training_protocol.py
```

最终补丁后的结果：`3 passed, 4 warnings`，覆盖 TFA 城市等权、CCA 完整 Source、每城 `1/S` OT 边际、`1-cosine` 代价和禁止 Euclidean 回退。

最终补丁后的第一阶段专项合并执行：

```bash
pytest -q tests/test_static_hierarchy.py tests/test_stage1_training_protocol.py
```

结果：`18 passed, 4 warnings`。

## 真实四城入口验证

验证命令：

```bash
python3 scripts/run_stage1_static.py \
  --config configs/stage1_three_layer_static.yaml \
  --action validate \
  --source_cities beijing chengdushi xianshi \
  --target_city chi
```

结果：四城均通过严格构建与契约校验：

| city | regions | roads | syntax nodes |
|---|---:|---:|---:|
| `beijing` | 81 | 14685 | 293 |
| `chengdushi` | 23 | 3514 | 70 |
| `xianshi` | 26 | 4147 | 82 |
| `chi` | 73 | 31093 | 621 |

前向与梯度连通性命令：

```bash
python3 scripts/run_stage1_static.py \
  --config configs/stage1_three_layer_static.yaml \
  --action smoke \
  --source_cities beijing chengdushi xianshi \
  --target_city chi
```

结果：`beijing [81,128]`、`chengdushi [23,128]`、`xianshi [26,128]`、`chi [73,128]`，`grad_check: true`，输出全部 finite。

原 CRAFT 静态表征机制的 1 epoch 实际执行：

```bash
python3 scripts/run_stage1_static.py \
  --config configs/stage1_three_layer_static.yaml \
  --action pretrain \
  --source_cities beijing chengdushi xianshi \
  --target_city chi
```

结果：命令正常结束并写入 `outputs/stage1_static/` 下四城 `*_region_rep.npy` 和 `static_encoder.pth`。这只是 1 epoch 的静态预训练验证，不代表完整研究训练收敛。

## 未运行项目

本报告没有运行 `src/craft_integrated/retrieve.py`、`diffusion.py`、`unet.py`、`generate.py`、`src/hcfm/`、Flow Matching、RAG、Diffusion 训练/生成或 Stage 2 测试，也没有进行消融实验。
