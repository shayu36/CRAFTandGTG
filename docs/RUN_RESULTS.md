# 当前运行结果

更新时间：2026-09-20。以下只记录当前有效产物和已经实际验证的结果。

## 环境

```text
GPU: 5 × NVIDIA GeForce RTX 2080 Ti，单卡约 11 GiB
PyTorch: 2.0.1+cu118
Python: 3.10
CUDA: available
```

## Stage 2

```text
城市: beijing, chengdushi, xianshi
epoch: 100
最优 epoch: 97
最优 valid_city_macro_rmse: 0.06179773683349291
```

产物为：

```text
outputs/stage2_three_layer_graphgps_lappe/best.pt
outputs/stage2_three_layer_graphgps_lappe/last.pt
outputs/stage2_three_layer_graphgps_lappe/metrics.json
outputs/stage2_three_layer_graphgps_lappe/spectral_features/*_spectral_features.pt
```

## Stage 3

```text
train snapshots: 69
  beijing=23, chengdushi=23, xianshi=23
eval snapshots: 18
  每城 val=3, test=3
RAG memory: 69 个 source-train snapshots
RAG version: three-layer-hierarchical-rag-v3-weighted-stage2
```

当前有效文件：

```text
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/train_snapshots.normalizer.json
outputs/stage3_three_layer_rag/rag_memory_v3.pt
```

## Stage 4 单卡 smoke

已成功完成 1 个 epoch：

```text
train_loss: 35.48188781738281
validation_normalized_noise_loss: 3.570333480834961
validation_region_normalized_noise_loss: 1.2865732908248901
validation_syntax_normalized_noise_loss: 1.086984395980835
validation_road_normalized_noise_loss: 1.1967759132385254
validation_rag_mean_actual_top_k: 5.0
validation_rag_mean_top1_weight: 0.20030240714550018
```

## Stage 4 五卡 smoke

当前使用的命令为：

```bash
torchrun --standalone --nproc_per_node=5 scripts/train_three_layer_diffusion.py --config configs/stage4_three_layer_diffusion.yaml --device cuda:0 --max-epochs 1 --max-train-snapshots 1 --max-validation-snapshots 1
```

五卡 smoke 已成功完成，`EXIT_CODE=0`：

```text
epoch: 0
train_loss: 35.48188781738281
validation_normalized_noise_loss: 3.570333480834961
validation_region_normalized_noise_loss: 1.2865732908248901
validation_syntax_normalized_noise_loss: 1.086984395980835
validation_road_normalized_noise_loss: 1.1967759132385254
validation_rag_mean_actual_top_k: 5.0
validation_rag_mean_top1_weight: 0.20030241211255392
```

## 测试

最近一次定向测试：

```text
python -m py_compile scripts/train_three_layer_diffusion.py src/three_layer_diffusion/training.py
pytest -q tests/test_three_layer_diffusion.py tests/test_three_layer_rag.py tests/test_retriever_strict.py
36 passed in 10.73s
```

## 下一步

五卡 smoke 已通过，下一步使用 tmux 启动正式三城 Stage 4 训练；训练完成后运行 `generate_three_layer_diffusion.py`，再检查生成结果和物理空间指标。
