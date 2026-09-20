# CRAFTandGTG

当前主路线：三层静态层级图 → 单一联合 GraphGPS + LapPE → 三层 Hierarchical RAG → Region → Syntax → Road Conditional Diffusion。

## 当前状态（2026-09-20）

| 阶段 | 当前状态 |
|---|---|
| Stage 2 GraphGPS + LapPE | 三城 100 epoch 完成；最优 `valid_city_macro_rmse=0.06179773683349291`（epoch 97） |
| Stage 2 谱特征 | 三城已导出并通过结构/频率分解校验 |
| Stage 3 snapshots | train 69 个、eval 18 个，normalizer 已固定 |
| Stage 3 RAG memory | `rag_memory_v3.pt` 已生成，包含 69 个 source-train snapshots |
| Stage 4 低显存训练 | 单卡 1 epoch smoke 已通过 |
| Stage 4 五卡训练 | `torchrun` 五卡 smoke 已通过，`EXIT_CODE=0` |

当前硬件为 5 张 NVIDIA RTX 2080 Ti（每张约 11 GiB）。Stage 4 使用节点分块、梯度检查点、CPU source-cache 和五卡显式梯度平均。

## 环境

```bash
cd /root/autodl-tmp/projects/Paper
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
```

`base` 环境通过 `/root/miniconda3/etc/conda/activate.d/paper_env.sh` 自动设置：

```text
PYTHONPATH=/root/autodl-tmp/projects/Paper/src
LD_PRELOAD=/root/miniconda3/lib/libgomp.so.1:/root/miniconda3/lib/libstdc++.so.6
CUDA_VISIBLE_DEVICES=0,1,2,3,4
```

已验证 PyTorch `2.0.1+cu118`、CUDA、5 张 GPU、PyG 扩展和 `graph_tool` 均可导入。

## 当前有效产物

```text
outputs/stage2_three_layer_graphgps_lappe/best.pt
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/train_snapshots.normalizer.json
outputs/stage3_three_layer_rag/rag_memory_v3.pt
outputs/stage4_three_layer_diffusion/best.pt
outputs/stage4_three_layer_diffusion/last.pt
outputs/stage4_three_layer_diffusion/history.json
```

当前训练链只使用上述 v3 产物，不混用旧 checkpoint、旧谱特征或旧 snapshot。

## 五卡训练

长期任务使用 tmux：

```bash
tmux new -s paper_stage4
cd /root/autodl-tmp/projects/Paper
torchrun --standalone --nproc_per_node=5 scripts/train_three_layer_diffusion.py --config configs/stage4_three_layer_diffusion.yaml --device cuda:0
```

退出但保持后台运行：按 `Ctrl-b`，再按 `d`。重新连接后查看：

```bash
tmux attach -t paper_stage4
```

恢复训练：

```bash
torchrun --standalone --nproc_per_node=5 scripts/train_three_layer_diffusion.py --config configs/stage4_three_layer_diffusion.yaml --device cuda:0 --resume outputs/stage4_three_layer_diffusion/last.pt
```

## 当前训练策略

- Stage 2：一个联合三层 GraphGPS，联合 LapPE 后显式拆分 `H_low/H_high`。
- Stage 3：RAG 使用三层 low/history/calendar；source memory 只来自三城 train split，并启用 graph identity 与 value separation 检查。
- Stage 4：训练使用真实父层 future teacher forcing，推理按 Region → Syntax → Road 顺序生成。
- 显存策略：`node_chunk_size=64`、`gradient_checkpointing=true`、`train_source_keys=false`、`source_cache_device=cpu`。
- 多卡策略：城市图 batch 按 rank 分发，各 rank 反向后显式 all-reduce 梯度；仅 rank 0 做验证、保存 checkpoint 和写日志。

## 测试

最近一次定向回归为 `36 passed`，覆盖 Stage 4 Diffusion、RAG、retriever 严格契约、节点分块、CPU source-cache 和梯度检查点路径。`graph_tool` 全量测试仍受外部 `GOMP_5.0` ABI 约束。

当前完整建模流程见 [docs/THREE_LAYER_MODELING_PIPELINE.md](docs/THREE_LAYER_MODELING_PIPELINE.md)，Word 版见 [docs/THREE_LAYER_MODELING_PIPELINE.docx](docs/THREE_LAYER_MODELING_PIPELINE.docx)。运行状态见 [docs/CODE_STATUS.md](docs/CODE_STATUS.md)，实际结果见 [docs/RUN_RESULTS.md](docs/RUN_RESULTS.md)。
