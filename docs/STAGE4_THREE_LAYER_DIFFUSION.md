# Stage 4 三层 Hierarchical Conditional Diffusion

## 1. 状态与主路线

当前正式主路线为：

```text
Road / Syntax / Region static hierarchy
        ↓
single joint GraphGPS + joint LapPE
        ↓
H_*_low                              H_*_high
   ↓                                     ↓
history_Y_* + calendar                   │
   ↓                                     │
Hierarchical RAG                         │
   ↓                                     │
R_region / R_syntax / R_road ────────────┘
                    ↓
Hierarchical Conditional Diffusion
Region → Syntax → Road
                    ↓
normalized future dynamics
                    ↓
source-train normalizer inverse
                    ↓
physical three-layer dynamics
```

Stage 4 的代码、配置、训练/生成 CLI、unit tests 和 tiny synthetic smoke test 已完成。
截至本文档更新时尚未启动耗时的真实三城联合训练，因此没有 Stage 4 真实 checkpoint、
生成样本或城市指标。legacy `src/hcfm/` 仍保留用于历史实验，但不属于当前主路线。

## 2. 原始 CRAFT Diffusion 调用链

主要参考仓库是 `/root/autodl-tmp/projects/CRAFT`，实际调用链为：

```text
train.py::train_diffusion
  → CRAFTModel.calc_loss
  → CRAFTModel.get_cond
  → GaussianDiffusion1D.calc_loss
  → Unet1D(x_t, timestep, condition, self_condition)

generate.py::run_generation
  → CRAFTModel.generate
  → GaussianDiffusion1D.generate
  → loop_sample (DDPM) / skip_sample (DDIM)
```

本实现保留了 CRAFT 的 linear/cosine schedule、epsilon prediction、1D U-Net、
self-conditioning、DDPM、DDIM、EMA 和 sinusoidal timestep embedding。以下部分按当前
工程契约修正：

- 所有 beta/alpha/posterior 系数通过 `register_buffer` 注册；
- 不依赖修改 `sys.path` 或 CRAFT 的相对配置文件；
- 默认 `clip_x0=false`；
- 不执行 CRAFT 的 `[-1,1]` 输入变换和 `(x+1)/2` 后处理；
- 使用本项目 source-train `log1p_zscore` normalizer 反变换；
- 物理空间的 flow/speed/travel-time 输出约束为非负；
- 节点折叠到 batch，注意力只作用于 `T=24` 时间轴。

原始 CRAFT 仓库未被修改。

## 3. Diffusion 数学与损失

每层真实 normalized future `x0` 独立加噪：

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
epsilon ~ Normal(0, I)
```

U-Net 预测 `epsilon_theta(x_t,t,condition)`。每层损失按 mask 的有效元素数独立归一化：

```text
L_layer = sum(mask * (epsilon_theta-epsilon)^2) / max(sum(mask), 1)
L_total = lambda_region*L_region
        + lambda_syntax*L_syntax
        + lambda_road*L_road
```

Road 节点数不会改变 Region/Syntax loss 的尺度。接口允许传入 mask；没有显式 mask
时使用全 1。所有输入在进入训练前必须有限，NaN/Inf 会显式报错，不能静默变成零。

## 4. 三层专家与 shape

`HierarchicalThreeLayerDiffusion` 是单一层次外壳，内部包含：

```text
region_diffusion: 2 channels (in_flow/out_flow)
syntax_diffusion: 3 channels (passage_count/speed_kmh/travel_time_seconds)
road_diffusion:   3 channels (passage_count/speed_kmh/travel_time_seconds)
```

不同城市的节点数不同，因此每层执行：

```text
[B,N,C,T] → [B*N,C,T] → ConditionalUnet1D → [B*N,C,T] → [B,N,C,T]
```

U-Net 的 linear/full attention 都只在 temporal length 上运行，不构建 `[N,N]` 或
`[M,M]` 稠密注意力矩阵。

## 5. 条件边界

低频和高频严格分工：

```text
H_*_low  → RAG only
H_*_high → Diffusion only
```

条件分别为：

```text
Region:
H_region_high + encode(R_region) + calendar

Syntax:
H_syntax_high + encode(R_syntax) + calendar
+ encode(broadcast(Region future))

Road:
H_road_high + encode(R_road) + calendar
+ encode(broadcast(Syntax future))
```

拼接后经 `LayerNorm → MLP → cond_emb` 进入 CRAFT-style U-Net。RAG reference 只是
条件先验；采样起点始终是 Gaussian `x_T`，不会用 reference 替代噪声。

## 6. 稀疏父层广播

静态算子的存储语义是：

```text
road_to_syntax:   row=Syntax parent, col=Road child
syntax_to_region: row=Region parent, col=Syntax child
```

父层 future 向子层传播时使用转置语义，并对每个 child 的入边权重重新归一化：

```text
Region → Syntax: syntax_to_region.T
Syntax → Road:   road_to_syntax.T
```

实现使用 `index_add_` 的 sparse scatter，不生成稠密父子矩阵；Syntax→Region 的
多对多 UTM 几何权重得到保留，也没有 Road→Region 快捷边。

## 7. 训练和推理差异

训练 snapshot 的时间契约：

```text
history_temporal_features = [t-24,t) → RAG Query/Key
value_temporal_features   = [t,t+24) → Diffusion x0 / source RAG Value
```

对目标城市 A，RAG 通过 `target_city=A` 排除 A 的所有 source candidates，训练和验证
均保持 Leave-One-City-Out。

第一版采用明确 teacher forcing：

```text
train:
real Region future → Syntax parent condition
real Syntax future → Road parent condition

inference:
generated Region future → Syntax parent condition
generated Syntax future → Road parent condition
```

可配置 `parent_condition_dropout` 缓解 exposure gap。推理不会读取真实父层 future。
`ThreeLayerRAGDiffusionSystem.generate()` 会重建一个
`value_temporal_features=None` 的 query；若原输入带 future，只能在完整生成后用于
离线评估。

## 8. 数据、分桶与身份校验

Stage 4 使用：

```text
outputs/stage2_three_layer_graphgps_lappe/spectral_features/
outputs/stage3_three_layer_rag/train_snapshots.pt
outputs/stage3_three_layer_rag/eval_snapshots.pt
outputs/stage3_three_layer_rag/rag_memory_v3.pt
outputs/stage3_three_layer_rag/train_snapshots.normalizer.json
```

`CityGraphBucketBatchSampler` 只把相同 city 和 `joint_graph_hash` 的 snapshot 放入同一
bucket；trainer 在 bucket 内逐 snapshot 累积并平均 loss，不跨城市强行 stack，不用
错误 padding 让无效节点参与训练。

加载时校验：

- `city_id`；
- `joint_graph_hash`；
- GraphGPS `checkpoint_fingerprint`；
- `static_feature_version` / `spectral_feature_version`；
- 三层 node ranges 与 stable order；
- RAG memory version/graph identity；
- RAG memory 文件内容 SHA-256；
- LapPE version、weighted spectrum hash 与 GraphGPS global attention scope；
- dynamic normalizer SHA-256。

## 9. Checkpoint 与评估

checkpoint 包含：

```text
rag_model_state
diffusion_state
ema_state
optimizer_state / scheduler_state
epoch / global_step / random_seed / RNG state
complete config
GraphGPS fingerprint
per-city joint graph hashes
static/spectral feature versions
RAG memory version
RAG memory file SHA-256
LapPE version / per-city weighted spectrum hashes / attention scope
dynamic normalizer fingerprint
Diffusion contract version
```

不一致会显式拒绝加载。验证输出三层 normalized epsilon loss、RAG 实际 top-k 与
top-1 权重诊断；配置的 generation validation 还输出反归一化 MAE/RMSE、每通道
MAE/RMSE、三层加权指标，以及 DDPM/DDIM 的真实采样步数。

## 10. 命令

联合训练：

```bash
python scripts/train_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --device cuda:0
```

可先限制 snapshot/epoch 做本地性能检查：

```bash
python scripts/train_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --device cuda:0 \
  --max-epochs 1 \
  --max-train-snapshots 3 \
  --max-validation-snapshots 1
```

正式生成：

```bash
python scripts/generate_three_layer_diffusion.py \
  --config configs/stage4_three_layer_diffusion.yaml \
  --checkpoint outputs/stage4_three_layer_diffusion/best.pt \
  --input outputs/stage3_three_layer_rag/eval_snapshots.pt \
  --output outputs/stage4_three_layer_diffusion/generated \
  --device cuda:0
```

## 11. 当前验证边界与风险

已完成的验证是 unit tests 与 tiny synthetic end-to-end smoke。尚未运行：

- 三城 Stage 4 正式 GPU 训练；
- 真实 eval snapshot 的完整 500-step 生成；
- 真实物理空间 MAE/RMSE；
- Diffusion 输出到 GTG 轨迹解码的对接。

当前实现已采用分块、分城市 Top-K 检索；source key 在训练阶段保持可微在线计算，
在 eval/generation 阶段按 memory/device/dtype 缓存。仍需在正式三城 GPU 训练前进行
少量性能 smoke，确认实际 batch 与显存配置。

Stage-4 验证使用固定的 timestep/noise bank（不启用随机 self-conditioning），因此
best checkpoint 和 scheduler 监控的是可重复的 normalized epsilon loss。恢复训练会
恢复 optimizer、scheduler、EMA、RNG、best metric 与 history。

EMA 每次完成参数/浮点 buffer 更新后都会调用 `ema_model.rag.clear_source_cache()`；因此
下一次验证会用当前 EMA 的 temporal/key encoder 重新计算 eligible source keys，不会沿用
上一 epoch 的缓存。
