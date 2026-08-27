# 第二阶段实验与消融计划

## 1. 固定实验协议

每次跨城市实验明确 `src_cities/trg_cities`。源城市 train 动态可训练；目标城市只提供静态 Region/Road/P，目标 val/test 动态只评价。normalizer、conservation calibrator、RAG DB 都只在源 train 拟合。零样本主实验不混入目标 few-shot；few-shot 必须另建配置。

随机 seed、数据版本、feature order、split/date、normalizer/calibrator metadata 与完整配置进入 checkpoint。所有消融使用相同源/目标、快照、训练预算、ODE solver/steps（除明确研究 solver 的实验），并分别报告 in/out。

## 2. Warm-up

1. Stage B `encoder_pretrain`：可先用 GTG-main 三城独立预训练 RoadEncoder/语义域模块；其动态标签绝不与 CRAFT 宏观流联合。CRAFT 同城数据到位后，以源城市 cost/rank 继续训练。
2. Stage C `macro_fm`：加载第一阶段 Region/GFA/Reference 可兼容权重，只训练 Macro FM，建立 Diffusion 公平对比。
3. Stage D `joint_fm`：加入 Road 静态层次表示和 Micro FM，再启 topology。
4. `cross_state`：源 train 拟合 passage→macro calibration 后启 0.1 state。
5. Stage E `coupled_velocity`：模型稳定后切 coupled prior，先很小 velocity weight，再观察梯度/守恒 gap。

## 3. A0--A8

| ID | 配置 | 唯一主要增量 |
|---|---|---|
| A0 | `A0_craft_diffusion.yaml` | 原 CRAFT Diffusion，无 GTG |
| A1 | `A1_gtg_craft_diffusion.yaml` | + 第一阶段 Region GTG |
| A2 | `A2_macro_fm.yaml` | A1 条件，Diffusion→Macro FM |
| A3 | `A3_macro_fm_hierarchy.yaml` | + 显式 Road/P 双向层次静态条件 |
| A4 | `A4_hierarchy_adversarial.yaml` | + GTG 道路领域对抗 |
| A5 | `A5_micro_fm.yaml` | + Micro FM/topology |
| A6 | `A6_state_consistency.yaml` | + 物理状态一致性 |
| A7 | `A7_coupled_velocity.yaml` | + coupled prior/velocity consistency |
| A8 | `A8_full_hcfm.yaml` | 完整 HCFM 固化配置 |

配置只建立入口，不在本阶段启动长实验。A3/A4 的 `generate_micro=false` 已实现为真实模块开关：只使用 Road 静态/对抗条件，不实例化 Micro vector field，也不要求 micro 动态字段；A5 才首次要求 `micro_flow/road_mask`。

## 4. 指标与日志

- Macro：CPC、Min-Max MAE/RMSE，in/out 分开。
- Micro：Road CPC/MAE/RMSE、topology difference error。
- Cross-scale：MAE、relative error、per-Region gap、inflow/outflow consistency。
- Representation：domain accuracy、cost MSE、rank、semantic-domain cosine。
- Efficiency：NFE、single-batch latency、samples/s、peak GPU memory、parameters。
- Loss：`L_macro/L_micro/L_cross_scale` 及全部子损失，不只打印 total。

## 5. 正式训练前门禁

必须先取得至少一个 CRAFT 城市同一日期/时区的 directed-road passage count 或完整 GPS 轨迹，完成方向/连续性 map matching 报告。随后：

1. 生成 micro cache 并验证 Road id 100% 对齐；
2. 显式确认 IANA timezone/DST 和滑窗口径；
3. 输出真实 macro/micro 联合窗口与缺失率；
4. 在源 train 拟合 normalizer/calibrator，验证目标无泄漏；
5. 运行真实 `S(Q_true)` 绝对/相对/逐 Region gap；
6. 完整真实 smoke 后才可启动 A0--A8。
