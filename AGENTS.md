# AGENTS.md — 协作约定与红线

本文件面向在本仓库上继续工作的任何代理/开发者，规定不可逾越的边界与工作方式。

## 硬性红线（必须遵守）

1. **写入范围**：所有新增/修改的代码、配置、文档、测试、小型中间结果，**只允许放在 `Paper/` 内**。
2. **只读原始资产**：`GTG/`、`CRAFT/` 原始目录、原始数据集目录**只允许读取**，不得修改、覆盖、移动、删除。
3. **禁止危险 git 操作**：不执行 `git push`、强制重置（`reset --hard`）、清理（`clean`）或删除操作。
4. **不跨项目借用**：不使用其他项目的路径、配置、模型、实验结果或代码。
5. **不造假**：不声称未执行的测试已通过，不伪造实验指标，不用占位数据绕过验收。
6. **严格模式**：缺失城市、坐标不一致、区域数不一致、特征含 NaN/Inf、映射覆盖率异常
   → 直接报明确错误，**不得静默补零并假装成功**。
7. **无泄漏归一化**：归一化参数只在训练城市/训练数据上拟合，再应用到验证/测试；
   禁止用全量数据统计造成信息泄漏。

## 第一阶段范围

- **纳入**：GTG 空间句法（Total Depth / Integration / Connectivity / Choice）、路网拓扑特征提取、
  GTG 拓扑聚合（Metis + TopoAggregator/GATv2）、路段级特征映射到 CRAFT 区域、在 GFA 前与
  CRAFT Population/POI/Road 融合。
- **排除**：GTG 的强化学习、路径规划、轨迹决策、出行偏好学习、成本预测。
- **不做**：不替换 CRAFT Diffusion、不把融合放到 GFA 之后、不引入额外创新模块/对比目标/新损失。
  保持最小、可运行、可验证。

## 关键路径

| 用途 | 路径 |
|------|------|
| CRAFT 只读数据根 | `/root/autodl-tmp/projects/CRAFT/cleared_data` |
| 生成的 norm 流量 | `/root/autodl-tmp/projects/Paper/data/norm_flow` |
| GTG 特征缓存 | `/root/autodl-tmp/projects/Paper/cache/gtg` |
| 融合改动的 CRAFT 副本 | `/root/autodl-tmp/projects/Paper/src/craft_integrated` |
| GTG 特征管线 | `/root/autodl-tmp/projects/Paper/src/gtg_features` |

## 环境注意

- `graph_tool` 与 `torch` 因捆绑的 `libgomp`/`libstdc++` 版本不同**同进程冲突**。
  二者在真实流程从不共存（graph_tool 只在离线构建，torch 只在训练）。
- 跑 graph_tool 相关代码/测试时预加载 conda 库：
  `LD_PRELOAD="/root/miniconda3/lib/libstdc++.so.6 /root/miniconda3/lib/libgomp.so.1"`。

## 暂停条件（遇到才暂停并询问）

- 无法定位 GTG/CRAFT；数据集确实缺失；`Paper/` 内已有冲突性既有工作；
  必须在两种研究意义显著不同的方案间抉择；权限/环境阻塞。

## 变更前自检

- 是否只写在 `Paper/` 内？
- 是否只读了 CRAFT/GTG？
- 归一化是否只用了训练数据拟合？
- 报告的测试/指标是否为真实运行所得？
