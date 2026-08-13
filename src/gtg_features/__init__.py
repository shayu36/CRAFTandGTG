"""GTG 拓扑特征预处理管线 (第一阶段)。

在 CRAFT 自身的 road.csv 上, 忠实移植 GTG 的算法, 产出 region 级拓扑/空间句法特征,
供 CRAFT GFA 之前融合使用。不含 GTG 的强化学习/成本预测/出行偏好。

模块:
  dual_graph     -- 构造 GTG 风格道路对偶图 (移植自 GTG prepare.gen_edge_data)
  space_syntax   -- Total Depth / Integration / Connectivity / Choice (graph_tool)
  partition      -- Metis 图分区 (移植自 GTG dataloader.metis_cluster)
  road_to_region -- road 级特征长度加权映射到 CRAFT region
  pipeline       -- 编排 + 缓存 + 元数据 + 严格模式校验
"""
