# 第二阶段真实数据审计

审计时间：2026-08-19（UTC）。本报告只记录服务器上实际读取到的数据，不推断 OSM/道路静态字段包含动态交通量。

## 1. 数据源

- CRAFT：`/root/autodl-tmp/projects/CRAFT/cleared_data`（只读），城市 `chi,dc,toronto,ny`。
- 第一阶段 GTG 缓存：`/root/autodl-tmp/projects/Paper/cache/gtg`。
- GTG 原项目实际路径：`/root/autodl-tmp/projects/GTG-main`（只读），不是旧说明中的 `.../GTG`。
- Paper 归一化宏观流：`data/norm_flow`。

## 2. CRAFT 四城静态图与宏观动态

`directed_roads` 按 `oneway=True` 一条、`oneway=False` 正反两条展开；下表初始审计的 `directed road edges` 统计全部 `target_node[j]==source_node[k]` 候选（包括环形 Road 的 `j==k`），最终训练缓存排除自环后的精确边数见 `STAGE2_IMPLEMENTATION.md`。Region CRS/road 源坐标均为 WGS84 (`EPSG:4326`)，度量相交使用各城 `data_feature.json` 的 UTM。

| city | Region | Region edges | source roads | directed roads | directed road edges | UTM | train 日期/行/Region | test 日期/行/Region |
|---|---:|---:|---:|---:|---:|---:|---|---|
| chi | 73 | 478 | 31,093 | 52,681 | 270,920 | 32616 | 2023-01-01..2023-08-31 / 71,836 / 41 | 2023-09-01..2023-12-31 / 35,949 / 39 |
| dc | 81 | 544 | 39,563 | 70,499 | 363,614 | 32618 | 2023-01-01..2023-08-31 / 41,416 / 31 | 2023-09-01..2023-12-31 / 22,916 / 27 |
| toronto | 60 | 378 | 20,343 | 37,943 | 201,004 | 32617 | 2020-01-01..2020-08-31 / 26,208 / 29 | 2020-09-01..2020-12-28 / 14,915 / 22 |
| ny | 95 | 632 | 45,289 | 68,995 | 292,244 | 32618 | 2023-01-01..2023-08-31 / 231,652 / 52 | 2023-09-01..2023-12-31 / 126,574 / 54 |

四城 `start_hour` 均覆盖 0..23。宏观 CSV 仅列有观测流量的 Region；相对全体 Region 的 train 未观测比例分别为 chi 43.84%、dc 61.73%、toronto 51.67%、ny 45.26%。第二阶段必须用 `region_mask` 表示，而不是把缺行当真实零。现有 `norm_flow` 测试逐项验证 24 步、有限值与范围。

第一阶段 road-to-Region 覆盖：chi/dc 均 0 空 Region；toronto 5/60 空 Region（8.33%）；ny 2/95 空 Region（2.11%）。四城缓存记录的 source road 映射率都是 100%。第二阶段已重新构建 Road 级 9 维缓存与软 `P_struct/B_in/B_out`：chi/dc/toronto/ny 的跨 Region 有向 Road 比例分别为 7.14%、7.06%、8.21%、6.63%；`B_in/B_out` nnz 分别为 3,799/3,799、5,197/5,198、3,224/3,223、4,782/4,784。矩阵按 Road 经过的完整 Region 序列记录中间边界；所有非空 P 行真实复核为和 1。

## 3. 动态道路与轨迹审计

CRAFT 的 `bike_trip.csv` 真实 schema 为：

```text
start_time,end_time,start_lon,start_lat,end_lon,end_lat[,out_of_bound]
```

它只有 OD 端点，没有逐点 GPS、没有 `directed_road_id` 序列、没有 enter/leave time。因而：

- 轨迹 map-matching 率：不可计算；
- road passage count：缺失；
- 道路流量缺失率：100%；
- 微观时间范围：无；
- 与宏观可联合窗口：0。

不能用最近道路连接 OD、最短路猜测或道路长度/等级替代真实通过量。

GTG-main 有北京、成都、西安的 `traj/{train,test}.csv`，schema 为 `traj_id;start_time;rid_list;dur_list`，并有基于 train 轨迹生成的 `dur_mean/speed_mean` 标签。样例 Unix 时间显示北京约 2015 年、成都/西安约 2018 年；城市、Region 划分和时间均不属于 CRAFT 四城。因此它们只能用于独立的道路编码器预训练研究，不能和 CRAFT 四城宏观流量计算跨尺度损失。本阶段实现复用其网络/损失语义，但不把三城动态标签混入 CRAFT 联合样本。

## 4. 时间区、DST 与泄漏

原始文件没有显式 timezone 字段。CRAFT 日期时间为城市本地时间的朴素字符串；在得到上游数据说明前不得擅自当 UTC 或做 DST 折叠。契约要求 manifest 显式声明 IANA timezone 和 DST 策略；当前真实联合数据因此还缺这一元数据。

现有宏观切分是 train 早期日期、test 后期日期；RAG 代码只从源城市 train 构库。Paper 的 norm 统计由各城 train 拟合，但第二阶段跨城市零样本规范更严格：必须由一次实验的源城市 train 联合拟合，不能沿用目标城市各自拟合的 norm 缓存做跨尺度物理损失。

## 5. NaN/Inf、CRS 与口径

- 四城 `road.csv` 统计到 0 个空单元；几何为 WGS84，长度字段为米，度量相交转各城 UTM。
- 第一阶段 GTG Region 缓存已有全有限值测试。
- 宏观 in/out 是 bike OD 滑窗流量，不是道路边界传感器守恒计数。
- 即使未来由 bike 轨迹得到道路 passage count，宏观 OD 区域流与道路边界通过量也可能因滑窗平滑、采样、路径覆盖产生口径差。校准只能用源城市 train 拟合，并报告绝对/相对/逐 Region gap。

## 6. 当前阻塞与可继续工作

已完成：四城静态 Region/Road 图、软 `P_struct`、方向性 `B_in/B_out`、严格 Dataset/manifest、道路对抗编码、双向层次交互、Macro/Micro FM、损失/solver/checkpoint、四城真实缓存测试和手工图算法 smoke。

不能声称完成的真实数据验收：道路流量缺失率、轨迹匹配率、真实 `S(Q_true)≈X_macro_true`、真实整城 HCFM batch/smoke、真实微观指标。解除阻塞需提供 CRAFT 四城中至少一个城市、同一日期窗口的 map-matched directed-road passage count（或带完整 GPS 点的轨迹）以及 timezone/统计口径说明。
