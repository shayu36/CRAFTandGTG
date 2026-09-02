# 第一阶段三层静态图测试报告

## 执行环境

- 工作目录：`/root/autodl-tmp/projects/Paper`
- Python：`3.10.20`
- PyTorch：`2.0.1+cu118`
- 设备：`cpu`
- PyG 原生扩展存在 ABI warning，测试使用仓库 `pyg_compat` 后备实现。

## START Road v2 特征与缓存

实际执行的三座城市 v2 预处理命令为等价 Python 调用：

```bash
PYTHONPATH=src python - <<'PY'
from static_hierarchy.preprocessing import build_city_static_hierarchy
from static_hierarchy.data import save_city_static_hierarchy
for city in ['beijing', 'chengdushi', 'xianshi']:
    hierarchy = build_city_static_hierarchy(
        city, 'data/gtg_craft', syntax_cache_dir='cache/gtg',
        local_size=50, empty_region_error_ratio=0.2,
        road_feature_mode='start_static', maxspeed_unit='km/h',
    )
    save_city_static_hierarchy(hierarchy, 'cache/static_hierarchy_start_v2')
PY
```

结果：三座城市均成功生成 `three-layer-start-road-v2` cache：

| city | `region_x` | `road_x` | `syntax_x` | 空 Region 比例 |
|---|---:|---:|---:|---:|
| `beijing` | `[81,45]` | `[14685,33]` | `[293,5]` | 0 |
| `chengdushi` | `[23,45]` | `[3514,33]` | `[70,5]` | 0 |
| `xianshi` | `[26,45]` | `[4147,33]` | `[82,5]` | 0 |

v2 metadata 记录完整 33 维列名、`maxspeed_unit: km/h`、`lanes/maxspeed` 缺失计数与比例。旧 `cache/static_hierarchy/` 未被覆盖。

## v2 加载、前向和反向

实际执行 CPU smoke 命令为：

```bash
PYTHONPATH=src:src/craft_integrated python - <<'PY'
import torch
from static_hierarchy.data import load_city_static_hierarchy
from static_hierarchy.model import ThreeLayerStaticEncoder
cfg = {
    'static_structure_mode': 'three_layer', 'road_feature_mode': 'start_static',
    'road_feature_dim': 33, 'rep_dim': 16, 'road_gat_layers': 1,
    'road_gat_heads': 2, 'road_dropout': 0.0, 'syntax_gat_layers': 1,
    'syntax_gat_heads': 2, 'syntax_dropout': 0.0,
}
model = ThreeLayerStaticEncoder(cfg).train()
for city in ['beijing', 'chengdushi', 'xianshi']:
    hierarchy = load_city_static_hierarchy(
        'cache/static_hierarchy_start_v2', city,
        expected_feature_version='three-layer-start-road-v2',
    )
    outputs = model(hierarchy, return_intermediates=True)
    assert outputs['region_rep'].shape == (hierarchy.num_regions, 16)
    assert all(torch.isfinite(value).all() for value in outputs.values())
    model.zero_grad(set_to_none=True)
    outputs['region_rep'].sum().backward()
PY
```

结果：三座城市的 `road_h`、`road_to_syntax_h`、`syntax_h`、`syntax_to_region_h`、`region_rep` shape 均正确且 finite；Road encoder、Syntax encoder、Region init、Region fusion 和 Region GraphTransformer 均获得非空有限梯度。三城共用同一个编码器实例。

## 自动化测试

实际执行：

```bash
pytest -q tests/test_start_road_features.py tests/test_static_hierarchy.py tests/test_stage1_training_protocol.py
```

结果：`26 passed, 5 warnings`。

覆盖内容包括：

- START Road `[M,33]` schema 与固定列顺序；
- Road type、长度归一化、lanes/maxspeed unknown bucket；
- `mph` 转 `km/h`；
- 有向入度/出度在自环之前计算；
- 非法 `road_type_id` 严格报错；
- v1 cache 不会被当作 v2 读取；
- 旧三层前向、跨层算子、target 不读取动态流量；
- 多 Source TFA/CCA 等权协议与 cosine CCA。

随后合并执行模型集成与 data loader 回归：

```bash
pytest -q tests/test_start_road_features.py tests/test_static_hierarchy.py \
  tests/test_stage1_training_protocol.py tests/test_model_integration.py \
  tests/test_data_loaders.py
```

结果：`43 passed, 6 warnings`。

语法和空白检查：

```bash
python -m compileall -q src/static_hierarchy src/craft_integrated scripts/build_static_hierarchy.py scripts/run_stage1_static.py
git diff --check
```

结果：命令成功。

## 未运行项目

本次没有运行 RAG/Retriever、Diffusion、Flow Matching、HCFM、Stage 2、生成流程或消融实验，也没有执行完整研究训练。

## 环境限制

当前环境的 `graph_tool` 存在 `libgomp` ABI 错误，无法用于缺失 GTG cache 的离线重算；本次三座城市均使用已存在且已对齐的 GTG Road cache，因此不影响 v2 预处理和测试。
