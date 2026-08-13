"""模型集成测试:
  1. 基线(use_gtg_topology=false) 与原始 CRAFT GTAggregator 结构一致、数值等价。
  2. 融合模型新增且仅新增 gtg_branch/fusion_proj 参数。
  3. 旧 ckpt 可加载进融合模型 (missing ⊆ gtg 新增, unexpected == [])。
  4. 融合前向 smoke: (N, 45+9) -> (N, 128)。
所有测试在 CPU 上运行。
"""
import importlib.util
import os
import sys

import torch
import pytest

import rep_model as new_rep  # Paper/src/craft_integrated (patched)

DEVICE = "cpu"
RAW_DIM = 45
GTG_DIM = 9
REP_DIM = 128


def base_cfg(use_gtg):
    cfg = {
        "device": DEVICE,
        "raw_feature_dim": RAW_DIM,
        "retrieve_metric": "euclidean",
        "rep_dim": REP_DIM,
        "use_sim_loss": True,
        "use_w_loss": True,
        "use_gtg_topology": use_gtg,
    }
    if use_gtg:
        cfg.update({
            "gtg_feature_dim": GTG_DIM,
            "gtg_gat_layers": 4,
            "gtg_gat_heads": 8,
            "gtg_dropout": 0.1,
        })
    return cfg


def _synthetic_graph(n=12, dim=RAW_DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    nodes = torch.randn(n, dim, generator=g)
    # 简单链式无向邻接
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return nodes, edge_index


def test_baseline_keys_no_gtg():
    m = new_rep.GTAggregator(base_cfg(False))
    keys = set(m.state_dict().keys())
    assert not any(k.startswith("gtg_branch") or k.startswith("fusion_proj") for k in keys)


def test_fusion_adds_only_gtg_modules():
    base_keys = set(new_rep.GTAggregator(base_cfg(False)).state_dict().keys())
    fus_keys = set(new_rep.GTAggregator(base_cfg(True)).state_dict().keys())
    extra = fus_keys - base_keys
    assert base_keys.issubset(fus_keys), "融合模型丢失了基线参数键"
    assert all(k.startswith("gtg_branch") or k.startswith("fusion_proj") for k in extra), \
        f"融合模型新增了非 GTG 键: {extra}"
    assert len(extra) > 0


def test_old_ckpt_loads_into_fusion(tmp_path):
    base = new_rep.GTAggregator(base_cfg(False))
    ckpt = os.path.join(tmp_path, "base.pth")
    torch.save(base.state_dict(), ckpt)

    fusion = new_rep.GTAggregator(base_cfg(True))
    missing, unexpected = fusion.load_state_dict(
        torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False
    )
    assert list(unexpected) == [], f"不应有多余键: {unexpected}"
    assert all(k.startswith("gtg_branch") or k.startswith("fusion_proj") for k in missing), \
        f"缺失键应仅为 GTG 新增: {missing}"


def test_fusion_forward_smoke():
    torch.manual_seed(0)
    m = new_rep.GTAggregator(base_cfg(True)).to(DEVICE)
    m.eval()
    craft, edge_index = _synthetic_graph(dim=RAW_DIM)
    gtg = torch.randn(craft.shape[0], GTG_DIM)
    nodes = torch.cat([craft, gtg], dim=1)  # (N, 54)
    with torch.no_grad():
        out = m(nodes, edge_index)
    assert out.shape == (craft.shape[0], REP_DIM)
    assert torch.all(torch.isfinite(out))


def test_baseline_forward_smoke():
    torch.manual_seed(0)
    m = new_rep.GTAggregator(base_cfg(False)).to(DEVICE)
    m.eval()
    nodes, edge_index = _synthetic_graph(dim=RAW_DIM)
    with torch.no_grad():
        out = m(nodes, edge_index)
    assert out.shape == (nodes.shape[0], REP_DIM)
    assert torch.all(torch.isfinite(out))


def test_baseline_numerically_equals_original_craft():
    """加载原始 CRAFT GTAggregator, 复制权重到基线, 相同输入前向应完全一致。"""
    orig_path = "/root/autodl-tmp/projects/CRAFT/rep_model.py"
    # 原始 rep_model 依赖 graph_transformer_pytorch (与 Paper 副本一致, 已在 sys.path)
    spec = importlib.util.spec_from_file_location("orig_rep_model", orig_path)
    orig = importlib.util.module_from_spec(spec)
    sys.modules["orig_rep_model"] = orig
    spec.loader.exec_module(orig)

    cfg = base_cfg(False)
    m_new = new_rep.GTAggregator(cfg).to(DEVICE).eval()
    m_orig = orig.GTAggregator(cfg).to(DEVICE).eval()
    # 权重对齐 (键集应完全相同)
    assert set(m_new.state_dict().keys()) == set(m_orig.state_dict().keys())
    m_orig.load_state_dict(m_new.state_dict())

    nodes, edge_index = _synthetic_graph(dim=RAW_DIM, seed=3)
    with torch.no_grad():
        out_new = m_new(nodes, edge_index)
        out_orig = m_orig(nodes, edge_index)
    assert torch.allclose(out_new, out_orig, atol=1e-6), "基线与原始 CRAFT 数值不一致"
