import numpy as np
import ot
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from pyg_compat import GATv2Conv
from torch.autograd import Function

try:
    from static_hierarchy.model import ThreeLayerStaticEncoder
    from static_hierarchy.contracts import CityStaticHierarchy
except ModuleNotFoundError:  # 允许从 src/craft_integrated 直接执行旧入口
    import pathlib
    import sys
    _SRC_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)
    from static_hierarchy.model import ThreeLayerStaticEncoder
    from static_hierarchy.contracts import CityStaticHierarchy

from graph_transformer_pytorch import GraphTransformer


def calc_cosine_similarity_matrix(x1, x2):
    x1 = x1.unsqueeze(1)  # (N, 1, d)
    x2 = x2.unsqueeze(0)  # (1, M, d)
    cosine_mx = F.cosine_similarity(x1, x2, dim=2)
    return cosine_mx


def cross_sim_loss(emb1, value1, emb2, value2):
    # 两个集合之间的相似度对齐损失
    m1, m2 = len(emb1), len(emb2)
    emb1 = emb1.unsqueeze(1).repeat(1, m2, 1)
    value1 = value1.unsqueeze(1).repeat(1, m2, 1)
    emb2 = emb2.unsqueeze(0).repeat(m1, 1, 1)
    value2 = value2.unsqueeze(0).repeat(m1, 1, 1)
    emb_sim = F.cosine_similarity(emb1, emb2, dim=2)
    dist = torch.sum((value1 - value2) ** 2, dim=2).sqrt()
    min_dist, max_dist = torch.min(dist), torch.max(dist)
    norm_dist = 2 * (dist - min_dist) / (max_dist - min_dist) - 1
    value_sim = -norm_dist
    return F.mse_loss(emb_sim, value_sim)


def self_sim_loss(embeddings, values, metric='euclidean'):
    if metric == 'cosine':
        n = len(embeddings)
        emb1 = embeddings.unsqueeze(1).repeat(1, n, 1)
        value1 = values.unsqueeze(1).repeat(1, n, 1)
        emb2 = embeddings.unsqueeze(0).repeat(n, 1, 1)
        value2 = values.unsqueeze(0).repeat(n, 1, 1)
        emb_sim = F.cosine_similarity(emb1, emb2, dim=2)
        dist = torch.sum((value1 - value2) ** 2, dim=2).sqrt()
        min_dist, max_dist = torch.min(dist), torch.max(dist)
        norm_dist = 2 * (dist - min_dist) / (max_dist - min_dist) - 1
        value_sim = -norm_dist
        loss = F.mse_loss(emb_sim, value_sim)
    elif metric == 'euclidean':
        # eps = 1e-8
        # norm_emb = embeddings / torch.clamp(torch.norm(embeddings, dim=-1, keepdim=True), min=eps)
        # norm_value = values / torch.clamp(torch.norm(values, dim=-1, keepdim=True), min=eps)
        emb_dist = torch.cdist(embeddings, embeddings, p=2)
        emb_dist = (emb_dist - emb_dist.min()) / (emb_dist.max() - emb_dist.min())
        value_dist = torch.cdist(values, values, p=2)
        value_dist = (value_dist - value_dist.min()) / (value_dist.max() - value_dist.min())
        loss = F.mse_loss(emb_dist, value_dist)
    else:
        raise ValueError('ERROR metric')

    return loss


def wasserstein_loss(src_emb, trg_emb, metric, src_marginals=None, trg_marginals=None):
    """
    借助 POT 求解器计算 Wasserstein 损失 (精确计算)。

    ``src_marginals``/``trg_marginals`` 可显式指定 OT 边际；未提供时保持
    原有的样本等质量行为。边际必须是有限、非负且总质量为 1，避免城市
    Region 数量通过隐式样本质量改变 CCA 权重。
    """
    if src_emb.ndim != 2 or trg_emb.ndim != 2 or len(src_emb) == 0 or len(trg_emb) == 0:
        raise ValueError('严格模式: Wasserstein 输入必须为非空二维表征')
    if not torch.isfinite(src_emb).all() or not torch.isfinite(trg_emb).all():
        raise ValueError('严格模式: Wasserstein 输入含 NaN/Inf')

    def _marginals(values, expected_len, name):
        if values is None:
            result = np.ones(expected_len, dtype=np.float64) / expected_len
        else:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            result = np.asarray(values, dtype=np.float64)
            if result.shape != (expected_len,):
                raise ValueError(f'严格模式: {name} shape 错误，期望 [{expected_len}]')
            if not np.isfinite(result).all() or (result < 0).any():
                raise ValueError(f'严格模式: {name} 必须为有限非负值')
            if not np.isclose(result.sum(), 1.0, atol=1e-6, rtol=1e-6):
                raise ValueError(f'严格模式: {name} 总质量必须为 1，实得 {result.sum()}')
        return result

    wa = _marginals(src_marginals, len(src_emb), 'src_marginals')
    wb = _marginals(trg_marginals, len(trg_emb), 'trg_marginals')
    if metric == 'euclidean':
        eps = 1e-8
        src_emb = src_emb / torch.clamp(torch.norm(src_emb, dim=-1, keepdim=True), min=eps)
        trg_emb = trg_emb / torch.clamp(torch.norm(trg_emb, dim=-1, keepdim=True), min=eps)
        dist = torch.cdist(src_emb, trg_emb, p=2)
        cost = dist.detach().cpu().numpy()
    elif metric == 'cosine':
        cosine_sim = calc_cosine_similarity_matrix(src_emb, trg_emb)
        dist = 1. - cosine_sim
        cost = dist.detach().cpu().numpy()
    else:
        raise ValueError(f'严格模式: 不支持的 Wasserstein metric={metric!r}')
    if not np.isclose(wa.sum(), wb.sum(), atol=1e-6, rtol=1e-6):
        raise ValueError('严格模式: Source/Target OT 边际总质量不一致')
    transition = ot.emd(a=wa, b=wb, M=cost)
    transition = torch.tensor(transition, dtype=torch.float32).to(src_emb.device)
    loss = torch.sum(transition * dist)
    return loss


class GTAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.get('device', 'cpu')
        self.raw_feature_dim = cfg.get('raw_feature_dim', 45)
        self.retrieve_metric = cfg.get('retrieve_metric', 'euclidean')
        self.rep_dim = cfg['rep_dim']

        # note
        self.use_sim_loss = cfg.get('use_sim_loss', True)
        self.use_w_loss = cfg.get('use_w_loss', True)
        assert self.use_sim_loss or self.use_w_loss  # 不能同时为 False

        # self.init_proj = nn.Sequential(
        #     nn.BatchNorm1d(self.raw_feature_dim, affine=False, track_running_stats=False),
        #     nn.Linear(self.raw_feature_dim, self.rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.rep_dim, self.rep_dim)
        # )
        self.static_structure_mode = cfg.get(
            'static_structure_mode',
            'flat_gtg_region' if cfg.get('use_gtg_topology', False) else 'craft_only',
        )
        if self.static_structure_mode not in {'craft_only', 'flat_gtg_region', 'three_layer'}:
            raise ValueError(f'未知 static_structure_mode={self.static_structure_mode!r}')
        if self.static_structure_mode == 'three_layer':
            self.cca_metric = cfg.get('cca_metric', 'cosine')
            if self.cca_metric != 'cosine':
                raise ValueError('three_layer 模式的 CCA metric 必须为 cosine (1-cosine cost)')
        else:
            self.cca_metric = self.retrieve_metric
        if self.static_structure_mode == 'three_layer':
            if cfg.get('use_gtg_topology', False):
                raise ValueError('three_layer mode and flat GTG region fusion cannot be enabled together')
            self.three_layer_encoder = ThreeLayerStaticEncoder(cfg)
        else:
            self.init_proj = FeatureInitLayer(raw_feature_dim=self.raw_feature_dim, rep_dim=self.rep_dim)
            self.gnn = GraphTransformer(
                dim=self.rep_dim,
                depth=3,
                heads=4,
                dim_head=64,
                with_feedforwards=True,
                rel_pos_emb=False,
                accept_adjacency_matrix=True
            )

        # ===== GTG 拓扑融合分支 (第一阶段新增) =====
        # use_gtg_topology=False 时结构与原 CRAFT 逐字节等价, 旧 ckpt 完全可加载。
        # use_gtg_topology=True 时: 输入节点特征为 [craft(raw_feature_dim) | gtg(gtg_feature_dim)],
        #   craft 部分走 init_proj, gtg 部分走 GTGTopoBranch, 二者在 GFA(GraphTransformer) 之前融合。
        self.use_gtg_topology = cfg.get('use_gtg_topology', False)
        if self.static_structure_mode == 'three_layer':
            self.use_gtg_topology = False
        if self.use_gtg_topology:
            self.gtg_feature_dim = cfg['gtg_feature_dim']
            self.gtg_branch = GTGTopoBranch(
                gtg_dim=self.gtg_feature_dim,
                rep_dim=self.rep_dim,
                num_layers=cfg.get('gtg_gat_layers', 4),
                heads=cfg.get('gtg_gat_heads', 8),
                dropout=cfg.get('gtg_dropout', 0.1),
                device=self.device,
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(2 * self.rep_dim, self.rep_dim),
                nn.ReLU(),
                nn.Linear(self.rep_dim, self.rep_dim),
            )

    def forward(self, nodes, edge_index=None):
        if self.static_structure_mode == 'three_layer':
            if not isinstance(nodes, CityStaticHierarchy):
                raise TypeError('three_layer 模式 forward 需要 CityStaticHierarchy')
            return self.three_layer_encoder(nodes)
        if edge_index is None:
            raise TypeError('craft_only/flat_gtg_region 模式 forward 需要 edge_index')
        nodes = nodes.to(self.device)
        edge_index = edge_index.to(self.device)
        num_nodes = nodes.shape[0]
        adj_mat = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(self.device)
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            adj_mat[src, dst] = 1

        if self.use_gtg_topology:
            # 拆分 CRAFT 原始特征与 GTG 拓扑特征
            craft_feat = nodes[:, :self.raw_feature_dim]
            gtg_feat = nodes[:, self.raw_feature_dim:]
            assert gtg_feat.shape[1] == self.gtg_feature_dim, \
                f'GTG 特征维度不符: 期望 {self.gtg_feature_dim}, 实得 {gtg_feat.shape[1]}'
            h_craft = self.init_proj(craft_feat)                       # (N, rep_dim)
            h_gtg = self.gtg_branch(gtg_feat, edge_index)              # (N, rep_dim)
            fused = self.fusion_proj(torch.cat([h_craft, h_gtg], dim=-1))  # (N, rep_dim)
            nodes = fused
        else:
            nodes = self.init_proj(nodes)

        nodes, _ = self.gnn(nodes=nodes.unsqueeze(0), adj_mat=adj_mat.unsqueeze(0))
        nodes = nodes.squeeze()
        return nodes

    def encode_graph(self, graph):
        """编码旧 CRAFT graph 或三层 CityStaticHierarchy。"""
        if self.static_structure_mode == 'three_layer':
            hierarchy = graph if isinstance(graph, CityStaticHierarchy) else getattr(graph, 'static_hierarchy', None)
            if hierarchy is None:
                raise TypeError('three_layer 模式 graph 缺少 CityStaticHierarchy')
            return self.three_layer_encoder(hierarchy)
        return self.forward(graph.x, graph.edge_index)

    def get_multi_graph_embs(self, graphs):
        src_emb = []
        for graph in graphs:
            reps = self.encode_graph(graph)
            src_emb.append(reps)
        src_emb = torch.cat(src_emb, dim=0)
        return src_emb

    def calc_contrast_loss(self, batch):
        pos_neg_samples = batch['pos_neg_samples']
        src_graphs = batch['src_graphs']
        trg_graphs = batch['trg_graphs']
        src_emb = self.get_multi_graph_embs(src_graphs)
        sim = F.cosine_similarity(src_emb.unsqueeze(1), src_emb.unsqueeze(0), dim=2)

        k = 20
        temp = 0.1
        sim_loss = torch.tensor(0.0).to(self.device)
        for item in pos_neg_samples:
            idx = item['idx']
            pos_idx = np.random.choice(item['pos'])
            neg_indices = np.random.choice(item['neg'], k, replace=True if k < len(item['neg']) else False).tolist()
            indices = [pos_idx] + neg_indices
            scores = torch.softmax(sim[idx][indices] / temp, dim=-1)
            label = torch.tensor(0).to(self.device)
            sim_loss += F.cross_entropy(scores, label)
        sim_loss = sim_loss / len(pos_neg_samples)

        # 目标城市
        trg_emb = self.get_multi_graph_embs(trg_graphs)
        w_loss = wasserstein_loss(src_emb, trg_emb, metric='cosine')

        total_loss = sim_loss + w_loss
        loss_items = {
            'sim_loss': sim_loss.item(),
            'w_loss': w_loss.item(),
        }
        return total_loss, loss_items

    def calc_loss(self, batch):
        src_graphs = batch['src_graphs']
        trg_graphs = batch['trg_graphs']

        if self.static_structure_mode == 'three_layer':
            return self._calc_three_layer_loss(src_graphs, trg_graphs)

        src_emb, src_value = [], []
        # 源城市
        for graph in src_graphs:
            reps = self.encode_graph(graph)
            # 仅有正值训练窗口的区域参与流量-表征监督；完整区域图仍
            # 参与 GNN 消息传播，不对无监督区域伪造零流量。
            value_region_ids = getattr(graph, 'value_region_ids', None)
            if value_region_ids is not None:
                value_region_ids = value_region_ids.to(self.device)
                src_emb.append(reps[value_region_ids])
            else:
                src_emb.append(reps)
            src_value.append(graph.value.to(self.device))
        src_emb = torch.cat(src_emb, dim=0)
        src_value = torch.cat(src_value, dim=0)

        # 源城市的流量-表征对齐损失
        sim_loss = self_sim_loss(src_emb, src_value)

        # 目标城市
        trg_emb = []
        for graph in trg_graphs:
            reps = self.encode_graph(graph)
            trg_emb.append(reps)
        trg_emb = torch.cat(trg_emb)

        w_loss = wasserstein_loss(src_emb, trg_emb, metric=self.retrieve_metric)

        # total_loss = sim_loss + w_loss

        total_loss = torch.tensor(0.0).to(self.device)
        if self.use_sim_loss:
            total_loss += sim_loss
        if self.use_w_loss:
            total_loss += w_loss
        loss_items = {
            'sim_loss': sim_loss.item(),
            'w_loss': w_loss.item(),
        }
        return total_loss, loss_items

    def _calc_three_layer_loss(self, src_graphs, trg_graphs):
        """三层模式的多 Source TFA/CCA 协议。

        TFA 在每个 Source 城市内部、仅对有动态标签的 Region 计算，最后按
        城市等权平均；CCA 使用每个 Source 城市的完整静态 Region 表征，且
        每座城市的 OT 总质量固定为 ``1 / num_source_cities``。
        """

        if not src_graphs:
            raise ValueError('严格模式: three_layer TFA/CCA 至少需要一个 Source 城市')
        if not trg_graphs:
            raise ValueError('严格模式: three_layer CCA 至少需要一个 Target 城市')

        full_source_reps = []
        source_marginals = []
        city_tfa_losses = []
        num_sources = len(src_graphs)

        def _checked_reps(graph, role):
            reps = self.encode_graph(graph)
            city = getattr(graph, "city", role)
            if not isinstance(reps, torch.Tensor):
                raise TypeError(f'严格模式: {city} {role} 表征必须为 torch.Tensor')
            if reps.ndim != 2 or reps.shape[0] == 0 or reps.shape[1] != self.rep_dim:
                raise ValueError(
                    f'严格模式: {city} {role} 表征 shape 错误，'
                    f'期望 [N,{self.rep_dim}]，实得 {tuple(reps.shape)}'
                )
            if not torch.isfinite(reps).all():
                raise ValueError(f'严格模式: {city} {role} 表征含 NaN/Inf')
            return reps

        for graph in src_graphs:
            reps = _checked_reps(graph, "Source")
            full_source_reps.append(reps)

            value = getattr(graph, 'value', None)
            value_region_ids = getattr(graph, 'value_region_ids', None)
            if value is None or value_region_ids is None:
                raise ValueError(
                    f'严格模式: {getattr(graph, "city", "source")} 缺少 TFA 所需动态 Region value'
                )
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(
                    f'严格模式: {getattr(graph, "city", "source")} value 必须为浮点 torch.Tensor'
                )
            if not isinstance(value_region_ids, torch.Tensor):
                raise TypeError(
                    f'严格模式: {getattr(graph, "city", "source")} value_region_ids 必须为 torch.Tensor'
                )
            value_region_ids = value_region_ids.to(reps.device)
            value = value.to(reps.device)
            if value_region_ids.dtype != torch.long or value_region_ids.ndim != 1:
                raise ValueError(f'严格模式: {getattr(graph, "city", "source")} value_region_ids dtype/shape 错误')
            if (
                value.ndim != 2 or value.shape[1] != 48
                or value_region_ids.shape[0] != value.shape[0]
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f'严格模式: {getattr(graph, "city", "source")} value ID/标签 shape 不一致，'
                    '期望 value 为 [num_active_regions,48]'
                )
            if value_region_ids.numel() == 0:
                raise ValueError(f'严格模式: {getattr(graph, "city", "source")} 没有可用于 TFA 的 Region')
            if int(value_region_ids.min()) < 0 or int(value_region_ids.max()) >= reps.shape[0]:
                raise ValueError(f'严格模式: {getattr(graph, "city", "source")} value_region_ids 越界')
            if torch.unique(value_region_ids).shape[0] != value_region_ids.shape[0]:
                raise ValueError(f'严格模式: {getattr(graph, "city", "source")} value_region_ids 重复')
            city_tfa = self_sim_loss(reps[value_region_ids], value)
            if not torch.isfinite(city_tfa):
                raise FloatingPointError(f'严格模式: {getattr(graph, "city", "source")} TFA loss 为 NaN/Inf')
            city_tfa_losses.append(city_tfa)

            city_mass = 1.0 / (num_sources * reps.shape[0])
            source_marginals.append(torch.full(
                (reps.shape[0],), city_mass, dtype=reps.dtype, device=reps.device
            ))

        full_source_rep = torch.cat(full_source_reps, dim=0)
        source_marginal = torch.cat(source_marginals, dim=0)
        sim_loss = torch.stack(city_tfa_losses).mean()

        full_target_reps = []
        for graph in trg_graphs:
            full_target_reps.append(_checked_reps(graph, "Target"))
        full_target_rep = torch.cat(full_target_reps, dim=0)

        w_loss = wasserstein_loss(
            full_source_rep,
            full_target_rep,
            metric=self.cca_metric,
            src_marginals=source_marginal,
        )

        total_loss = torch.zeros((), dtype=sim_loss.dtype, device=sim_loss.device)
        if self.use_sim_loss:
            total_loss = total_loss + sim_loss
        if self.use_w_loss:
            total_loss = total_loss + w_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError('严格模式: three_layer TFA/CCA loss 为 NaN/Inf')
        return total_loss, {
            'sim_loss': sim_loss.item(),
            'w_loss': w_loss.item(),
        }


class GTGTopoBranch(nn.Module):
    """GTG 拓扑聚合分支 (第一阶段新增)。

    忠实移植 GTG-main/models.py: TopoAggregator 的核心聚合机制:
      proj(Linear) -> num_layers x GATv2Conv(concat=False, heads) 残差 + relu。
    差异说明:
      - GTG 原实现在 road 级, 输入含离散属性 embedding 与 4 维边属性(length/dist/angle/bet);
        本阶段作用于 CRAFT region 邻接图, 输入为已池化的 region 级空间句法/拓扑连续特征,
        故不再重复注入 road 级离散/边属性 (它们已在离线空间句法阶段被消费)。
      - 采用与 CRAFT FeatureInitLayer 一致的逐图(逐城市)标准化: 仅用该城市自身节点统计,
        不跨 train/test 集合汇聚参数, 因而无信息泄漏 (与 CRAFT 既有方法一致)。
    """
    def __init__(self, gtg_dim, rep_dim, num_layers, heads, dropout, device):
        super().__init__()
        self.device = device
        self.eps = 1e-5
        self.proj = nn.Linear(gtg_dim, rep_dim)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=rep_dim, out_channels=rep_dim,
                    heads=heads, concat=False, dropout=dropout,
                )
            )

    def forward(self, gtg_feat, edge_index):
        x = gtg_feat.to(self.device)
        edge_index = edge_index.to(self.device)
        # 逐图标准化 (对齐 FeatureInitLayer)
        mean = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.proj(x)
        for layer in self.gnn_layers:
            # 残差连接 (GTG TopoAggregator 同款)
            x = layer(x, edge_index) + x
            x = torch.relu(x)
        return x


class FeaturePreLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = 1e-8

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        scale = 1. / (torch.std(x, dim=0, keepdim=True) + self.eps)
        x = (x - mean) * scale
        if self.training:
            pass
        else:
            return x


class FeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super(FeatureAggregator, self).__init__()
        self.device = cfg['device']
        self.raw_feature_dim = cfg['raw_feature_dim']
        self.rep_dim = cfg['rep_dim']
        self.gat_layers = cfg['gat_layers']
        self.gat_heads = cfg['gat_heads']
        self.dropout = cfg['dropout']
        # self.pretrain_dom_num = cfg['pretrain_dom_num']

        self.proj = nn.Linear(self.raw_feature_dim, self.rep_dim)
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.gat_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=self.rep_dim, out_channels=self.rep_dim,
                    heads=self.gat_heads, concat=False, dropout=self.dropout
                )
            )

        # 特征重构
        self.decoder = nn.Sequential(
            nn.Linear(self.rep_dim, self.raw_feature_dim),
            nn.GELU(),
            nn.Linear(self.raw_feature_dim, self.raw_feature_dim)
        )

        # 域判别器
        self.dom_classifier = DomClassifier(cfg, use_grl=True)

    def forward(self, nodes, edges):
        nodes, edges = nodes.to(self.device), edges.to(self.device)
        nodes = self.proj(nodes)
        for layer in self.gnn_layers:
            # 残差连接
            nodes = layer(nodes, edges) + nodes
            nodes = torch.relu(nodes)
        return nodes

    def reconstruct(self, reps):
        return self.decoder(reps)

    def dom_disc(self, reps):
        return self.dom_classifier(reps)

    @staticmethod
    def _mmd_loss(rep_src, rep_trg):
        def gaussian_kernel(x, y, sigma):
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            return torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * sigma ** 2))

        k_ss = gaussian_kernel(rep_src, rep_src, 1.0)
        k_tt = gaussian_kernel(rep_trg, rep_trg, 1.0)
        k_st = gaussian_kernel(rep_src, rep_trg, 1.0)
        mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()

        return mmd

    @staticmethod
    def _coral_loss(rep_src, rep_trg):
        # CORAL Loss
        d = rep_src.data.shape[1]
        ns, nt = rep_src.data.shape[0], rep_trg.data.shape[0]
        # source covariance
        xm = torch.mean(rep_src, 0, keepdim=True) - rep_src
        xc = xm.t() @ xm / (ns - 1)
        # target covariance
        xmt = torch.mean(rep_trg, 0, keepdim=True) - rep_trg
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        coral = torch.mul((xc - xct), (xc - xct))
        coral = torch.sum(coral) / (4 * d * d)
        return coral

    def _align_loss(self, rep_src: torch.Tensor, rep_trg: torch.Tensor):
        # TODO: kernel method
        mmd = self._mmd_loss(rep_src, rep_trg)
        coral = self._coral_loss(rep_src, rep_trg)
        align_loss = mmd + coral
        return align_loss


    def calc_supervise_loss(self, batch):
        src_graphs = batch['src_graphs']
        trg_graphs = batch['trg_graphs']
        emb_list, value_list, dom_label_list = [], [], []

        # 源城市
        for graph in src_graphs:
            node_feature, edge_index = graph.x.to(self.device), graph.edge_index.to(self.device)
            reps = self.forward(node_feature, edge_index)
            emb_list.append(reps)
            value = graph.value.to(self.device)
            value_list.append(value)
            dom_label = torch.full(
                size=(node_feature.shape[0],), fill_value=graph.dom_label, dtype=torch.long
            ).to(self.device)
            dom_label_list.append(dom_label)
        # 源城市之间的相似度对齐损失
        sim_loss = self.calc_sim_align_loss(emb_list, value_list)

        # 目标城市
        for graph in trg_graphs:
            node_feature, edge_index = graph.x.to(self.device), graph.edge_index.to(self.device)
            reps = self.forward(node_feature, edge_index)
            emb_list.append(reps)
            dom_label = torch.full(
                size=(node_feature.shape[0],), fill_value=graph.dom_label, dtype=torch.long
            ).to(self.device)
            dom_label_list.append(dom_label)
        total_emb = torch.cat(emb_list, dim=0)
        total_dom_label = torch.cat(dom_label_list, dim=0)
        total_dom_pred = self.dom_disc(total_emb)
        adv_loss = F.cross_entropy(total_dom_pred, total_dom_label)

        total_loss = sim_loss + adv_loss
        loss_items = {
            'sim_loss': sim_loss.item(),
            'adv_loss': adv_loss.item(),
        }
        return total_loss, loss_items

    def calc_loss(self, batch):
        """
        尝试: 不使用对抗训练, 仅对源城市流量做对齐
        """
        graphs = batch['src_graphs']
        embeddings, values = [], []
        for graph in graphs:
            value = graph.value.to(self.device)
            node_feature, edge_index = graph.x.to(self.device), graph.edge_index.to(self.device)
            node_feature = self.forward(node_feature, edge_index)
            embeddings.append(node_feature)
            values.append(value)
        embeddings = torch.cat(embeddings, dim=0)
        values = torch.cat(values, dim=0)
        sim_loss = self.self_sim_loss(embeddings=embeddings, values=values)
        return sim_loss


class GradReverseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 添加系数

    def forward(self, x):
        return GradReverseFunc.apply(x)


class GradReverseFunc(Function):
    @ staticmethod
    def forward(ctx, x, **kwargs: None):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -1.0 * grad_output, None


class DomClassifier(nn.Module):
    def __init__(self, cfg, use_grl: bool):
        super(DomClassifier, self).__init__()
        self.dom_clf = nn.Sequential(
            nn.Linear(cfg['rep_dim'], cfg['rep_dim']),
            nn.BatchNorm1d(cfg['rep_dim']),
            nn.ReLU(),
            nn.Linear(cfg['rep_dim'], cfg['pretrain_dom_num']),
        )
        self.use_grl = use_grl

    def forward(self, x):
        if self.use_grl:
            x = GradReverseFunc.apply(x)
        return self.dom_clf(x).squeeze()


class FeatureInitLayer(nn.Module):
    def __init__(self, raw_feature_dim, rep_dim):
        super().__init__()
        self.raw_feature_dim = raw_feature_dim
        self.rep_dim = rep_dim
        self.eps = 1e-5
        self.init_proj = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(self.raw_feature_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.rep_dim),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.init_proj(x)
