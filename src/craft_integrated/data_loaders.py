import ast
import random
import torch
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from retrieve import Retriever


# ===== 数据路径与 GTG 融合配置 (第一阶段新增) =====
# 由 main/train 在启动时通过 configure(cfg) 注入; 默认值保持原始 CRAFT 行为。
_CRAFT_DATA_ROOT = 'cleared_data'      # CRAFT 只读区域特征/道路根目录
_NORM_FLOW_ROOT = None                  # None 时回退到 _CRAFT_DATA_ROOT (兼容原始)
_USE_GTG_TOPOLOGY = False               # 是否在区域特征后拼接 GTG 拓扑特征
_GTG_CACHE_DIR = None                   # GTG region 级特征缓存目录
_GTG_FEATURE_DIM = None                 # 期望的 GTG 特征维度 (严格校验)


def configure(cfg):
    """从配置注入数据路径与 GTG 融合开关 (只读 CRAFT, 归一化/缓存写在 Paper)。"""
    global _CRAFT_DATA_ROOT, _NORM_FLOW_ROOT, _USE_GTG_TOPOLOGY, _GTG_CACHE_DIR, _GTG_FEATURE_DIM
    _CRAFT_DATA_ROOT = cfg.get('craft_data_root', 'cleared_data')
    _NORM_FLOW_ROOT = cfg.get('norm_flow_root', None)
    _USE_GTG_TOPOLOGY = cfg.get('use_gtg_topology', False)
    _GTG_CACHE_DIR = cfg.get('gtg_cache_dir', None)
    _GTG_FEATURE_DIM = cfg.get('gtg_feature_dim', None)
    if _USE_GTG_TOPOLOGY and _GTG_CACHE_DIR is None:
        raise ValueError('严格模式: use_gtg_topology=True 但未提供 gtg_cache_dir')


def load_norm_flow(city, seq_length, phase):
    data_root = _NORM_FLOW_ROOT if _NORM_FLOW_ROOT is not None else _CRAFT_DATA_ROOT
    data_dir = f'{data_root}/{city}'
    flow_df = pd.read_csv(join(data_dir, f'norm_{phase}_len_{seq_length}.csv'))

    flow_df['in_flow'] = flow_df['in_flow'].apply(lambda x: np.array(ast.literal_eval(x)))
    flow_df['out_flow'] = flow_df['out_flow'].apply(lambda x: np.array(ast.literal_eval(x)))
    flow_df['norm_flow'] = flow_df.apply(lambda row: np.stack([row['in_flow'], row['out_flow']], axis=0), axis=1)
    flow_df['is_weekend'] = flow_df['weekday'].apply(lambda x: x > 4)
    flow_df['date'] = flow_df['date'].apply(lambda x: pd.Timestamp(x).date())
    return flow_df


class FlowDataset(Dataset):
    def __init__(self, data_list):
        """
        :param data_list: [{
            'x': Tensor (2, 24)
            'feature': Tensor (feature_dim,) 区域表征
            'reference': Tensor (2, seq_length) 检索得到的特征
            'start_hour': int
            'weekday': int
            'date': str
            'region_id': int 原始的 region_id 不同城市之间会重复, 仅供测试阶段和真实数据做对应
        }]
        """
        self.data_list = [
            {
                # 输入数据已经归一化到 [0, 1], 训练数据转换到 [-1, 1]
                'x': torch.tensor(2 * item['x'] - 1, dtype=torch.float32),
                'feature': torch.tensor(item['feature'], dtype=torch.float32),
                'reference': torch.tensor(2 * item['reference'] - 1, dtype=torch.float32),
                'start_hour': item['start_hour'],
                'weekday': item['weekday'],
                'month': item['month'],
                # date 和 region_id 是为了让生成数据与真实数据对应起来
                'date': item['date'],
                'region_id': item['region_id']
            }
            for item in data_list
        ]
        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    @staticmethod
    def collator(indices):
        batch = dict()
        batch['x'] = torch.stack([item['x'] for item in indices], dim=0)  # (batch_size, 2, 24)
        batch['feature'] = torch.stack([item['feature'] for item in indices], dim=0)
        batch['reference'] = torch.stack([item['reference'] for item in indices], dim=0)

        batch['start_hour'] = torch.LongTensor([item['start_hour'] for item in indices])
        batch['weekday'] = torch.LongTensor([item['weekday'] for item in indices])
        batch['month'] = torch.LongTensor([item['month'] for item in indices])

        # 用于测试时追踪生成数据对应的 date 和 region
        batch['date'] = [item['date'] for item in indices]
        batch['region_id'] = [item['region_id'] for item in indices]
        return batch


def get_retriever(city_data_list):
    """
    :param city_data_list: [{'feature': np.ndarray, 'flow_df': pd.DataFrame}]
    """
    base_idx = 0
    item_data_list = []
    total_features = []
    for city_dict in city_data_list:
        feature = city_dict['feature']
        flow_df = city_dict['flow_df']
        flow_df['region_idx'] = flow_df['region_id'] + base_idx
        base_idx += len(feature)
        total_features.append(feature),
        groups = flow_df.groupby(by=['region_idx', 'month', 'weekday', 'start_hour'])
        for (region_idx, month, weekday, start_hour), group in groups:
            value = np.stack([row['norm_flow'] for _, row in group.iterrows()], axis=0)
            item_data_list.append({
                'region_idx': region_idx,
                'month': month,
                'weekday': weekday,
                'is_weekend': weekday > 4,
                'start_hour': start_hour,
                'value': value
            })
    total_features = np.concatenate(total_features)
    retriever = Retriever(data_list=item_data_list, features=total_features)
    return retriever


def get_source_train_datasets(cfg, phase):
    exp_dir = cfg['exp_dir']
    src_cities = cfg['src_cities']
    seq_length = cfg['seq_length']
    # 在训练阶段, 使用训练集本身作为检索数据源
    refer_flow_dfs = []
    for city in src_cities:
        flow_df = load_norm_flow(city=city, seq_length=seq_length, phase='train')
        flow_df = flow_df.groupby(by=['region_id', 'weekday', 'start_hour', 'month']).agg({
            'norm_flow': lambda series: np.mean(series.tolist(), axis=0)
        }).reset_index()
        flow_df['refer_flow'] = flow_df.apply(lambda row: row['norm_flow'].flatten(), axis=1)
        flow_df['city'] = city
        refer_flow_dfs.append(flow_df)
    refer_df = pd.concat(refer_flow_dfs)
    refer_group_dict = {
        (month, weekday, start_hour): group
        for (month, weekday, start_hour), group in refer_df.groupby(by=['month', 'weekday', 'start_hour'])
    }
    dataset_list = []
    for city in src_cities:
        data_sample_list = []
        miss_count = 0
        features = np.load(join(exp_dir, f'{city}_rep.npy'))
        flow_df = load_norm_flow(city, seq_length=seq_length, phase=phase)
        for _, row in tqdm(flow_df.iterrows(), total=len(flow_df), desc='load source city samples'):
            # 查找源城市其他 region 的相似样本, 获取训练阶段用的 reference
            candidates = refer_group_dict.get((row['month'], row['weekday'], row['start_hour']))
            candidates = candidates[(candidates['city'] != city) | (candidates['region_id'] != row['region_id'])]
            if len(candidates) == 0:
                # 在训练阶段, 如果其他 region 中没有命中的样本, 则使用本身作为 reference
                reference = row['norm_flow']
                miss_count += 1
            else:
                candidates_flow = np.stack(candidates['refer_flow'].tolist(), axis=0)
                query = row['norm_flow'].flatten()
                distances = np.linalg.norm(candidates_flow - query, axis=1)
                refer_idx = np.argmin(distances)
                reference = candidates.iloc[refer_idx]['norm_flow']
            data_sample_list.append({
                'x': row['norm_flow'],
                'feature': features[int(row['region_id'])],
                'reference': reference,
                'start_hour': row['start_hour'],
                'weekday': row['weekday'],
                'month': row['month'],
                'date': row['date'],
                'region_id': row['region_id']
            })
        print(f'Reference miss {miss_count} in {city} {phase}')
        dataset_list.append(FlowDataset(data_list=data_sample_list))
    return dataset_list


def get_test_dataloader(cfg):
    """
    返回生成阶段的 dataloader
    支持返回多个目标城市的 dataloader
    """
    retrieve_metric = cfg['retrieve_metric']
    top_k = cfg['retrieve_top_k']
    exp_dir = cfg['exp_dir']
    src_cities, trg_cities = cfg['src_cities'], cfg['trg_cities']
    seq_length = cfg['seq_length']

    # 检索数据源是源城市的训练集
    retrieve_data_list = [
        {
            'city': city,
            'feature': np.load(join(exp_dir, f'{city}_rep.npy')),
            'flow_df': load_norm_flow(
                city=city, seq_length=seq_length, phase='train'
            )
        }
        for city in src_cities
    ]

    loaders_dict = dict()
    for trg_city in trg_cities:
        # 确保 retriever 不包含目标城市
        retriever = get_retriever([item for item in retrieve_data_list if item['city'] != trg_city])

        data_list = []
        trg_feature = np.load(join(exp_dir, f'{trg_city}_rep.npy'))

        trg_flow_df = load_norm_flow(
            city=trg_city, seq_length=seq_length, phase='test'
        )

        for _, row in tqdm(trg_flow_df.iterrows(), total=len(trg_flow_df), desc=f'load {trg_city} test flow'):
            feature = trg_feature[row['region_id']]
            reference = retriever.query_by_feature(
                query_info={
                    'feature': feature,
                    'month': row['month'],
                    'weekday': row['weekday'],
                    'start_hour': row['start_hour']
                }, top_k=top_k, metric=retrieve_metric
            )
            data_list.append({
                'x': row['norm_flow'],
                'feature': feature,
                'reference': reference,
                'start_hour': row['start_hour'],
                'weekday': row['weekday'],
                'month': row['month'],
                'date': row['date'],
                'region_id': row['region_id']
            })
        dataset = FlowDataset(data_list=data_list)
        loaders_dict[trg_city] = DataLoader(
            dataset, batch_size=cfg['batch_size'], shuffle=False, collate_fn=FlowDataset.collator
        )
    return loaders_dict


def _load_gtg_region_feature(city, num_regions):
    """加载缓存的 region 级 GTG 拓扑特征 (原始/未归一化, 归一化在模型分支内逐图完成)。
    严格模式: 缺失缓存/维度不符/行数不符/含 NaN·Inf 直接报错。"""
    if _GTG_CACHE_DIR is None:
        raise ValueError('严格模式: 未配置 gtg_cache_dir')
    pth = join(_GTG_CACHE_DIR, f'{city}_gtg_region.npz')
    import os
    if not os.path.exists(pth):
        raise FileNotFoundError(f'严格模式: 缺失 GTG 缓存 {pth}, 请先运行 scripts/build_gtg_features.py')
    data = np.load(pth, allow_pickle=True)
    gtg_feat = data['region_feat'].astype(np.float32)
    if gtg_feat.shape[0] != num_regions:
        raise ValueError(f'严格模式: {city} GTG 区域数 {gtg_feat.shape[0]} != CRAFT 区域数 {num_regions}')
    if _GTG_FEATURE_DIM is not None and gtg_feat.shape[1] != _GTG_FEATURE_DIM:
        raise ValueError(f'严格模式: {city} GTG 特征维度 {gtg_feat.shape[1]} != 期望 {_GTG_FEATURE_DIM}')
    if not np.all(np.isfinite(gtg_feat)):
        raise ValueError(f'严格模式: {city} GTG 特征含 NaN/Inf')
    return gtg_feat


def load_region_feature(city, feature_cols=None):
    poi_type_num = 12
    road_type_num = 8
    data_dir = _CRAFT_DATA_ROOT

    region_df = pd.read_csv(join(data_dir, city, 'grid_region_feature.csv'))
    if feature_cols is None:
        feature_cols = (['population', 'population_density', 'dist_to_center', 'road_num', 'road_length']
                        + [f'poi_num_{k}' for k in range(poi_type_num)]
                        + [f'poi_score_{k}' for k in range(poi_type_num)]
                        + [f'road_num_{k}' for k in range(road_type_num)]
                        + [f'road_length_{k}' for k in range(road_type_num)])
    region_df = region_df.fillna(0)

    # 确保 region_id 是从 0 开始编号的
    assert all(rid == idx for idx, rid in enumerate(region_df['region_id'].tolist())), 'region_id != index'
    node_feature = region_df[feature_cols].to_numpy()

    # ===== 融合模式: 在 CRAFT 45 维特征后拼接 GTG 拓扑特征 =====
    # forward 阶段按 raw_feature_dim=45 切分, 前 45 维走 init_proj, 后 K 维走 GTG 分支。
    if _USE_GTG_TOPOLOGY:
        gtg_feat = _load_gtg_region_feature(city, num_regions=len(region_df))
        node_feature = np.concatenate([node_feature, gtg_feat], axis=1)

    # 加载邻接关系
    rel_df = pd.read_csv(join(data_dir, city, 'grid_region_rel.csv'))
    rel_df = rel_df[rel_df['is_adj'] == 1]
    edge_index = rel_df[['ori', 'des']].to_numpy().T

    return node_feature, edge_index


def load_region_graph(city):
    node_feature, edge_index = load_region_feature(city=city)
    node_feature = torch.tensor(node_feature, dtype=torch.float32)
    edge_index = torch.LongTensor(edge_index)

    flow_df = load_norm_flow(city=city, seq_length=24, phase='train')
    flow_df = flow_df[flow_df['start_hour'] == 0]
    groups = flow_df.groupby('region_id')
    region_values = {
        region_id: np.mean(group['norm_flow'].tolist(), axis=0).flatten()
        for region_id, group in groups
    }
    values = np.stack([
        region_values.get(region_id, np.zeros(48))
        for region_id in range(len(node_feature))
    ])
    values = torch.tensor(values, dtype=torch.float32)
    graph = Data(x=node_feature, edge_index=edge_index, value=values, city=city)
    return graph


def get_graph_datasets(src_cities, trg_cities):
    src_graphs = [load_region_graph(city=src_city) for src_city in src_cities]
    trg_graphs = [load_region_graph(city=trg_city) for trg_city in trg_cities]

    return {
        'src_graphs': src_graphs,
        'trg_graphs': trg_graphs
    }
