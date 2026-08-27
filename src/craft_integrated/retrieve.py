import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, data_list, features, match_month=True):
        """
        :param data_list [{
            'region_idx': int, 不是 region_id, 而是多个城市堆叠的 features 的索引
            'is_weekend': bool,
            'start_hour': int,
            'value': np.ndarray (n, 2, 24)
        }]
        :param features: np.ndarray
        """
        self.idx_to_value = {
            idx: item['value']
            for idx, item in enumerate(data_list)
        }
        self.data_info_df = pd.DataFrame(
            data=[{
                'idx': idx,
                'region_idx': item['region_idx'],
                'month': item['month'],
                'weekday': item['weekday'],
                'is_weekend': item['is_weekend'],
                'start_hour': item['start_hour'],
            } for idx, item in enumerate(data_list)]
        )
        self.features = features
        self.match_month = bool(match_month)

    @staticmethod
    def get_top_k_indices(query, candidates, top_k, metric='cosine'):
        if metric == 'cosine':
            # note: 越大越相似
            similarities = cosine_similarity([query], candidates).flatten()
            top_k_indices = similarities.argsort()[::-1][:top_k]
        elif metric == 'euclidean':
            # note: 越小越相似
            similarities = np.sqrt(np.sum((query - candidates) ** 2, axis=1))
            top_k_indices = similarities.argsort()[:top_k]
        else:
            raise ValueError('similarity metric error')
        return top_k_indices

    def query_by_feature(self, query_info, top_k, metric='cosine'):
        """
        返回 top_k 相似区域的流量平均值
        """
        month = query_info['month']
        weekday = query_info['weekday']
        start_hour = query_info['start_hour']
        feature = query_info['feature']
        temporal_mask = (
            (self.data_info_df['weekday'] == weekday) &
            (self.data_info_df['start_hour'] == start_hour)
        )
        if self.match_month:
            temporal_mask &= self.data_info_df['month'] == month
        candidate_info = self.data_info_df[temporal_mask]
        if candidate_info.empty:
            condition = 'month+weekday+start_hour' if self.match_month else 'weekday+start_hour'
            raise ValueError(
                '严格模式: 检索无候选 '
                f'(condition={condition}, month={month}, weekday={weekday}, start_hour={start_hour})'
            )
        # 取消月份条件后，同一区域可能按月出现多行。top-k 必须针对不同区域，
        # 不能让同一区域因月份多而重复占位；选中区域后再汇总其全部候选月份流量。
        regions = candidate_info['region_idx'].drop_duplicates().tolist()
        candidates = self.features[regions]
        top_idx = self.get_top_k_indices(query=feature, candidates=candidates, top_k=top_k, metric=metric)
        top_region_idx = [regions[idx] for idx in top_idx]
        value_idx = candidate_info[candidate_info['region_idx'].isin(top_region_idx)]['idx'].tolist()
        value = np.concatenate([self.idx_to_value[idx] for idx in value_idx], axis=0)
        reference = np.mean(value,  axis=0)
        return reference
