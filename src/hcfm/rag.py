"""保持第一阶段语义的 Region 级、源城市 train-only RAG。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalRecord:
    city_id: str
    region_id: int
    month: int
    weekday: int
    start_hour: int
    feature: np.ndarray
    flow: np.ndarray


class SourceTrainRegionRetriever:
    """CRAFT 同类的非学习 Region 检索，不实现 Road RAG/重排/新损失。"""

    def __init__(self, records: Sequence[RetrievalRecord], source_cities: Iterable[str], seq_length: int):
        self.source_cities = tuple(sorted(set(source_cities)))
        self.seq_length = int(seq_length)
        if not records or not self.source_cities:
            raise ValueError("RAG source train records/source_cities 不能为空")
        self.records = list(records)
        self.groups: dict[tuple[int, int, int], list[RetrievalRecord]] = {}
        for record in self.records:
            if record.city_id not in self.source_cities:
                raise ValueError("RAG 泄漏: record 包含非源城市")
            if record.flow.shape != (2, self.seq_length) or not np.isfinite(record.flow).all():
                raise ValueError("RAG record flow shape/有限性错误")
            self.groups.setdefault((record.month, record.weekday, record.start_hour), []).append(record)

    @classmethod
    def fit(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_cities: Iterable[str],
        split: str,
        seq_length: int,
    ) -> "SourceTrainRegionRetriever":
        if split != "train":
            raise ValueError("RAG 泄漏防护: 非 train split 禁止构库")
        sources = tuple(sorted(set(source_cities)))
        records = []
        for row in rows:
            if row["city_id"] not in sources or row.get("split", split) != "train":
                raise ValueError("RAG 泄漏防护: 发现目标城市或非 train 记录")
            records.append(RetrievalRecord(
                city_id=str(row["city_id"]), region_id=int(row["region_id"]),
                month=int(row["month"]), weekday=int(row["weekday"]),
                start_hour=int(row["start_hour"]),
                feature=np.asarray(row["feature"], dtype=np.float32),
                flow=np.asarray(row["flow"], dtype=np.float32),
            ))
        return cls(records, sources, seq_length)

    def query_city(
        self,
        features: np.ndarray,
        *,
        month: int,
        weekday: int,
        start_hour: int,
        top_k: int,
        metric: str = "euclidean",
        query_city: str | None = None,
    ) -> np.ndarray:
        """整城逐 Region 独立查询，输出 ``[N,2,T]``。"""

        candidates = self.groups.get((int(month), int(weekday), int(start_hour)), [])
        if query_city is not None:
            # 目标城市本不在 source DB；源城市训练时排除同 Region，延续第一阶段语义。
            candidates_by_region = [
                [r for r in candidates if not (r.city_id == query_city and r.region_id == region)]
                for region in range(len(features))
            ]
        else:
            candidates_by_region = [candidates] * len(features)
        references = []
        for region, region_candidates in enumerate(candidates_by_region):
            if not region_candidates:
                raise LookupError(
                    f"严格模式: RAG 无候选 month={month} weekday={weekday} hour={start_hour} region={region}"
                )
            candidate_features = np.stack([r.feature for r in region_candidates])
            query = np.asarray(features[region], dtype=np.float32)
            if metric == "euclidean":
                score = np.linalg.norm(candidate_features - query[None], axis=1)
                indices = np.argsort(score)[:top_k]
            elif metric == "cosine":
                denom = np.linalg.norm(candidate_features, axis=1) * max(np.linalg.norm(query), 1e-8)
                score = (candidate_features @ query) / np.maximum(denom, 1e-8)
                indices = np.argsort(score)[::-1][:top_k]
            else:
                raise ValueError(f"未知 retrieve metric={metric!r}")
            references.append(np.mean([region_candidates[index].flow for index in indices], axis=0))
        return np.stack(references).astype(np.float32)


def assert_retriever_no_target_leakage(
    retriever: SourceTrainRegionRetriever, target_cities: Iterable[str]
) -> None:
    overlap = set(retriever.source_cities) & set(target_cities)
    if overlap:
        raise ValueError(f"RAG 泄漏: target cities 出现在 source DB: {sorted(overlap)}")
