import numpy as np
import pytest

from retrieve import Retriever


def test_empty_temporal_candidates_raise_actionable_error():
    retriever = Retriever(
        data_list=[{
            "region_idx": 0,
            "month": 11,
            "weekday": 0,
            "is_weekend": False,
            "start_hour": 8,
            "value": np.ones((1, 2, 24)),
        }],
        features=np.ones((1, 4)),
    )
    with pytest.raises(ValueError, match=r"condition=month\+weekday\+start_hour"):
        retriever.query_by_feature(
            {
                "feature": np.ones(4),
                "month": 9,
                "weekday": 0,
                "start_hour": 8,
            },
            top_k=1,
            metric="euclidean",
        )


def test_month_disabled_matches_weekday_hour_and_deduplicates_regions():
    retriever = Retriever(
        data_list=[
            {
                "region_idx": 0,
                "month": month,
                "weekday": 0,
                "is_weekend": False,
                "start_hour": 8,
                "value": np.full((1, 2, 24), float(month)),
            }
            for month in (10, 11)
        ] + [{
            "region_idx": 1,
            "month": 11,
            "weekday": 0,
            "is_weekend": False,
            "start_hour": 8,
            "value": np.full((1, 2, 24), 20.0),
        }],
        features=np.asarray([[0.0, 0.0], [10.0, 10.0]]),
        match_month=False,
    )
    reference = retriever.query_by_feature(
        {
            "feature": np.asarray([0.0, 0.0]),
            "month": 9,
            "weekday": 0,
            "start_hour": 8,
        },
        top_k=1,
        metric="euclidean",
    )
    # top-1 选中 region 0，然后聚合它跨月的两条流量，而不是重复占两个 top-k 名额。
    assert np.allclose(reference, 10.5)
