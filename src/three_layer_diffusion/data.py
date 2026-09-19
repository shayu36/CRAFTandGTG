"""Strict local-artifact loading for Stage-4 Diffusion."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from three_layer_rag import STAGE2_SPECTRAL_VERSION, ThreeLayerRAGInputs, ThreeLayerRAGMemory


SNAPSHOT_BUNDLE_VERSION = "three-layer-rag-snapshot-bundle-v1"
DYNAMIC_FEATURE_VERSION = "three-layer-rag-dynamics-v1"
CHANNEL_NAMES = {
    "road": ("passage_count", "speed_kmh", "travel_time_seconds"),
    "syntax": ("passage_count", "speed_kmh", "travel_time_seconds"),
    "region": ("in_flow", "out_flow"),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_from_spectral(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "city_id": payload.get("city_id"),
        "joint_graph_hash": payload.get("joint_graph_hash"),
        "checkpoint_fingerprint": payload.get("checkpoint_fingerprint"),
        "static_feature_version": payload.get("static_feature_version"),
        "spectral_feature_version": payload.get("format_version"),
        "road_node_range": tuple(payload.get("road_node_range", ())),
        "syntax_node_range": tuple(payload.get("syntax_node_range", ())),
        "region_node_range": tuple(payload.get("region_node_range", ())),
    }


def load_stage2_high_features(
    path: str | Path,
    *,
    expected_city_id: str | None = None,
    expected_graph_metadata: Mapping[str, Any] | None = None,
    expected_low_features: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format_version") != STAGE2_SPECTRAL_VERSION:
        raise ValueError("严格模式: Diffusion 只接受 Stage-2 joint GraphGPS spectral v2")
    identity = _identity_from_spectral(payload)
    if expected_city_id is not None and identity["city_id"] != expected_city_id:
        raise ValueError("严格模式: high feature city_id 不匹配")
    if expected_graph_metadata is not None:
        expected = dict(expected_graph_metadata)
        for key in (
            "joint_graph_hash", "checkpoint_fingerprint", "static_feature_version",
            "spectral_feature_version", "road_node_range", "syntax_node_range",
            "region_node_range",
        ):
            actual = identity[key]
            wanted = expected.get(key)
            if key.endswith("_range") and wanted is not None:
                wanted = tuple(wanted)
            if wanted is not None and actual != wanted:
                raise ValueError(f"严格模式: high/snapshot graph identity 不匹配: {key}")
    high: dict[str, torch.Tensor] = {}
    for layer in ("road", "syntax", "region"):
        key = f"H_{layer}_high"
        value = payload.get(key)
        node_range = identity[f"{layer}_node_range"]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or not torch.isfinite(value).all()
            or len(node_range) != 2
            or value.shape[0] != node_range[1] - node_range[0]
        ):
            raise ValueError(f"严格模式: 非法 Stage-2 高频字段 {key}")
        high[key] = value.detach().float().contiguous()
        if expected_low_features is not None:
            low_key = f"H_{layer}_low"
            expected_low = expected_low_features.get(layer)
            actual_low = payload.get(low_key)
            if (
                not isinstance(expected_low, torch.Tensor)
                or not isinstance(actual_low, torch.Tensor)
                or expected_low.shape != actual_low.shape
                or not torch.equal(expected_low.detach().cpu(), actual_low.detach().cpu())
            ):
                raise ValueError(
                    f"严格模式: snapshot {layer} low tensor 与 Stage-2 stable node order 不一致"
                )
    if len({value.shape[1] for value in high.values()}) != 1:
        raise ValueError("严格模式: 三层 high feature hidden dimension 不一致")
    return high, identity


def load_snapshot_bundle(
    path: str | Path,
    *,
    splits: Iterable[str] | None = None,
) -> tuple[list[ThreeLayerRAGInputs], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format_version") != SNAPSHOT_BUNDLE_VERSION:
        raise ValueError("严格模式: snapshot bundle 版本不匹配")
    allowed = set(splits) if splits is not None else None
    snapshots = []
    for item in payload.get("snapshots", []):
        if not isinstance(item, ThreeLayerRAGInputs):
            raise ValueError("严格模式: snapshot bundle 含未知对象")
        item.validate()
        if allowed is None or item.split in allowed:
            snapshots.append(item)
    if not snapshots:
        raise ValueError("严格模式: snapshot bundle 在指定 split 下为空")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("dynamic_feature_version") != DYNAMIC_FEATURE_VERSION:
        raise ValueError("严格模式: dynamic feature version 不匹配")
    return snapshots, metadata


@dataclass(frozen=True)
class DynamicNormalizer:
    mode: str
    layers: Mapping[str, Mapping[str, tuple[float, ...]]]
    fingerprint: str
    source_path: str

    @classmethod
    def load(cls, path: str | Path) -> "DynamicNormalizer":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("format_version") != DYNAMIC_FEATURE_VERSION:
            raise ValueError("严格模式: dynamic normalizer version 不匹配")
        mode = str(payload.get("mode"))
        if mode not in {"log1p_zscore", "none"}:
            raise ValueError(f"严格模式: 未知 dynamic normalization: {mode}")
        layers = {}
        for layer, channel_names in CHANNEL_NAMES.items():
            config = payload.get("layers", {}).get(layer)
            if not isinstance(config, Mapping):
                raise ValueError(f"严格模式: normalizer 缺少 {layer}")
            mean = tuple(float(value) for value in config.get("mean", ()))
            std = tuple(float(value) for value in config.get("std", ()))
            if len(mean) != len(channel_names) or len(std) != len(channel_names):
                raise ValueError(f"严格模式: normalizer {layer} channel 数不匹配")
            if any(not math_is_finite(value) for value in (*mean, *std)) or any(value <= 0 for value in std):
                raise ValueError(f"严格模式: normalizer {layer} mean/std 非法")
            layers[layer] = {"mean": mean, "std": std}
        return cls(mode, layers, file_sha256(source), str(source.resolve()))

    def inverse(self, layer: str, normalized: torch.Tensor, *, nonnegative: bool = True) -> torch.Tensor:
        if layer not in CHANNEL_NAMES:
            raise KeyError(f"未知动态层级 {layer}")
        if normalized.ndim < 2 or normalized.shape[-2] != len(CHANNEL_NAMES[layer]):
            raise ValueError(f"{layer} normalized tensor channel axis 必须位于倒数第二维")
        if not torch.isfinite(normalized).all():
            raise ValueError("严格模式: 反归一化输入含 NaN/Inf")
        shape = [1] * normalized.ndim
        shape[-2] = len(CHANNEL_NAMES[layer])
        mean = normalized.new_tensor(self.layers[layer]["mean"]).reshape(shape)
        std = normalized.new_tensor(self.layers[layer]["std"]).reshape(shape)
        value = normalized if self.mode == "none" else torch.expm1(normalized * std + mean)
        if nonnegative:
            value = value.clamp_min(0.0)
        return value

    def validate_bundle_metadata(self, metadata: Mapping[str, Any]) -> None:
        declared = metadata.get("normalization")
        if not isinstance(declared, Mapping):
            raise ValueError("严格模式: snapshot bundle 缺少 normalization metadata")
        if declared.get("format_version") != DYNAMIC_FEATURE_VERSION:
            raise ValueError("严格模式: bundle normalizer version 不匹配")
        if str(declared.get("mode")) != self.mode:
            raise ValueError("严格模式: bundle/source normalizer mode 不匹配")
        for layer in CHANNEL_NAMES:
            config = declared.get("layers", {}).get(layer, {})
            mean = tuple(float(value) for value in config.get("mean", ()))
            std = tuple(float(value) for value in config.get("std", ()))
            if mean != tuple(self.layers[layer]["mean"]) or std != tuple(self.layers[layer]["std"]):
                raise ValueError(f"严格模式: bundle/source normalizer 不匹配: {layer}")


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


class SnapshotDataset(Dataset[ThreeLayerRAGInputs]):
    def __init__(self, snapshots: Sequence[ThreeLayerRAGInputs]):
        if not snapshots:
            raise ValueError("SnapshotDataset 不能为空")
        self.snapshots = tuple(snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int) -> ThreeLayerRAGInputs:
        return self.snapshots[index]


class CityGraphBucketBatchSampler(Sampler[list[int]]):
    """Batch only equal city/graph identities; no cross-city tensor padding."""

    def __init__(
        self,
        snapshots: Sequence[ThreeLayerRAGInputs],
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        buckets: dict[tuple[str, str], list[int]] = {}
        for index, snapshot in enumerate(snapshots):
            graph_hash = str((snapshot.graph_metadata or {}).get("joint_graph_hash", ""))
            buckets.setdefault((snapshot.city_id, graph_hash), []).append(index)
        self.buckets = buckets
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.buckets.values():
            current = list(indices)
            if self.shuffle:
                rng.shuffle(current)
            for start in range(0, len(current), self.batch_size):
                batch = current[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.buckets.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def collate_city_graph_bucket(items: Sequence[ThreeLayerRAGInputs]) -> list[ThreeLayerRAGInputs]:
    """Return a homogeneous list; the trainer accumulates it without bad padding."""

    if not items:
        raise ValueError("空 snapshot batch")
    identities = {
        (item.city_id, (item.graph_metadata or {}).get("joint_graph_hash")) for item in items
    }
    if len(identities) != 1:
        raise ValueError("严格模式: 不同 city/graph identity 不能直接 stack")
    return list(items)


def validate_runtime_identities(
    snapshots: Sequence[ThreeLayerRAGInputs],
    high_identities: Mapping[str, Mapping[str, Any]],
    memory: ThreeLayerRAGMemory,
) -> dict[str, Any]:
    memory.validate()
    graph_hashes: dict[str, str] = {}
    checkpoint_fingerprints: set[str] = set()
    static_versions: set[str] = set()
    spectral_versions: set[str] = set()
    city_low: dict[str, dict[str, torch.Tensor]] = {}
    for snapshot in snapshots:
        metadata = dict(snapshot.graph_metadata or {})
        city = snapshot.city_id
        if city not in high_identities:
            raise ValueError(f"严格模式: 缺少 city={city} 高频特征")
        identity = high_identities[city]
        for key in (
            "joint_graph_hash", "checkpoint_fingerprint", "static_feature_version",
            "spectral_feature_version", "road_node_range", "syntax_node_range",
            "region_node_range",
        ):
            expected = metadata.get(key)
            actual = identity.get(key)
            if key.endswith("_range"):
                expected = tuple(expected or ())
                actual = tuple(actual or ())
            if expected != actual:
                raise ValueError(f"严格模式: city={city} runtime identity 不匹配: {key}")
        previous = graph_hashes.setdefault(city, str(identity["joint_graph_hash"]))
        if previous != str(identity["joint_graph_hash"]):
            raise ValueError(f"严格模式: city={city} graph hash 不稳定")
        checkpoint_fingerprints.add(str(identity["checkpoint_fingerprint"]))
        static_versions.add(str(identity["static_feature_version"]))
        spectral_versions.add(str(identity["spectral_feature_version"]))
        if city not in city_low:
            city_low[city] = {
                layer: snapshot.low_features[layer].detach().cpu()
                for layer in ("road", "syntax", "region")
            }
        elif any(
            not torch.equal(city_low[city][layer], snapshot.low_features[layer].detach().cpu())
            for layer in ("road", "syntax", "region")
        ):
            raise ValueError(f"严格模式: city={city} snapshot low stable node order 不一致")
    for city, memory_identity in dict(memory.graph_identity or {}).items():
        memory_hash = str(memory_identity.get("joint_graph_hash", ""))
        if not memory_hash:
            raise ValueError(f"严格模式: RAG memory 缺少 city={city} joint_graph_hash")
        previous_hash = graph_hashes.setdefault(city, memory_hash)
        if previous_hash != memory_hash:
            raise ValueError(f"严格模式: city={city} snapshot/memory graph hash 不一致")
        checkpoint_fingerprints.add(str(memory_identity.get("checkpoint_fingerprint", "")))
        static_versions.add(str(memory_identity.get("static_feature_version", "")))
        spectral_versions.add(str(memory_identity.get("spectral_feature_version", "")))
        if city not in high_identities:
            continue
        high_identity = high_identities[city]
        for key in (
            "joint_graph_hash", "checkpoint_fingerprint", "static_feature_version",
            "spectral_feature_version", "road_node_range", "syntax_node_range",
            "region_node_range",
        ):
            left, right = memory_identity.get(key), high_identity.get(key)
            if key.endswith("_range"):
                left, right = tuple(left or ()), tuple(right or ())
            if left != right:
                raise ValueError(f"严格模式: city={city} RAG memory/high identity 不匹配: {key}")
    if len(checkpoint_fingerprints) != 1:
        raise ValueError("严格模式: 三城 Stage-2 checkpoint fingerprint 不一致")
    if len(static_versions) != 1 or len(spectral_versions) != 1:
        raise ValueError("严格模式: 三城 feature version 不一致")
    return {
        "graphgps_checkpoint_fingerprint": next(iter(checkpoint_fingerprints)),
        "joint_graph_hashes": graph_hashes,
        "static_feature_version": next(iter(static_versions)),
        "spectral_feature_version": next(iter(spectral_versions)),
        "rag_memory_version": memory.version,
    }
