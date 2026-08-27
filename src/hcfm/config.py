"""沿用 YAML 的递归继承、路径解析与模式约束。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .flow_matching import validate_velocity_consistency


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _seen: set[Path] | None = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"配置继承循环: {path}")
    seen.add(path)
    with open(path) as handle:
        current = yaml.safe_load(handle) or {}
    base_name = current.pop("base_config", None)
    if base_name:
        base = load_config(path.parent / base_name, seen)
        current = _merge(base, current)
    validate_config(current)
    return current


def validate_config(config: Mapping[str, Any]) -> None:
    mode = config.get("model_mode")
    if mode not in {"stage1_diffusion", "macro_flow_matching", "hierarchical_flow_matching"}:
        raise ValueError(f"非法 model_mode={mode!r}")
    generator = config.get("generator_type")
    expected = "diffusion" if mode == "stage1_diffusion" else "flow_matching"
    if generator != expected:
        raise ValueError(f"{mode} 必须使用 generator_type={expected}")
    if mode == "stage1_diffusion":
        if config.get("use_hierarchy") or config.get("generate_micro"):
            raise ValueError("stage1_diffusion 不得启用 hierarchy/micro generation")
        return
    flow = config.get("flow_matching", {})
    if flow.get("solver") not in {"euler", "heun"} or int(flow.get("steps", 0)) <= 0:
        raise ValueError("Flow Matching solver/steps 配置非法")
    prior = flow.get("prior_mode", config.get("prior_mode"))
    validate_velocity_consistency(prior, float(config.get("loss", {}).get("cross_velocity", 0.0)))
    if mode == "macro_flow_matching" and (config.get("use_hierarchy") or config.get("generate_micro")):
        raise ValueError("macro_flow_matching 只允许宏观生成")
    if mode == "hierarchical_flow_matching" and not config.get("use_hierarchy"):
        raise ValueError("hierarchical_flow_matching 必须 use_hierarchy=true")

