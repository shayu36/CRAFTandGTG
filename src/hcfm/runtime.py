"""三模式模型路由。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from .model import HCFMModel, MacroFlowMatchingModel


def build_model(config: Mapping[str, Any]):
    mode = config["model_mode"]
    if mode == "macro_flow_matching":
        return MacroFlowMatchingModel(config)
    if mode == "hierarchical_flow_matching":
        return HCFMModel(config)
    if mode == "stage1_diffusion":
        # 保持原第一阶段类和 state_dict；仅解决其历史绝对 import 方式。
        craft_dir = Path(__file__).resolve().parents[1] / "craft_integrated"
        if str(craft_dir) not in sys.path:
            sys.path.insert(0, str(craft_dir))
        from craft import CRAFTModel
        return CRAFTModel(config)
    raise ValueError(f"未知 model_mode={mode!r}")

