#!/usr/bin/env python3
"""第一阶段静态三层图入口。

该文件只是稳定的用户入口，实际实现位于 ``build_static_hierarchy.py``；
不会转入 Diffusion、Flow Matching、RAG 或生成流程。
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_static_hierarchy import main  # noqa: E402


if __name__ == "__main__":
    main()
