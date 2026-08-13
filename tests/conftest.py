import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# 让测试可 import Paper/src/gtg_features 与 Paper/src/craft_integrated
for p in [os.path.join(_ROOT, "src"), os.path.join(_ROOT, "src", "craft_integrated")]:
    if p not in sys.path:
        sys.path.insert(0, p)

CACHE_DIR = os.path.join(_ROOT, "cache", "gtg")
NORM_DIR = os.path.join(_ROOT, "data", "norm_flow")
CRAFT_ROOT = "/root/autodl-tmp/projects/CRAFT/cleared_data"
CITIES = ["chi", "dc", "toronto", "ny"]
