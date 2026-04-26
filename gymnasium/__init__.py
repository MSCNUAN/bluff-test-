"""
轻量级 gymnasium 兼容层。
用于在未安装 gymnasium 依赖时，让本项目最小可运行。
"""

from . import spaces


class Env:
    """最小 Env 基类，仅提供本项目需要的 reset 行为。"""

    metadata = {}

    def reset(self, seed=None, options=None):
        return None

