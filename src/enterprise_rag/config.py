"""集中配置入口：configs/*.yaml + .env。

全项目唯一的配置读取方式，禁止在其他模块直接读环境变量或 yaml。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# 项目根目录：src/enterprise_rag/config.py 向上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

# .env 只在首次调用时加载一次
_loaded = False


def load_config(name: str = "default") -> dict[str, Any]:
    """按名称加载实验配置，并加载 .env 环境变量。"""
    global _loaded
    if not _loaded:
        load_dotenv(PROJECT_ROOT / ".env")
        _loaded = True

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"配置不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    """读取 LLM API Key。缺失时给出明确提示。"""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("未设置 OPENAI_API_KEY：请复制 .env.example 为 .env 并填入")
    return key


def get_base_url() -> str | None:
    """可选的 OpenAI 兼容中转/本地部署地址。"""
    return os.getenv("OPENAI_BASE_URL")
