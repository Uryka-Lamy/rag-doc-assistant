"""全局配置：从 .env 文件读取环境变量。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（rag/config.py 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载根目录下的 .env
load_dotenv(BASE_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


class Settings:
    """集中管理所有可配置项，避免魔法字符串散落各处。"""

    def __init__(self) -> None:
        # 智谱 AI 平台（OpenAI 兼容协议）
        self.api_key: str = _get("ZHIPU_API_KEY")
        self.base_url: str = _get("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        self.llm_model: str = _get("LLM_MODEL", "glm-4-flash")
        self.embedding_model: str = _get("EMBEDDING_MODEL", "embedding-3")

        # 检索参数
        self.chunk_size: int = int(_get("CHUNK_SIZE", "500"))
        self.chunk_overlap: int = int(_get("CHUNK_OVERLAP", "80"))
        self.top_k: int = int(_get("TOP_K", "4"))

        # 目录
        self.vector_dir: Path = BASE_DIR / "vectorstore"
        self.upload_dir: Path = BASE_DIR / "uploads"

    def validate(self) -> None:
        """启动前校验 API Key 是否已配置。"""
        if not self.api_key or self.api_key == "your_api_key_here":
            raise RuntimeError(
                "未配置 ZHIPU_API_KEY！\n"
                "请复制 .env.example 为 .env，并填入你的智谱 API Key。\n"
                "获取地址: https://open.bigmodel.cn"
            )


settings = Settings()
