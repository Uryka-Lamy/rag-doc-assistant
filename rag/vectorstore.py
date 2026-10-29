"""文本切分 + FAISS 向量库管理（持久化到本地）。"""
import shutil
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .loader import load_documents_from_dir


def get_embeddings() -> OpenAIEmbeddings:
    """智谱 embedding-3 通过 OpenAI 兼容协议调用。

    check_embedding_ctx_length=False: 避免用 OpenAI 的 tiktoken 预切分 token，
    非官方端点必须关闭，否则接口报错。
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        check_embedding_ctx_length=False,
    )


def split_documents(docs: list[Document]) -> list[Document]:
    """递归字符切分：按 段落 -> 换行 -> 句子 层级拆分，尽量保留语义完整。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(docs: list[Document]) -> tuple[FAISS, int]:
    """文档向量化并入库，返回 (向量库, 切分后的块数)。"""
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("没有解析出任何文本内容，请检查文档是否为空。")
    vs = FAISS.from_documents(chunks, get_embeddings())
    settings.vector_dir.mkdir(exist_ok=True)
    vs.save_local(str(settings.vector_dir))
    return vs, len(chunks)


def ingest_paths(paths: list[str], progress_cb: Callable[[str], None] | None = None) -> int:
    """从文件路径列表构建向量库，返回入库块数（供 UI 调用）。"""
    docs: list[Document] = []
    for p in paths:
        from pathlib import Path
        docs.extend(load_document(Path(p)))
        if progress_cb:
            progress_cb(f"已解析: {Path(p).name}")
    vs, n = build_vectorstore(docs)
    if progress_cb:
        progress_cb(f"向量化完成，共入库 {n} 个文本块")
    return n


def ingest_sample(progress_cb: Callable[[str], None] | None = None) -> int:
    """用 docs/sample 目录下的示例文档一键建库。"""
    sample_dir = settings.vector_dir.parent / "docs" / "sample"
    docs = load_documents_from_dir(sample_dir)
    if not docs:
        raise FileNotFoundError("示例目录下没有可加载的文档")
    vs, n = build_vectorstore(docs)
    if progress_cb:
        progress_cb(f"示例文档向量化完成，共入库 {n} 个文本块")
    return n


def load_or_none() -> FAISS | None:
    """加载本地已有的向量库；不存在则返回 None。"""
    index_file = settings.vector_dir / "index.faiss"
    if not index_file.exists():
        return None
    return FAISS.load_local(
        str(settings.vector_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def reset_vectorstore() -> None:
    """清空本地向量库。"""
    if settings.vector_dir.exists():
        shutil.rmtree(settings.vector_dir)
