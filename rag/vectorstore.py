"""文本切分 + FAISS 向量库管理（持久化到本地）。"""
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings
from .loader import load_documents_from_dir


def _stage_dir() -> Path:
    """纯 ASCII 中转目录。

    FAISS 的 C++ 层 fopen 在 Windows 上无法处理含中文的路径，
    因此保存/加载索引都先经过 ASCII 中转目录，再由 Python 层搬运。
    """
    d = Path(tempfile.gettempdir()) / "rag_faiss_stage"
    d.mkdir(exist_ok=True)
    return d


def _clear_dir(d: Path) -> None:
    for f in d.iterdir():
        f.unlink() if f.is_file() else shutil.rmtree(f)


def save_vectorstore(vs: FAISS) -> None:
    """把向量库安全持久化到项目目录（绕过 FAISS 的中文路径问题）。"""
    stage = _stage_dir()
    _clear_dir(stage)
    vs.save_local(str(stage))
    if settings.vector_dir.exists():
        shutil.rmtree(settings.vector_dir)
    settings.vector_dir.parent.mkdir(exist_ok=True)
    shutil.move(str(stage), str(settings.vector_dir))
    stage.mkdir(exist_ok=True)  # move 会搬走中转目录，重建备用


def load_vectorstore() -> FAISS | None:
    """从项目目录加载向量库；不存在则返回 None。"""
    if not (settings.vector_dir / "index.faiss").exists():
        return None
    stage = _stage_dir()
    _clear_dir(stage)
    for f in settings.vector_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, stage / f.name)
    return FAISS.load_local(
        str(stage),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


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
    save_vectorstore(vs)
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
    return load_vectorstore()


def reset_vectorstore() -> None:
    """清空本地向量库。"""
    if settings.vector_dir.exists():
        shutil.rmtree(settings.vector_dir)
