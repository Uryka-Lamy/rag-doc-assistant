"""文档加载：支持 PDF / Word(docx) / Markdown / TXT。"""
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from .config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


def _read_text_with_fallback(path: Path) -> str:
    """先按 UTF-8 读，失败再尝试 GBK，兼容国内常见的中文编码。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> list[Document]:
    """按扩展名分发到对应的 LangChain Loader。"""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {ext}（支持 {sorted(SUPPORTED_EXTENSIONS)}）")

    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()
        if not docs or not docs[0].page_content.strip():
            # TextLoader 编码失败时内容可能为空，退回手动读取
            docs = [Document(page_content=_read_text_with_fallback(path))]

    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["filename"] = path.name
    return docs


def load_documents_from_dir(dir_path: Path) -> list[Document]:
    """加载目录下所有受支持的文档。"""
    all_docs: list[Document] = []
    for file in sorted(dir_path.rglob("*")):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_docs.extend(load_document(file))
    return all_docs
