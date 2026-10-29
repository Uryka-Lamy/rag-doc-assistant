"""RAG 问答链：历史感知检索 + 文档增强生成 + 多轮会话记忆。"""
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from .config import settings
from .vectorstore import get_embeddings, load_or_none

# ---- 提示词模板 -----------------------------------------------------------

# 1) 历史感知：把多轮对话 + 追问 改写成独立完整的问题，解决“它为什么好”这类指代问题
CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "根据聊天历史和用户的最新提问，把最新提问改写成一个不依赖上下文也能理解的独立问题。"
            "不要回答问题，只输出改写后的问题；若无需改写则原样输出。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2) 知识问答：只根据检索到的资料回答，答不了就明说，并要求标注来源
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的中文知识库问答助手。请仅依据下面提供的资料回答用户问题。\n"
            "规则：\n"
            "1. 优先使用资料内容作答，不要编造；\n"
            "2. 若资料不足以回答，直接说明“资料中没有相关内容”，可补充自己的常识但需注明；\n"
            "3. 回答使用简洁的中文 Markdown，并在末尾用【来源: 文件名】标注引用的文档。\n\n"
            "资料：\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ---- 会话记忆存储（内存版，按 session_id 隔离） -----------------------------

_session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def clear_session(session_id: str) -> None:
    _session_store.pop(session_id, None)


# ---- 链构建 ---------------------------------------------------------------


def get_llm() -> ChatOpenAI:
    """智谱 GLM 通过 OpenAI 兼容协议调用。"""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.7,
        streaming=True,
    )


def build_rag_chain(vectorstore):
    """组装完整的对话式 RAG 链：

    用户提问 -> [LLM改写独立问题] -> [向量检索 top-k]
             -> [资料+问题+历史 -> LLM] -> 答案 + 来源文档
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONDENSE_PROMPT
    )
    qa_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history",
    )


def get_chain():
    """加载本地向量库并构建链；向量库不存在时返回 None。"""
    vs = load_or_none()
    return build_rag_chain(vs) if vs else None


def embeddings_ready() -> bool:
    return load_or_none() is not None


__all__ = [
    "build_rag_chain",
    "get_chain",
    "clear_session",
    "embeddings_ready",
    "get_embeddings",
]
