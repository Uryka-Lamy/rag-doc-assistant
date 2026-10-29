"""Web UI：基于 Gradio 的 RAG 知识库问答助手。

启动: python app.py  （浏览器自动打开 http://127.0.0.1:7860）
"""
import shutil
from pathlib import Path

import gradio as gr

from rag import settings
from rag.chain import clear_session, embeddings_ready, get_chain
from rag.vectorstore import ingest_paths, ingest_sample, reset_vectorstore

# ---- 界面美化：主题 + 自定义样式 ------------------------------------------

CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
footer {display: none !important;}

/* 面板卡片化 */
#kb-panel, #chat-panel {
    border-radius: 16px !important;
    background: #ffffff !important;
    box-shadow: 0 4px 18px rgba(31, 41, 55, .07) !important;
}

/* 区块小标题 */
.panel-title {
    font-weight: 700; font-size: 15px; color: #0f172a;
    display: flex; align-items: center; gap: 6px; margin: 2px 0 6px;
}

/* 主按钮：渐变 + 悬浮微动效 */
#upload-btn {
    background: linear-gradient(135deg, #0ea5e9, #10b981) !important;
    color: #fff !important; border: none !important; font-weight: 600;
    box-shadow: 0 4px 14px rgba(14, 165, 233, .35);
    transition: transform .12s ease, filter .12s ease;
}
#upload-btn:hover { filter: brightness(1.06); transform: translateY(-1px); }

/* 次按钮 / 危险按钮 */
#sample-btn {background: #eef2ff !important; color: #4f46e5 !important;
    border: 1px solid #c7d2fe !important; font-weight: 600;}
#reset-btn {font-weight: 600;}

/* 聊天区 */
#chatbot {height: 520px !important; border-radius: 12px !important;}
#chatbot .message.user .message-content,
#chatbot .message.user {
    background: linear-gradient(135deg, #0ea5e9, #10b981) !important;
    color: #fff !important; border-radius: 14px 14px 4px 14px !important;
}
#chatbot .message.bot .message-content,
#chatbot .message.bot {
    background: #f1f5f9 !important; color: #0f172a !important;
    border-radius: 14px 14px 14px 4px !important;
}

/* 状态框 */
#status-box textarea {background: #f8fafc !important; font-size: 13px !important;}

/* 上传组件 */
#files-box {border-radius: 12px !important;}
"""


def format_sources(context_docs) -> str:
    """把检索到的来源文档格式化为可折叠的引用区。"""
    if not context_docs:
        return ""
    seen, lines = set(), []
    for doc in context_docs:
        name = doc.metadata.get("filename", "未知文档")
        if name not in seen:
            seen.add(name)
            snippet = doc.page_content.strip().replace("\n", " ")[:60]
            lines.append(f"- 【{name}】{snippet}…")
    return "\n\n<details><summary>📎 检索到的来源</summary>\n\n" + "\n".join(lines) + "\n</details>"


def respond(message: str, chat_history: list):
    """处理一轮问答；知识库未就绪时给出引导提示。"""
    chain = get_chain()
    if chain is None and (settings.vector_dir / "index.faiss").exists():
        chain = get_chain()  # 索引刚建完时可能加载竞争，重试一次
    if chain is None:
        chat_history.append({"role": "user", "content": message})
        chat_history.append(
            {"role": "assistant", "content": "⚠️ 知识库还没有建立，请先在左侧上传文档并点击「上传建库」，或点击「加载示例文档」。"}
        )
        yield "", chat_history
        return

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "🤔 正在检索资料……"})
    yield "", chat_history

    result = chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": "web"}},
    )
    answer = result["answer"] + format_sources(result.get("context"))
    chat_history[-1]["content"] = answer
    yield "", chat_history


def _file_info(f):
    """兼容不同 Gradio 版本的文件对象，返回 (临时路径, 原始文件名)。"""
    if isinstance(f, dict):
        return f.get("path") or f.get("name"), f.get("orig_name")
    if isinstance(f, str):  # Gradio 6 的 NamedString：字符串值即临时文件路径
        tmp = str(f)
        return tmp, Path(tmp).name
    tmp = getattr(f, "path", None) or getattr(f, "name", None)
    return tmp, getattr(f, "orig_name", None) or Path(tmp).name


def on_upload(files, progress=gr.Progress()):
    """保存上传文件并构建向量库。"""
    try:
        if not files:
            return "请先选择要上传的文档（支持 PDF / DOCX / MD / TXT）"
        saved: list[str] = []
        settings.upload_dir.mkdir(exist_ok=True)
        for f in files:
            tmp, orig = _file_info(f)
            dest = settings.upload_dir / Path(orig or Path(tmp).name).name
            shutil.copy(tmp, dest)
            saved.append(str(dest))
            progress((files.index(f) + 1) / len(files), desc=f"已接收: {dest.name}")
        logs: list[str] = []
        n = ingest_paths(saved, progress_cb=logs.append)
        return "\n".join(logs) + f"\n✅ 完成！共 {n} 个文本块入库，可以开始提问了。"
    except Exception as e:  # noqa: BLE001 把失败原因直接显示在状态框
        import traceback
        return "❌ 建库失败: " + repr(e) + "\n" + traceback.format_exc()[-600:]


def on_sample():
    logs: list[str] = []
    try:
        n = ingest_sample(progress_cb=logs.append)
        return "\n".join(logs) + f"\n✅ 示例知识库就绪（{n} 个文本块），试试问：这门课的考核方式是什么？"
    except Exception as e:  # noqa: BLE001
        return f"❌ 加载失败: {e}"


def on_reset():
    reset_vectorstore()
    clear_session("web")
    return "🗑️ 知识库已清空。"


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "PingFang SC", "Microsoft YaHei", "sans-serif"],
    )
    with gr.Blocks(title="智汇文档 · RAG 知识库问答助手", css=CUSTOM_CSS, theme=theme) as demo:
        gr.HTML(
            '<div style="background:linear-gradient(135deg,#06b6d4 0%,#10b981 100%);'
            'border-radius:16px;padding:22px 28px;box-shadow:0 10px 30px rgba(16,185,129,.28);margin-bottom:8px;">'
            '<h1 style="color:#fff;font-size:24px;margin:0;font-weight:700;">📚 智汇文档 · RAG 知识库问答助手</h1>'
            '<p style="color:rgba(255,255,255,.9);font-size:14px;margin:8px 0 0;">'
            'LangChain + 智谱 GLM ｜ 上传文档 → 智能切分 → 向量检索 → 多轮问答，答案自动标注来源</p></div>'
        )
        with gr.Row():
            with gr.Column(scale=1, elem_id="kb-panel"):
                gr.HTML('<div class="panel-title">🗂️ 知识库管理</div>')
                files = gr.File(
                    label="上传文档（PDF / DOCX / MD / TXT，可多选）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".md", ".txt"],
                    elem_id="files-box",
                )
                with gr.Row():
                    btn_upload = gr.Button("⬆️ 上传建库", variant="primary", elem_id="upload-btn")
                    btn_sample = gr.Button("🧪 加载示例", variant="secondary", elem_id="sample-btn")
                btn_reset = gr.Button("🗑️ 清空知识库", variant="stop", elem_id="reset-btn")
                status = gr.Textbox(label="状态", lines=7, interactive=False, elem_id="status-box")
            with gr.Column(scale=2, elem_id="chat-panel"):
                gr.HTML('<div class="panel-title">💬 智能问答</div>')
                chatbot = gr.Chatbot(label="", height=460, elem_id="chatbot")
                msg = gr.Textbox(label="输入问题", placeholder="例如：这门课的考核方式是什么？", autofocus=True)
                with gr.Row():
                    clear = gr.Button("🧹 清空对话", size="sm")

        # 事件绑定
        btn_upload.click(on_upload, inputs=files, outputs=status)
        btn_sample.click(on_sample, outputs=status)
        btn_reset.click(on_reset, outputs=status)
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(lambda: [], outputs=chatbot)
        demo.load(lambda: "✅ 服务已启动。" + ("向量库已就绪，可直接提问。" if embeddings_ready() else "尚未建立知识库，请先上传文档或加载示例。"), outputs=status)
    return demo


if __name__ == "__main__":
    settings.validate()
    build_ui().launch(inbrowser=True)
