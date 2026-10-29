"""Web UI：基于 Gradio 的 RAG 知识库问答助手。

启动: python app.py  （浏览器自动打开 http://127.0.0.1:7860）
"""
import shutil
from pathlib import Path

import gradio as gr

from rag import settings
from rag.chain import clear_session, embeddings_ready, get_chain
from rag.vectorstore import ingest_paths, ingest_sample, reset_vectorstore


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


def on_upload(files, progress=gr.Progress()):
    """保存上传文件并构建向量库。"""
    if not files:
        return "请先选择要上传的文档（支持 PDF / DOCX / MD / TXT）"
    saved: list[str] = []
    settings.upload_dir.mkdir(exist_ok=True)
    for f in files:
        dest = settings.upload_dir / Path(f.orig_name or Path(f.name).name).name
        shutil.copy(f.name, dest)
        saved.append(str(dest))
    logs: list[str] = []
    try:
        n = ingest_paths(saved, progress_cb=logs.append)
        return "\n".join(logs) + f"\n✅ 完成！共 {n} 个文本块入库，可以开始提问了。"
    except Exception as e:  # noqa: BLE001 给用户看到失败原因
        return "\n".join(logs) + f"\n❌ 建库失败: {e}"


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
    with gr.Blocks(title="智汇文档 · RAG 知识库问答助手") as demo:
        gr.Markdown(
            "# 📚 智汇文档 · RAG 知识库问答助手\n"
            "LangChain + 智谱 GLM：上传你的文档，基于文档内容进行多轮智能问答"
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🗂️ 知识库管理")
                files = gr.File(
                    label="上传文档（PDF / DOCX / MD / TXT，可多选）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".md", ".txt"],
                )
                with gr.Row():
                    btn_upload = gr.Button("⬆️ 上传建库", variant="primary")
                    btn_sample = gr.Button("🧪 加载示例", variant="secondary")
                btn_reset = gr.Button("🗑️ 清空知识库", variant="stop")
                status = gr.Textbox(label="状态", lines=6, interactive=False)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="问答对话", height=460, type="messages")
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
