"""命令行入口：不启动 Web 界面，直接在终端使用。

用法:
    python cli.py                 # 若无向量库则自动用示例文档建库
    python cli.py --ingest 路径1 路径2   # 用指定文件建库
"""
import argparse
import sys

from rag import settings
from rag.chain import clear_session, get_chain
from rag.vectorstore import ingest_paths, ingest_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 知识库问答助手（CLI 版）")
    parser.add_argument("--ingest", nargs="*", help="用指定文档建立知识库后进入问答")
    args = parser.parse_args()

    settings.validate()

    chain = get_chain()
    if args.ingest:
        print(f"正在为 {len(args.ingest)} 个文档建立向量索引……")
        n = ingest_paths(args.ingest, progress_cb=print)
        print(f"✅ 入库完成，共 {n} 个文本块。")
        chain = get_chain()
    if chain is None:
        print("未检测到知识库，正在用 docs/sample 示例文档建库……")
        n = ingest_sample(progress_cb=print)
        print(f"✅ 示例库就绪（{n} 个文本块）。")
        chain = get_chain()

    print("\n=== 📚 RAG 知识库问答助手（输入 /new 新会话，/quit 退出） ===\n")
    session_id = "cli"
    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in {"/quit", "/exit", "exit", "quit"}:
            break
        if question == "/new":
            clear_session(session_id)
            print("（已开始新会话）\n")
            continue
        try:
            result = chain.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ 调用出错: {e}\n")
            continue
        print(f"\n助手: {result['answer']}")
        sources = {d.metadata.get("filename", "?") for d in result.get("context", [])}
        if sources:
            print(f"📎 来源: {', '.join(sorted(sources))}\n")
    print("再见！")
    sys.exit(0)


if __name__ == "__main__":
    main()
