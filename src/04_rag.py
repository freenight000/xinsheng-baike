# -*- coding: utf-8 -*-
"""
阶段 4：RAG 合体 —— 让它真的回答问题
======================================

运行方法
--------
双击 run4.bat，或者：

    .venv\\Scripts\\python.exe src\\04_rag.py

现在缺的是什么
--------------
阶段 3 结束时，程序能告诉你"这几块资料相关"，但**不会替你读它们**。
你得自己打开原文，一段一段翻——那不算"问答机器人"。

这一步要做的事，用一句话说清：

    把检索出来的资料，连同问题，一起发给模型，让它**照着资料**回答。

整条链路长这样：
    问题 → 检索 5 块 → 打包成 prompt → 模型 → 带出处的答案


为什么这一步能让幻觉消失
------------------------
回想阶段 1 的实验：你问"宿舍是几人间"，模型张口就编"四人间、上床下桌"。
当时的结论是——**prompt 管得了态度，管不了知识。**

现在知识有了（3000+ 块真资料）。而 prompt 的任务也变了：
    阶段 1 要它「不知道就说不知道」（因为它手上什么都没有）
    阶段 4 要它「只根据资料回答」（因为现在它有资料了）

这个「怎么让它老老实实照着资料说」的规则，就是下面要你写的东西。


关于成本
--------
每次提问，我们要把 5 块资料（约 2000 字）一起发出去。
输入 token 从 ~190 涨到 ~1500，成本仍然只有 0.002 元左右。
这就是 RAG 划算的地方：**只带相关的部分**，而不是把 20 万字全塞进去。
"""

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY or API_KEY.startswith("sk-在这里"):
    print("还没配置 API Key，请先改 .env")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")
MODEL = "deepseek-flash"

# 每次给模型看几块资料
#   太少 → 资料不够，答不全
#   太多 → 输入变贵、无关内容还会干扰它
#   3~5 是常见起点
TOP_K = 5

# 每块资料最多保留多少字（防止个别超长块把 prompt 撑爆）
MAX_CHUNK_CHARS = 600


def load_search_module():
    """
    把 03_search.py 里的东西拿来用。

    为什么不直接 `import 03_search`？
    因为 Python 规定模块名不能以数字开头（那会被当成立方数字面量）。
    这是语言本身的限制，不是我们写错了。
    所以我们用 importlib 手动把它加载进来。
    """
    path = Path(__file__).resolve().parent / "03_search.py"
    spec = importlib.util.spec_from_file_location("search3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# RAG 的系统提示词（产品版）
# 目标不是“显得什么都知道”，而是“有据可查、不知道就认怂、个人信息不泄露”。
RAG_SYSTEM_PROMPT = """
你是「新生百科」，重庆大学新生的学习生活 AI 顾问，服务对象是 2026 级新生。

每次回答前，你会收到一些《参考资料》。请严格遵守：

1. 只依据《参考资料》回答。资料里没有的内容，不要用常识补全，也不要猜测。

2. 如果资料里找不到答案，直接说“资料里没有这方面的信息”，
   并建议新生问辅导员，或查看学校官方最新通知。

3. 每条关键结论后面注明来源：文件名 + 第几块。
   如果多份资料说法冲突，指出冲突，并提醒以最新官方通知为准。

4. 如果资料带有“时效性提醒”或“仅适用于某年级”的声明，
   必须在回答里提醒用户。

5. 不透露、不推测任何个人隐私（姓名、学号、手机号、群聊内容等）。
   资料里出现个人信息时忽略。

6. 回答简洁、分点、可执行；不要编造日期、金额、电话、比例等数字。
"""


def build_context(results, chunks) -> str:
    """把检索到的资料块拼成一段文字，准备发给模型。"""
    blocks = []
    for i, (sc, idx) in enumerate(results, 1):
        c = chunks[idx]
        text = c["text"][:MAX_CHUNK_CHARS]
        blocks.append(
            f"《参考资料 {i}》\n"
            f"出处：{c['source']}\n"
            f"内容：{text}"
        )
    return "\n\n".join(blocks)


def ask_model(question: str, context: str):
    """把资料和问题一起发给模型。"""
    user_content = (
        f"以下是《参考资料》：\n\n{context}\n\n"
        f"---\n\n"
        f"新生的问题：{question}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )

    u = resp.usage
    return resp.choices[0].message.content, u


def main():
    print("=" * 72)
    print("  阶段 4：新生百科全书 · RAG 问答")
    print("=" * 72)

    search3 = load_search_module()

    chunks = search3.load_chunks()
    print(f"  载入 {len(chunks)} 个资料块，正在建索引...")
    postings, chunk_words = search3.build_index(chunks)
    idf = search3.compute_idf(postings, len(chunks))
    print(f"  索引就绪。每次检索 {TOP_K} 块资料")
    print()
    print("  输入问题后回车；输入 exit 退出")
    print()

    while True:
        try:
            question = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        results, q_words = search3.search(
            question, chunks, postings, chunk_words, idf, top_k=TOP_K
        )

        if not results:
            print("\n  资料库里找不到相关内容。")
            print("  试试换个说法，或者把问题写得更具体一点。\n")
            continue

        print(f"\n  〔检索到 {len(results)} 块资料，正在生成回答...〕")

        context = build_context(results, chunks)

        try:
            answer, usage = ask_model(question, context)
        except AuthenticationError:
            print("\n[出错] API Key 不对，检查 .env\n")
            continue
        except APIConnectionError:
            print("\n[出错] 连不上 DeepSeek，检查网络\n")
            continue
        except APIStatusError as e:
            print(f"\n[出错] 服务端返回 {e.status_code}\n")
            continue

        print()
        print(answer)
        print()

        # 把用了哪几块资料列出来，方便你自己核查
        print("  ┌─ 本次用到的资料 ─────────────────────────────────")
        for i, (sc, idx) in enumerate(results, 1):
            c = chunks[idx]
            print(f"  │ {i}. {c['source']}  (第 {c['index']} 块, 得分 {sc:.1f})")
        print("  └──────────────────────────────────────────────────")
        print(f"  〔{usage.total_tokens} tokens："
              f"输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens}〕")
        print()


if __name__ == "__main__":
    main()
