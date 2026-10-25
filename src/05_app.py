# -*- coding: utf-8 -*-
"""
阶段 5：把问答机器人做成网页
================================

运行方法
--------
双击 run5.bat

（它会自动打开浏览器。想关掉服务，回到那个黑窗口按 Ctrl+C）

注意：这不是一个"双击就结束"的脚本，而是一个**一直运行的服务**。
      它在你电脑上开了个小网站，浏览器访问 localhost:8501 就能用。
      想给同学用，同一个 WiFi 下让他们访问你的局域网地址即可。


Streamlit 最重要的一个特性（理解了它，你就理解了这个文件）
-----------------------------------------------------------
**每次你在网页上点一下、输一句，Streamlit 都会把整个脚本从头到尾重跑一遍。**

这带来两个必须处理的问题：

  问题 1：建索引很慢（要读 3000+ 块、切词、算 IDF）
          如果每问一句话都重建一次，网页会卡到没法用。
          → 解法：@st.cache_resource，让它在第一次之后一直复用。

  问题 2：脚本一重跑，所有变量都会重新初始化，对话历史就没了。
          → 解法：st.session_state，这是 Streamlit 提供的"跨次运行还活着"的储物柜。

这两样是 Streamlit 新手最容易踩的坑。代码里标了 ★ 的地方就是它们。
"""

import importlib.util
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Streamlit 要求 set_page_config 必须是第一个 st.* 调用，否则会报错/警告。
st.set_page_config(
    page_title="新生百科 · 重大学习生活 AI 顾问",
    page_icon="🎓",
    layout="centered",
)

MODEL = "deepseek-flash"
TOP_K = 5
MAX_CHUNK_CHARS = 600


# 产品文案：面向新生，第一屏就说清“能问什么、答案从哪来、不会编”。
WELCOME = (
    "我是「新生百科」——重庆大学新生的学习生活 AI 顾问。"
    "我读过学生手册、培养方案、选课通知等公开资料；"
    "每个回答都会标出来源，资料里没有的我会直接说没有。"
)

EXAMPLE_QUESTIONS = [
     "军训要准备什么？",
    "虎溪校区宿舍床多大？",
    "国家奖学金怎么申请？",
    "挂科了怎么办？",
    "宿舍能不能养猫？",
]


def load_search_module():
    """加载 03_search.py（模块名不能以数字开头，所以用 importlib）。"""
    path = Path(__file__).resolve().parent / "03_search.py"
    spec = importlib.util.spec_from_file_location("search3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_rag_module():
    """加载 04_rag.py，主要是为了复用它的 build_context 和那段 System Prompt。"""
    path = Path(__file__).resolve().parent / "04_rag.py"
    spec = importlib.util.spec_from_file_location("rag4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
#  ★ 关键点 1：@st.cache_resource
# ======================================================================
#  没有这个装饰器，你每敲一句话，Streamlit 都会把 3000+ 块资料
#  重新读一遍、分词一遍、算一遍 IDF —— 每次要等好几秒。
#
#  加上它之后：第一次加载慢一点，之后就一直是"秒回"。
#  show_spinner 是加载时显示的那句提示。
# ======================================================================
@st.cache_resource(show_spinner="正在加载资料库（只需一次）…")
def build_engine():
    search3 = load_search_module()
    chunks = search3.load_chunks()
    postings, chunk_words = search3.build_index(chunks)
    idf = search3.compute_idf(postings, len(chunks))
    return search3, chunks, postings, chunk_words, idf


# Streamlit Cloud 的 Secrets 不一定会进 os.environ；云上没 .env 时从 st.secrets 兜底。
# 本地 .env 仍然优先。
if not os.getenv("DEEPSEEK_API_KEY"):
    try:
        os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key or api_key.startswith("sk-在这里"):
    st.error("还没配置 API Key。请打开项目根目录的 .env 文件填入你的 key。")
    st.stop()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

st.title("🎓 新生百科 · 重大学习生活 AI 顾问")
st.caption(WELCOME)

if not (
    (PROJECT_ROOT / "data" / "chunks.json").exists()
    or (PROJECT_ROOT / "data" / "chunks_demo.json").exists()
):
    st.error("找不到知识库。请先双击 run2.bat 生成 data/chunks.json，"
             "或放入公开演示数据 data/chunks_demo.json。")
    st.stop()

search3, chunks, postings, chunk_words, idf = build_engine()
rag4 = load_rag_module()


# ======================================================================
#  ★ 关键点 2：st.session_state
# ======================================================================
#  Streamlit 每次交互都会重跑整个脚本，普通变量会被重置。
#  session_state 是这个"储物柜"，东西放进去，下次重跑还在。
#
#  这里存两样：对话历史、以及侧边栏按钮点了哪个问题。
# ======================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


with st.sidebar:
    st.header("📚 资料库")
    st.metric("资料块", f"{len(chunks):,}")
    st.metric("来源文件", f"{len({c['source'] for c in chunks})}")
    kb_label = "公开演示版" if "chunks_demo" in str(search3.CHUNKS_FILE) else "本地完整版"
    st.caption(f"{kb_label} · 资料以学校官方最新通知为准")
    st.divider()

    st.header("💡 试试这些问题")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
            st.rerun()

    st.divider()
    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 依据的资料（{len(msg['sources'])} 块）"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** · 第 {s['index']} 块 "
                                f"· 相关度 {s['score']:.1f}")
                    st.caption(s["text"][:300] + "…")


# 取输入：可能是用户敲的，也可能是点了侧边栏的示例问题
typed = st.chat_input("问点什么，比如：军训要准备什么？")
question = typed or st.session_state.pending

if question:
    st.session_state.pending = None

    # 多轮对话：把最近两轮带进上下文，追问“那自动化呢？”能接上。
    # 截断到 800 字，避免历史回答把 prompt 撑得太大。
    history = [
        {"role": m["role"], "content": m["content"][:800]}
        for m in st.session_state.messages[-4:]
    ]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        results, q_words = search3.search(
            question, chunks, postings, chunk_words, idf, top_k=TOP_K
        )

        if not results:
            answer = ("资料库里找不到和这个问题相关的内容。\n\n"
                      "换个说法试试，或者把问题写得更具体一点。")
            sources = []
            st.markdown(answer)
        else:
            context = rag4.build_context(results, chunks)

            with st.spinner("正在翻资料…"):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": rag4.RAG_SYSTEM_PROMPT},
                            *history,
                            {"role": "user",
                             "content": f"以下是《参考资料》：\n\n{context}\n\n"
                                        f"---\n\n新生的问题：{question}"},
                        ],
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    answer = resp.choices[0].message.content
                    usage = resp.usage
                except AuthenticationError:
                    answer = "API Key 不对，请检查 .env"
                    st.error(answer)
                    st.stop()
                except APIConnectionError:
                    answer = "连不上 DeepSeek，检查一下网络。"
                    st.error(answer)
                    st.stop()
                except APIStatusError as e:
                    answer = f"服务端返回 {e.status_code}"
                    st.error(answer)
                    st.stop()

            st.markdown(answer)

            sources = [
                {
                    "source": chunks[i]["source"],
                    "index": chunks[i]["index"],
                    "score": sc,
                    "text": chunks[i]["text"],
                }
                for sc, i in results
            ]

            with st.expander(f"📎 依据的资料（{len(sources)} 块）"):
                for s in sources:
                    st.markdown(f"**{s['source']}** · 第 {s['index']} 块 "
                                f"· 相关度 {s['score']:.1f}")
                    st.caption(s["text"][:300] + "…")

            st.caption(f"本次消耗 {usage.total_tokens} tokens "
                       f"（输入 {usage.prompt_tokens} / 输出 {usage.completion_tokens}）")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
