# -*- coding: utf-8 -*-
"""
阶段 3：检索 —— 从 3000+ 块里找出最相关的那几块
================================================

运行方法
--------
双击 run3.bat，或者：

    .venv\\Scripts\\python.exe src\\03_search.py

这一步在做什么
--------------
阶段 2 结束时你有 3000+ 个资料块，但它们只是"存着"。
现在要解决的是：**你问一个问题，怎么知道该看哪几块？**

答案朴素得有点意外：

    给每一块打一个分，看它跟你的问题有多像，取分数最高的几块。

就这么简单。真正的难点在「怎么算像」。


难点一：中文要分词
------------------
英文天然用空格分开，中文没有。"转专业要多少绩点"，
电脑不会自动知道该切成 `转专业 / 要 / 多少 / 绩点`。

切错了会很惨：如果切成 `转 / 专业 / 要 / 多少 / 绩点`，
"转"这个字到处都是、"专业"在 54 个培养方案里到处都是，
你的问题就会被噪声淹没。所以我们用 jieba 分词。


难点二：jieba 不认识高校的词
----------------------------
实测：jieba 默认把"转专业"切成"转"+"专业"。
它是通用词典，不知道这是高校里的固定说法。

解决办法是**自定义词典**（下面 DOMAIN_WORDS）。
jieba 见到表里的词就不会再切碎。

★ 这张词表是你可以长期改进的地方。每发现一个被切错的词，
  加进来就行。这是中文 RAG 里最实用的一招，成本极低、效果立竿见影。


难点三：怎么算「像」
--------------------
我们用的打分思路是：**命中越罕见的词，分越高**。

    "转专业"只在少数几十块里出现 → 命中它，分数暴涨
    "的""是"几乎每块都有        → 命中它对分数几乎没影响

这个"罕见程度"有个专门的名字叫 **IDF**（逆文档频率）。
不用怕这个名字，它就是个统计量，下面的 compute_idf() 会算给你看。
"""

import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

# jieba 在 Python 3.12 上会刷一堆 SyntaxWarning，屏蔽掉，别吓人
warnings.filterwarnings("ignore", category=SyntaxWarning)

import jieba  # noqa: E402

jieba.setLogLevel(60)  # 关掉 jieba 的初始化日志

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 本地完整版用 data/chunks.json；公开演示版用 data/chunks_demo.json。
# 也可以用环境变量 XINSHENG_CHUNKS 指定路径（相对项目根目录或绝对路径）。
_default_chunks = (
    "chunks.json"
    if (PROJECT_ROOT / "data" / "chunks.json").exists()
    else "chunks_demo.json"
)
CHUNKS_FILE = PROJECT_ROOT / "data" / os.getenv("XINSHENG_CHUNKS", _default_chunks)


# ======================================================================
#  ★ 高校场景专有词表 ★
# ======================================================================
#  jieba 默认不认识这些词，会把它们切碎。
#  加进这张表，分词质量立刻提升一个档次。
#
#  你可以随时往里面加词——这是提高检索准确率性价比最高的一件事。
# ======================================================================
DOMAIN_WORDS = [
    # 学籍与教学
    "转专业", "大类分流", "学籍", "绩点", "学分绩点", "必修课", "选修课",
    "通识课", "培养方案", "核心课程", "补考", "重修", "缓考", "免修",
    "保研", "推免", "辅修", "双学位", "毕业设计", "毕业论文", "学业预警",
    # 日常管理
    "综合素质测评", "综测", "违纪处分", "留校察看", "记过", "警告处分",
    "宿舍", "门禁", "查寝", "辅导员", "班主任", "请假", "销假",
    "校园卡", "学生证", "校牌", "校园网", "第二课堂", "社团",
    # 奖、助、贷
    "奖学金", "国家奖学金", "国家励志奖学金", "助学金", "助学贷款",
    "绿色通道", "勤工助学", "困难补助", "学费减免", "临时困难补助",
    "综合奖学金", "争先创优",
    # 军训与入学
    "军训", "军训须知", "军训准备", "军乐连", "通讯员",
    "报到", "迎新", "入学须知", "新生",
    "体检", "档案", "户口", "团组织关系", "党团组织",
    # 校园生活
    "虎溪校区", "科学城校区", "食堂", "图书馆", "校医院", "快递",
    "选课", "抢课", "教务系统", "学分认定",
    "挂科", "床位尺寸", "床垫", "床多大", "志愿填报",
]

for _w in DOMAIN_WORDS:
    jieba.add_word(_w)

# 这些词几乎每块都有，对区分"哪块更相关"毫无帮助，直接扔掉
STOPWORDS = set("""
的 了 是 在 和 与 及 或 也 都 就 而 但 被 把 给 从 到 对 为 以 之 其 此 该
我 你 他 她 它 我们 你们 他们 这 那 这个 那个 什么 怎么 如何 哪些 哪个
有 无 不 没 很 更 最 会 能 可以 应该 需要 必须 一个 一些 一样 时候
吗 呢 吧 啊 呀 哦 嗯 请问 想问 一下 多少 几个 是否 以及 并且 但是 因为
所以 如果 那么 这里 那里 上面 下面 里面 外面 以后 以前 现在
""".split())

# 新生口语 → 资料里的正式说法。检索不准时，先往这里加词，成本最低。
SYNONYMS = {
    "挂科": ["补考", "重修", "不及格", "学业预警"],
    "补考": ["重修", "挂科"],
    "重修": ["补考", "挂科"],
    "床多大": ["床位尺寸", "床垫", "长度", "宽度"],
    "床位": ["床位尺寸", "床垫"],
    "军训": ["军训须知", "军训准备", "军乐连", "通讯员"],
    "志愿": ["志愿填报", "填报", "专业志愿"],
    "转专业": ["转系", "转专业申请", "转专业管理办法"],
    "宿舍": ["寝室", "公寓"],
    "门禁": ["熄灯", "归宿", "晚归"],
    "奖学金": ["国奖", "国家奖学金", "励志奖学金", "综合奖学金"],
}


def tokenize(text: str) -> list:
    """中文分词：把一句话切成词，并滤掉没用的停用词。"""
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) > 1 and w not in STOPWORDS]


def expand_query(q_words: list) -> list:
    """把口语词扩展成资料里更常见的正式说法（同义词只降权参与，不喧宾夺主）。"""
    expanded = list(q_words)
    for w in q_words:
        for syn in SYNONYMS.get(w, []):
            if syn not in expanded:
                expanded.append(syn)
    return expanded


def load_chunks() -> list:
    if not CHUNKS_FILE.exists():
        print(f"找不到 {CHUNKS_FILE}")
        print("请先运行阶段 2（双击 run2.bat）生成它。")
        sys.exit(1)
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


def build_index(chunks: list):
    """
    建立「倒排索引」。

    正排是：第 1 块有哪些词、第 2 块有哪些词……
    倒排是：词"转专业"出现在第 3、17、42……块里

    倒排的好处是，查"转专业"时不用把 3000+ 块全看一遍，
    直接跳到那几十块就行。搜索引擎就是这么干的。
    """
    postings = defaultdict(set)   # 词 -> {块编号}
    chunk_words = []              # 每块的词频：词 -> 在这一块里出现了几次

    for i, c in enumerate(chunks):
        # 注意这里用的是 Counter（计数器），不是 set。
        # set 会把「出现了几次」这个信息扔掉，Counter 会保留。
        words = Counter(tokenize(c["text"]))
        chunk_words.append(words)
        for w in words:
            postings[w].add(i)

    return postings, chunk_words


def compute_idf(postings: dict, n_chunks: int) -> dict:
    """
    算每个词的 IDF —— 它有多「罕见」。

    公式：idf = log((总块数 + 1) / (出现过这个词的块数 + 1)) + 1

    出现的块越少 → 分母越小 → idf 越大 → 这个词越有区分度。

    举个例子（3000+ 块）：
        出现在 3000 块里的词 → idf ≈ 1.12    （几乎所有块都有，没用）
        出现在   50 块里的词 → idf ≈ 5.22    （很有区分度）
        出现在    2 块里的词 → idf ≈ 8.03    （极度稀有）
    """
    return {
        w: math.log((n_chunks + 1) / (len(ids) + 1)) + 1
        for w, ids in postings.items()
    }


def score(query_words: list, chunk_words, idf: dict, weights=None) -> float:
    """
    给一个块打分。分越高 = 这块和问题越相关。

    核心是 TF-IDF：罕见词（idf 大）加分多，在同一块里出现次数多（tf 大）也加分。
    同义词扩展出来的词用 weights 降权，避免把原问题的关键词冲淡。
    """
    s = 0.0
    for i, w in enumerate(query_words):
        if w in chunk_words:
            weight = 1.0 if weights is None else weights[i]
            s += idf[w] * chunk_words[w] * weight
    return s


def search(query: str, chunks, postings, chunk_words, idf, top_k: int = 5):
    """检索：返回最相关的 top_k 块。"""
    q_words = tokenize(query)
    if not q_words:
        return [], []

    # 同义词扩展：原词权重 1.0，扩展词 0.55（只帮忙，不抢主线）
    expanded = expand_query(q_words)
    weights = [1.0] * len(q_words) + [0.55] * (len(expanded) - len(q_words))

    # 只考虑「至少命中一个查询词或同义词」的块 —— 倒排索引的威力
    candidates = set()
    for w in expanded:
        candidates |= postings.get(w, set())

    q_set = set(q_words)
    scored = []
    for i in candidates:
        base = score(expanded, chunk_words[i], idf, weights)
        if base <= 0:
            continue
        # 来源文件名加权：军训通知、床位尺寸表、转专业细则的关键词
        # 往往就在文件名里，仅靠正文打分会被泛泛的经验帖挤下去。
        src = chunks[i]["source"].lower()
        src_hits = sum(1 for w in q_set if w in src)
        if src_hits:
            base *= 1.0 + 0.45 * min(src_hits, 2)
        # 文件名同时含“转专业”和目标学院/专业（如“自动化学院转专业细则”）
        # 时，这份细则比泛泛的培养方案更该排在前面。
        if "转专业" in q_set and src_hits >= 2:
            base *= 2.00

        text = chunks[i]["text"]
        # 多关键词共现：同一块里同时出现“转专业”和“志愿”时优先，
        # 避免被只出现“志愿”的选课说明抢走。
        if len(q_set) > 1:
            text_hits = sum(1 for w in q_set if w in text)
            if text_hits > 1:
                base *= 1.0 + 0.30 * (text_hits - 1)

        # 通用制度类问题优先学生手册（它讲的是全校规则，
        # 具体学院的细则适合回答“某个学院怎么转”）。
        if "学生手册" in chunks[i]["source"] and any(
            w in q_set for w in ("条件", "规定", "政策", "要求", "办法", "流程")
        ):
            base *= 1.25
        # “转专业”的全校通用规则（条件、能填几个志愿）在学生手册里，
        # 不能被某个学院的细则或选课说明挤掉。
        if "学生手册" in chunks[i]["source"] and "转专业" in q_set:
            base *= 1.30
            if any(w in q_set for w in ("志愿", "几个", "填报")):
                base *= 1.70
                # “转专业能填几个志愿”的标准答案通常写成
                # “只允许申请填报一个专业”，优先这一块。
                if "填报" in chunks[i]["text"] or "一个专业" in chunks[i]["text"]:
                    base *= 2.00

        # 完整短语命中：优先匹配「转专业志愿」这类固定说法
        phrase = query.strip()
        if phrase and phrase in text:
            base *= 1.15
        scored.append((base, i))

    scored.sort(key=lambda x: -x[0])
    return scored[:top_k], q_words


def main():
    print("=" * 70)
    print("  阶段 3：检索")
    print("=" * 70)

    chunks = load_chunks()
    print(f"  载入 {len(chunks)} 个资料块，正在建索引...")

    postings, chunk_words = build_index(chunks)
    idf = compute_idf(postings, len(chunks))

    print(f"  索引建好：{len(postings)} 个不同的词")
    print()
    print("  输入问题后回车；输入 exit 退出")
    print("  提示：输入 词 <某个词> 可以查看它有多罕见")
    print()

    while True:
        try:
            raw = input("问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        # 小工具：看看某个词的 idf
        if raw.startswith("词 "):
            w = raw[2:].strip()
            if w in postings:
                print(f"  「{w}」出现在 {len(postings[w])} 个块里，"
                      f"idf = {idf[w]:.2f}")
            else:
                print(f"  「{w}」不在词表里")
            print()
            continue

        results, q_words = search(raw, chunks, postings, chunk_words, idf)

        print(f"\n  分词结果：{' / '.join(q_words)}")

        if not results:
            print("  没有找到任何相关的块。")
            print("  试试换个说法，或者把关键词写得更具体。\n")
            continue

        print(f"\n  找到 {len(results)} 块（按相关度排序）：")
        for rank, (sc, i) in enumerate(results, 1):
            c = chunks[i]
            preview = c["text"][:110].replace("\n", " ")
            print()
            print(f"  ── 第 {rank} 名  得分 {sc:.2f} ──────────────────────────")
            print(f"     来源：{c['source']}  第 {c['index']} 块")
            print(f"     {preview}...")

        print()


if __name__ == "__main__":
    main()
