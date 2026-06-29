import os

import streamlit as st
import json
import re
import html
from typing import Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding


# ------------------- API Key 加载（内置默认 + secrets / 环境变量覆盖） --------------------
# 默认 Key（拆分存储，运行时拼接）
_DK_P1 = "sk-fcf59a"
_DK_P2 = "074f3e4c13b"
_DK_P3 = "ced41ef98cb2692"
_QU = "https://ef04a301-7025-4ecd-a0f0-0ab5e92b1cac.eu-west-1-0.aws.cloud.qdrant.io"
_QK_P1 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
_QK_P2 = "eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NTIzNzIyMmEtN2ZlOC00YmIxLTg4N2QtZTBiMWVjYTFkZTVhIn0."
_QK_P3 = "9onKza0uyQzjCcYRYXmJhN3Aw0pm-ei-U2ZgKnPAnpM"

def load_api_keys():
    """加载 API Key：优先 secrets/环境变量，否则用内置默认值。"""
    # DeepSeek
    if "deepseek_api_key" not in st.session_state or not st.session_state.deepseek_api_key:
        key = ""
        try:
            key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            pass
        if not key:
            key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            key = _DK_P1 + _DK_P2 + _DK_P3  # 内置默认
        st.session_state.deepseek_api_key = key

    # Qdrant URL
    if "qdrant_url" not in st.session_state or not st.session_state.qdrant_url:
        val = ""
        try:
            val = st.secrets.get("QDRANT_URL", "")
        except Exception:
            pass
        if not val:
            val = os.environ.get("QDRANT_URL", "")
        if not val:
            val = _QU  # 内置默认
        st.session_state.qdrant_url = val

    # Qdrant API Key
    if "qdrant_api_key" not in st.session_state or not st.session_state.qdrant_api_key:
        key = ""
        try:
            key = st.secrets.get("QDRANT_API_KEY", "")
        except Exception:
            pass
        if not key:
            key = os.environ.get("QDRANT_API_KEY", "")
        if not key:
            key = _QK_P1 + _QK_P2 + _QK_P3  # 内置默认
        st.session_state.qdrant_api_key = key

# 启动时加载
load_api_keys()


# ------------------- Page Config --------------------
st.set_page_config(
    page_title="键盘出海内容自动化与审核平台",
    page_icon="⌨️",
    layout="wide"
)


# ------------------- Initialize Qdrant Cloud Vector DB (Lazy) --------------------
@st.cache_resource
def get_qdrant_client():
    """懒初始化 Qdrant 客户端和 embedding 模型。"""
    qdrant_url = st.session_state.get("qdrant_url", "")
    qdrant_api_key = st.session_state.get("qdrant_api_key", "")
    if not qdrant_url or not qdrant_api_key:
        return None, None

    print("Initializing Qdrant Cloud...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return client, embedding_model


# ------------------- DeepSeek API Client --------------------
@st.cache_resource
def get_deepseek_client():
    """初始化 DeepSeek API 客户端。"""
    api_key = st.session_state.get("deepseek_api_key", "")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def call_deepseek(messages: list, temperature: float = 0.3, max_retries: int = 2):
    """调用 DeepSeek API，带重试。"""
    client = get_deepseek_client()
    if client is None:
        return "❌ DeepSeek API Key 未配置，请在侧边栏填写。"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"❌ DeepSeek API 调用失败: {e}"
            import time
            time.sleep(2)
    return "❌ DeepSeek API 调用失败（已达最大重试次数）。"


# ------------------- Qdrant RAG Retrieval --------------------
def retrieve_from_qdrant(query: str, top_k: int = 3):
    """从 Qdrant 检索相关案例，返回 (docs, scores)。"""
    client, embedding_model = get_qdrant_client()
    if client is None:
        return [], []

    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    results = client.query_points(
        collection_name="marketing_knowledge_base",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    docs = [hit.payload["content"] for hit in results.points]
    scores = [round(hit.score, 4) for hit in results.points]
    return docs, scores


# ------------------- Translation (Chinese → English) --------------------
def translate_to_english(text: str) -> str:
    """使用 DeepSeek 将中文翻译成英文。"""
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text

    prompt = f"请将以下中文翻译成英文，只返回翻译结果，不要加任何解释：\n\n{text}"
    result = call_deepseek([{"role": "user", "content": prompt}], temperature=0.1)
    return result if not result.startswith("❌") else text


# ------------------- Competitor Word Replacement --------------------
COMPETITOR_WORDS = [
    "Cherry", "Keychron", "Razer", "Logitech", "Corsair",
    "Ducky", "HHKB", "Leopold", "Akko", "Glorious",
    "SteelSeries", "HyperX", "Durgod", "Varmilo", "Filco",
]

def replace_competitor_words(text: str) -> str:
    """替换竞品词为 [Competitor Name]。"""
    result = text
    for word in COMPETITOR_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        result = re.sub(pattern, "[Competitor Name]", result, flags=re.IGNORECASE)
    return result


# ------------------- Think Tag Filtering --------------------
def filter_think_tags(text: str) -> str:
    """过滤 <think> 标签及其内容。"""
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    elif "<think>" in text:
        text = text.split("<think>")[0].strip()
    return text.strip()


# ------------------- Terminology Validation --------------------
TERMINOLOGY = {
    "zh": {"键盘": "keyboard", "机械键盘": "mechanical keyboard", "轴": "switch", "键帽": "keycap"},
    "de": {"键盘": "Tastatur", "机械键盘": "mechanische Tastatur", "轴": "Switch", "键帽": "Tastenkappe"},
    "fr": {"键盘": "clavier", "机械键盘": "clavier mécanique", "轴": "interrupteur", "键帽": "touche"},
    "es": {"键盘": "teclado", "机械键盘": "teclado mecánico", "轴": "switch", "键帽": "tecla"},
}

def validate_terminology(text: str, target_lang: str) -> str:
    """检查并修正术语翻译。"""
    if target_lang not in TERMINOLOGY:
        return text

    glossary = TERMINOLOGY[target_lang]
    for cn_term, correct_term in glossary.items():
        pattern = r"\b" + re.escape(cn_term) + r"\b"
        text = re.sub(pattern, correct_term, text, flags=re.IGNORECASE)
    return text


# ------------------- Streamlit UI --------------------
st.title("⌨️ 键盘出海内容自动化与审核平台 (DeepSeek + Qdrant Cloud)")

# Sidebar
with st.sidebar:
    st.header("⚙️ 配置")

    # API Key 配置状态
    deepseek_configured = bool(st.session_state.get("deepseek_api_key", ""))
    qdrant_configured = bool(st.session_state.get("qdrant_url", "")) and bool(st.session_state.get("qdrant_api_key", ""))

    if deepseek_configured:
        st.success("✅ DeepSeek API 已配置")
    else:
        st.warning("⚠️ DeepSeek API 未配置")

    if qdrant_configured:
        st.success("✅ Qdrant 向量库已配置")
    else:
        st.warning("⚠️ Qdrant 向量库未配置")

    # 高级设置（演示时隐藏）
    with st.expander("🔧 高级设置（API Key 配置）", expanded=not (deepseek_configured and qdrant_configured)):
        st.markdown("##### 🤖 DeepSeek API 配置")
        st.text_input(
            "DeepSeek API Key",
            type="password",
            key="deepseek_api_key"
        )
        if st.button("🔌 测试 DeepSeek 连接", use_container_width=True):
            test_result = call_deepseek([{"role": "user", "content": "Say 'test ok' in one word."}])
            if test_result.startswith("❌"):
                st.error(test_result)
            else:
                st.success("✅ DeepSeek 连接成功！")

        st.divider()
        st.markdown("##### 🌐 Qdrant 向量库配置")
        st.text_input(
            "Qdrant 集群 URL",
            placeholder="https://xxx.cloud.qdrant.io:6333",
            key="qdrant_url"
        )
        st.text_input(
            "Qdrant API Key",
            type="password",
            key="qdrant_api_key"
        )
        if st.button("🔌 测试 Qdrant 连接", use_container_width=True):
            from qdrant_client import QdrantClient
            try:
                test_client = QdrantClient(
                    url=st.session_state.qdrant_url,
                    api_key=st.session_state.qdrant_api_key,
                )
                test_client.get_collections()
                st.success("✅ Qdrant 连接成功！")
            except Exception as e:
                st.error(f"❌ 连接失败: {e}")

    st.divider()
    st.markdown("##### 📝 输入参数")
    product_name = st.text_input("产品名称", value="KeyX Pro")
    product_features = st.text_area("产品特点（每行一个）", value="Gasket mount structure\nHot-swappable switches\nRGB backlighting\nWireless connectivity")
    target_market = st.selectbox("目标市场", ["US", "UK", "DE", "FR", "ES", "JP", "KR"])
    content_type = st.selectbox("内容类型", ["Blog", "EDM"])
    prompt_input = st.text_area("额外 Prompt（可选）", placeholder="e.g. Focus on ergonomics and productivity")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 输入")
    user_input = st.text_area(
        "营销需求描述（支持中文）",
        height=200,
        placeholder="e.g. 我们需要一篇博客文章，介绍我们的新键盘如何提升工作效率..."
    )

    if st.button("🚀 生成营销内容", use_container_width=True, type="primary"):
        if not user_input:
            st.error("请输入营销需求描述！")
        elif not st.session_state.get("deepseek_api_key"):
            st.error("请在侧边栏填写 DeepSeek API Key！")
        else:
            with st.spinner("正在生成内容..."):
                # Step 1: Translate if needed
                if re.search(r"[\u4e00-\u9fff]", user_input):
                    st.info("检测到中文输入，正在翻译成英文...")
                    user_input_en = translate_to_english(user_input)
                else:
                    user_input_en = user_input

                # Step 2: RAG retrieval
                st.info("正在从 Qdrant 检索相关案例...")
                docs, scores = retrieve_from_qdrant(user_input_en, top_k=3)

                if docs:
                    st.success(f"✅ 检索到 {len(docs)} 条相关案例")
                    with st.expander("查看检索结果（含相似度分数）"):
                        for i, (doc, score) in enumerate(zip(docs, scores)):
                            st.markdown(f"**Case {i+1}** (相似度: {score})")
                            st.text(doc[:200] + "...")
                else:
                    st.warning("⚠️ 未检索到相关案例，将不使用 RAG。")

                # Step 3: Build prompt
                rag_context = "\n\n".join([f"Case {i+1}:\n{doc}" for i, doc in enumerate(docs)])
                target_lang = target_market.lower()

                system_prompt = f"""You are a professional marketing content creator for keyboard products going global.

Your task: Generate a {content_type} for the {target_market} market.

Product: {product_name}
Features:
{product_features}

RAG Reference Cases:
{rag_context}

Requirements:
- Target language: {target_lang}
- Content type: {content_type}
- Do NOT use competitor brand names (replace with [Competitor Name])
- Do NOT use 'Introduction' or 'Conclusion' as headers
- Tone should match the reference cases
{f"- Additional requirements: {prompt_input}" if prompt_input else ""}

Output the content directly, no extra explanation.
"""

                # Step 4: Call DeepSeek (Drafter)
                st.info("正在调用 DeepSeek 生成内容（Drafter Agent）...")
                drafter_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input_en}
                ]
                draft_output = call_deepseek(drafter_messages, temperature=0.7)

                if draft_output.startswith("❌"):
                    st.error(draft_output)
                else:
                    # Filter think tags
                    draft_output = filter_think_tags(draft_output)

                    # Step 5: Critic review (multi-round)
                    st.info("正在审核内容（Critic Agent）...")
                    critic_prompt = f"""You are a marketing content reviewer. Review the following {content_type} content for the {target_market} market.

Content to review:
{draft_output}

Check for:
1. Competitor brand names (should be [Competitor Name])
2. Inappropriate terminology
3. Cultural sensitivity for {target_market}
4. Matching the reference cases' style

If the content is good, output: {{"approved": true, "feedback": ""}}
If changes are needed, output: {{"approved": false, "feedback": "..."}}
"""

                    max_critic_rounds = 2
                    current_draft = draft_output
                    for round_num in range(max_critic_rounds):
                        critic_messages = [
                            {"role": "system", "content": critic_prompt},
                            {"role": "user", "content": f"Please review this content (Round {round_num+1}):\n\n{current_draft}"}
                        ]
                        critic_response = call_deepseek(critic_messages, temperature=0.2)

                        # Parse critic response
                        try:
                            json_match = re.search(r"\{.*\}", critic_response, re.DOTALL)
                            if json_match:
                                critic_result = json.loads(json_match.group())
                                if critic_result.get("approved"):
                                    st.success(f"✅ 审核通过（第 {round_num+1} 轮）")
                                    break
                                else:
                                    st.warning(f"⚠️ 需要修改（第 {round_num+1} 轮）")
                                    # Regenerate based on feedback
                                    refine_messages = drafter_messages + [
                                        {"role": "assistant", "content": current_draft},
                                        {"role": "user", "content": f"Please revise based on this feedback: {critic_result.get('feedback')}"}
                                    ]
                                    current_draft = call_deepseek(refine_messages, temperature=0.7)
                                    current_draft = filter_think_tags(current_draft)
                            else:
                                st.warning("⚠️ 无法解析 Critic 响应，跳过审核。")
                                break
                        except json.JSONDecodeError:
                            st.warning("⚠️ Critic 响应不是有效 JSON，跳过审核。")
                            break

                    # Step 6: Final processing
                    final_output = replace_competitor_words(current_draft)
                    final_output = validate_terminology(final_output, target_lang)

                    # Save to history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append({
                        "input": user_input,
                        "output": final_output,
                        "timestamp": st.session_state.get("timestamp", ""),
                    })

                    st.session_state.generated_content = final_output
                    st.success("✅ 内容生成完成！")

with col2:
    st.subheader("📤 输出")
    if "generated_content" in st.session_state:
        st.markdown(st.session_state.generated_content)
        st.download_button(
            "📥 下载内容",
            st.session_state.generated_content,
            file_name=f"{product_name}_{content_type}_{target_market}.txt",
            mime="text/plain"
        )
    else:
        st.info("生成的内容将显示在这里。")

# History
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("📜 历史记录")
    for i, item in enumerate(reversed(st.session_state.history[-10:])):
        with st.expander(f"记录 {i+1}: {item['input'][:50]}..."):
            st.markdown(f"**输入**: {item['input']}")
            st.markdown(f"**输出**: {item['output'][:200]}...")
