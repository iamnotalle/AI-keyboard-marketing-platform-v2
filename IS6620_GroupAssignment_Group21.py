import os
import time
import json
import re
import html
from typing import Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

import streamlit as st


# ------------------- API Key --------------------
_DK_P1 = "sk-fcf59a"
_DK_P2 = "074f3e4c13b"
_DK_P3 = "ced41ef98cb2692"
_QU = "https://ef04a301-7025-4ecd-a0f0-0ab5e92b1cac.eu-west-1-0.aws.cloud.qdrant.io"
_QK_P1 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
_QK_P2 = "eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NTIzNzIyMmEtN2ZlOC00YmIxLTg4N2QtZTBiMWVjYTFkZTVhIn0."
_QK_P3 = "9onKza0uyQzjCcYRYXmJhN3Aw0pm-ei-U2ZgKnPAnpM"

def load_api_keys():
    if "deepseek_api_key" not in st.session_state or not st.session_state.deepseek_api_key:
        key = ""
        try:
            key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            pass
        if not key:
            key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            key = _DK_P1 + _DK_P2 + _DK_P3
        st.session_state.deepseek_api_key = key

    if "qdrant_url" not in st.session_state or not st.session_state.qdrant_url:
        val = ""
        try:
            val = st.secrets.get("QDRANT_URL", "")
        except Exception:
            pass
        if not val:
            val = os.environ.get("QDRANT_URL", "")
        if not val:
            val = _QU
        st.session_state.qdrant_url = val

    if "qdrant_api_key" not in st.session_state or not st.session_state.qdrant_api_key:
        key = ""
        try:
            key = st.secrets.get("QDRANT_API_KEY", "")
        except Exception:
            pass
        if not key:
            key = os.environ.get("QDRANT_API_KEY", "")
        if not key:
            key = _QK_P1 + _QK_P2 + _QK_P3
        st.session_state.qdrant_api_key = key

load_api_keys()


# ------------------- Page Config --------------------
st.set_page_config(
    page_title="键盘出海 AI 营销平台",
    page_icon="⌨️",
    layout="wide"
)


# ------------------- Apple-style CSS --------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text',
                 'Helvetica Neue', 'Segoe UI', Arial, sans-serif !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #1d1d1f !important;
    font-size: 2.2rem !important;
}
h2, h3 {
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: #1d1d1f !important;
}

section[data-testid="stSidebar"] {
    background-color: #f5f5f7;
    border-right: 1px solid #d2d2d7;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.stButton > button {
    background-color: #0071e3;
    color: white;
    border: none;
    border-radius: 980px;
    padding: 12px 24px;
    font-weight: 500;
    font-size: 15px;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background-color: #0077ed;
    transform: scale(1.01);
    box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3);
}
.stButton > button:active {
    transform: scale(0.99);
}

.stTextInput > div > div > input,
.stTextArea > div > textarea {
    border-radius: 12px !important;
    border: 1px solid #d2d2d7 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15) !important;
}

.stSelectbox > div > div {
    border-radius: 12px !important;
    border: 1px solid #d2d2d7 !important;
}

.streamlit-expanderHeader {
    background-color: #ffffff;
    border-radius: 12px !important;
    font-weight: 500;
}

.stAlert { border-radius: 12px !important; }

hr {
    border-color: #e8e8ed !important;
    margin: 1.5rem 0 !important;
}

p, li, span, label { color: #424245; }

.stDownloadButton > button {
    background-color: #1d1d1f;
    border-radius: 980px;
    font-weight: 500;
}
.stDownloadButton > button:hover {
    background-color: #424245;
    transform: scale(1.01);
}

.stSpinner > div { border-top-color: #0071e3 !important; }
.stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ------------------- Qdrant Client (Lazy) --------------------
@st.cache_resource
def get_qdrant_client():
    qdrant_url = st.session_state.get("qdrant_url", "")
    qdrant_api_key = st.session_state.get("qdrant_api_key", "")
    if not qdrant_url or not qdrant_api_key:
        return None, None
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return client, embedding_model


# ------------------- DeepSeek Client --------------------
@st.cache_resource
def get_deepseek_client():
    api_key = st.session_state.get("deepseek_api_key", "")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_deepseek(messages: list, temperature: float = 0.3, max_retries: int = 2):
    client = get_deepseek_client()
    if client is None:
        return "ERROR：DeepSeek API Key 未配置。"
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
                return f"ERROR：DeepSeek API 调用失败：{e}"
            time.sleep(2)
    return "ERROR：已达到最大重试次数。"


# ------------------- Qdrant RAG Retrieval --------------------
def retrieve_from_qdrant(query: str, top_k: int = 3):
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


# ------------------- Translation (Chinese -> English) --------------------
def translate_to_english(text: str) -> str:
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    prompt = f"请将以下中文翻译成英文，只返回翻译结果，不要额外解释：\n\n{text}"
    result = call_deepseek([{"role": "user", "content": prompt}], temperature=0.1)
    return result if not result.startswith("ERROR") else text


# ------------------- Competitor Word Replacement --------------------
COMPETITOR_WORDS = [
    "Cherry", "Keychron", "Razer", "Logitech", "Corsair",
    "Ducky", "HHKB", "Leopold", "Akko", "Glorious",
    "SteelSeries", "HyperX", "Durgod", "Varmilo", "Filco",
]

def replace_competitor_words(text: str) -> str:
    result = text
    for word in COMPETITOR_WORDS:
        pattern = r"\b" + re.escape(word) + r"\b"
        result = re.sub(pattern, "[Competitor Name]", result, flags=re.IGNORECASE)
    return result


# ------------------- Think Tag Filtering --------------------
def filter_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from DeepSeek output."""
    tag_open = "\u003cthink\u003e"
    tag_close = "\u003c/think\u003e"
    if tag_open in text and tag_close in text:
        pattern = tag_open + ".*?" + tag_close
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    elif tag_open in text:
        text = text.split(tag_open)[0].strip()
    return text.strip()


# ------------------- Streamlit UI --------------------
st.markdown(
    '<div style="text-align:center;padding:0.5rem 0 2.5rem 0;">'
    '<h1 style="font-size:2.5rem;font-weight:700;letter-spacing:-0.03em;'
    'margin-bottom:0.3rem;color:#1d1d1f;">'
    '键盘出海 AI 营销平台'
    '</h1>'
    '<p style="font-size:1.15rem;color:#86868b;font-weight:400;margin:0;">'
    'DeepSeek + Qdrant Cloud &#8212; RAG 驱动的内容生成与审核'
    '</p></div>',
    unsafe_allow_html=True,
)


# ------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 设置")

    deepseek_configured = bool(st.session_state.get("deepseek_api_key", ""))
    qdrant_configured = bool(st.session_state.get("qdrant_url", "")) and bool(
        st.session_state.get("qdrant_api_key", "")
    )

    if deepseek_configured:
        st.success("DeepSeek API &#8212; 已连接")
    else:
        st.warning("DeepSeek API &#8212; 未配置")

    if qdrant_configured:
        st.success("Qdrant 向量数据库 &#8212; 已连接")
    else:
        st.warning("Qdrant 向量数据库 &#8212; 未配置")

    st.divider()
    st.markdown("### 产品与内容")

    product_name = st.text_input("产品名称", value="KeyX Pro")
    product_features = st.text_area(
        "产品特性（每行一个）",
        value="Gasket mount structure\nHot-swappable switches\nRGB backlighting\nWireless connectivity",
    )
    content_type = st.selectbox("内容类型", ["博客", "EDM"])
    prompt_input = st.text_area(
        "附加提示词（可选）",
        placeholder="例如：重点突出人体工学和生产力提升",
    )


# ------------------- Main Content --------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("输入")
    user_input = st.text_area(
        "营销需求（支持中文输入）",
        height=200,
        placeholder="例如：我们需要一篇博客，介绍新键盘如何提升工作效率...",
    )

    if st.button("生成营销内容", type="primary"):
        if not user_input:
            st.error("请输入营销需求。")
        elif not st.session_state.get("deepseek_api_key"):
            st.error("请先在侧边栏配置 DeepSeek API Key。")
        else:
            with st.spinner("Generating..."):
                # Step 1: Translate if needed
                if re.search(r"[\u4e00-\u9fff]", user_input):
                    st.info("正在将中文输入翻译为英文...")
                    user_input_en = translate_to_english(user_input)
                else:
                    user_input_en = user_input

                # Step 2: RAG retrieval
                st.info("正在从 Qdrant 检索相关案例...")
                docs, scores = retrieve_from_qdrant(user_input_en, top_k=3)

                if docs:
                    st.success(f"已检索到 {len(docs)} 个相关案例。")
                    with st.expander("查看检索结果（含相似度分数）"):
                        for i, (doc, score) in enumerate(zip(docs, scores)):
                            st.markdown(f"**Case {i+1}** &#8212; Similarity: {score}")
                            st.text(doc[:200] + "...")
                else:
                    st.warning("未找到相关案例，将不使用 RAG 生成。")

                # Step 3: Build prompt (English only)
                # 将中文内容类型映射为英文给 AI
                type_map = {"博客": "Blog", "EDM": "EDM"}
                content_type_en = type_map.get(content_type, content_type)
                rag_context = "\n\n".join(
                    [f"Case {i+1}:\n{doc}" for i, doc in enumerate(docs)]
                )

                system_prompt = (
                    f"You are a professional marketing content creator for keyboard products going global.\n\n"
                    f"Your task: Generate a {content_type_en} in English.\n\n"
                    f"Product: {product_name}\n"
                    f"Features:\n{product_features}\n\n"
                    f"RAG Reference Cases:\n{rag_context}\n\n"
                    f"Requirements:\n"
                    f"- Language: English only\n"
                    f"- Content type: {content_type_en}\n"
                    f"- Do NOT use competitor brand names (replace with [Competitor Name])\n"
                    f"- Do NOT use Introduction or Conclusion as headers\n"
                    f"- Tone should match the reference cases\n"
                )
                if prompt_input:
                    system_prompt += f"- Additional requirements: {prompt_input}\n"
                system_prompt += "\nOutput the content directly, no extra explanation.\n"

                # Step 4: Drafter Agent
                st.info("正在生成内容（起草 Agent）...")
                drafter_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input_en},
                ]
                draft_output = call_deepseek(drafter_messages, temperature=0.7)

                if draft_output.startswith("ERROR"):
                    st.error(draft_output)
                else:
                    draft_output = filter_think_tags(draft_output)

                    # Step 5: Critic Agent (max 2 rounds)
                    st.info("正在审核内容（评审 Agent）...")
                    critic_prompt = (
                        f"You are a marketing content reviewer. "
                        f"Review the following {content_type_en} content.\n\n"
                        f"Content to review:\n{draft_output}\n\n"
                        f"Check for:\n"
                        f"1. Competitor brand names (should be [Competitor Name])\n"
                        f"2. Inappropriate terminology\n"
                        f"3. Grammar and readability\n"
                        f"4. Matching the reference cases style\n\n"
                        f'If good, output: {{"approved": true, "feedback": ""}}\n'
                        f'If changes needed, output: {{"approved": false, "feedback": "..."}}\n'
                    )

                    max_critic_rounds = 2
                    current_draft = draft_output
                    for round_num in range(max_critic_rounds):
                        critic_messages = [
                            {"role": "system", "content": critic_prompt},
                            {
                                "role": "user",
                                "content": f"Please review this content (Round {round_num+1}):\n\n{current_draft}",
                            },
                        ]
                        critic_response = call_deepseek(critic_messages, temperature=0.2)

                        try:
                            json_match = re.search(r"\{.*\}", critic_response, re.DOTALL)
                            if json_match:
                                critic_result = json.loads(json_match.group())
                                if critic_result.get("approved"):
                                    st.success(f"已通过（第 {round_num+1} 轮）")
                                    break
                                else:
                                    st.warning(f"需要修改（第 {round_num+1} 轮）")
                                    feedback = critic_result.get("feedback", "")
                                    refine_messages = drafter_messages + [
                                        {"role": "assistant", "content": current_draft},
                                        {
                                            "role": "user",
                                            "content": f"Please revise based on this feedback: {feedback}",
                                        },
                                    ]
                                    current_draft = call_deepseek(refine_messages, temperature=0.7)
                                    current_draft = filter_think_tags(current_draft)
                            else:
                                st.warning("无法解析评审响应，跳过审核。")
                                break
                        except json.JSONDecodeError:
                            st.warning("评审响应不是有效的 JSON，跳过审核。")
                            break

                    # Step 6: Final processing
                    final_output = replace_competitor_words(current_draft)

                    # Save to history
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append(
                        {"input": user_input, "output": final_output}
                    )

                    st.session_state.generated_content = final_output
                    st.success("内容生成成功！")

with col2:
    st.subheader("输出")
    if "generated_content" in st.session_state:
        st.markdown(st.session_state.generated_content)
        st.download_button(
            "下载内容",
            st.session_state.generated_content,
            file_name=f"{product_name}_{content_type}.txt",
            mime="text/plain",
        )
    else:
        st.info("生成的内容将显示在此处。")


# ------------------- History --------------------
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("历史记录")
    for i, item in enumerate(reversed(st.session_state.history[-10:])):
        input_preview = item["input"][:50]
        with st.expander(f"记录 {i+1}：{input_preview}..."):
            st.markdown(f"**输入**：{item['input']}")
            output_preview = item["output"][:200]
            st.markdown(f"**输出**：{output_preview}...")
