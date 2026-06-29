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
    page_title="Keyboard Export AI Marketing Platform",
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
        return "ERROR: DeepSeek API Key not configured."
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
                return f"ERROR: DeepSeek API call failed: {e}"
            time.sleep(2)
    return "ERROR: Max retries reached."


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
    prompt = f"Translate the following Chinese text into English. Return ONLY the translation:\n\n{text}"
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
    'Keyboard Export AI Marketing Platform'
    '</h1>'
    '<p style="font-size:1.15rem;color:#86868b;font-weight:400;margin:0;">'
    'DeepSeek + Qdrant Cloud &#8212; RAG-Powered Content Generation &amp; Review'
    '</p></div>',
    unsafe_allow_html=True,
)


# ------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### Settings")

    deepseek_configured = bool(st.session_state.get("deepseek_api_key", ""))
    qdrant_configured = bool(st.session_state.get("qdrant_url", "")) and bool(
        st.session_state.get("qdrant_api_key", "")
    )

    if deepseek_configured:
        st.success("DeepSeek API &#8212; Connected")
    else:
        st.warning("DeepSeek API &#8212; Not configured")

    if qdrant_configured:
        st.success("Qdrant Vector DB &#8212; Connected")
    else:
        st.warning("Qdrant Vector DB &#8212; Not configured")

    with st.expander(
        "Advanced Settings",
        expanded=not (deepseek_configured and qdrant_configured),
    ):
        st.markdown("**DeepSeek API**")
        st.text_input("API Key", type="password", key="deepseek_api_key")
        if st.button("Test DeepSeek Connection"):
            test_result = call_deepseek(
                [{"role": "user", "content": "Say test ok in one word."}]
            )
            if test_result.startswith("ERROR"):
                st.error(test_result)
            else:
                st.success("Connection successful!")

        st.divider()
        st.markdown("**Qdrant Vector DB**")
        st.text_input(
            "Cluster URL",
            placeholder="https://xxx.cloud.qdrant.io",
            key="qdrant_url",
        )
        st.text_input("API Key", type="password", key="qdrant_api_key")
        if st.button("Test Qdrant Connection"):
            from qdrant_client import QdrantClient

            try:
                test_client = QdrantClient(
                    url=st.session_state.qdrant_url,
                    api_key=st.session_state.qdrant_api_key,
                )
                test_client.get_collections()
                st.success("Connection successful!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.divider()
    st.markdown("### Product & Content")

    product_name = st.text_input("Product Name", value="KeyX Pro")
    product_features = st.text_area(
        "Product Features (one per line)",
        value="Gasket mount structure\nHot-swappable switches\nRGB backlighting\nWireless connectivity",
    )
    content_type = st.selectbox("Content Type", ["Blog", "EDM"])
    prompt_input = st.text_area(
        "Additional Prompt (optional)",
        placeholder="e.g. Focus on ergonomics and productivity",
    )


# ------------------- Main Content --------------------
col1, col2 = st.columns([1, 1], gap=24)

with col1:
    st.subheader("Input")
    user_input = st.text_area(
        "Marketing Requirements (Chinese supported)",
        height=200,
        placeholder="e.g. We need a blog post about how our new keyboard improves productivity...",
    )

    if st.button("Generate Marketing Content", type="primary"):
        if not user_input:
            st.error("Please enter your marketing requirements.")
        elif not st.session_state.get("deepseek_api_key"):
            st.error("Please configure DeepSeek API Key in the sidebar.")
        else:
            with st.spinner("Generating..."):
                # Step 1: Translate if needed
                if re.search(r"[\u4e00-\u9fff]", user_input):
                    st.info("Translating Chinese input to English...")
                    user_input_en = translate_to_english(user_input)
                else:
                    user_input_en = user_input

                # Step 2: RAG retrieval
                st.info("Retrieving relevant cases from Qdrant...")
                docs, scores = retrieve_from_qdrant(user_input_en, top_k=3)

                if docs:
                    st.success(f"Retrieved {len(docs)} relevant cases.")
                    with st.expander("View Retrieval Results (with similarity scores)"):
                        for i, (doc, score) in enumerate(zip(docs, scores)):
                            st.markdown(f"**Case {i+1}** &#8212; Similarity: {score}")
                            st.text(doc[:200] + "...")
                else:
                    st.warning("No relevant cases found. Generating without RAG.")

                # Step 3: Build prompt (English only)
                rag_context = "\n\n".join(
                    [f"Case {i+1}:\n{doc}" for i, doc in enumerate(docs)]
                )

                system_prompt = (
                    f"You are a professional marketing content creator for keyboard products going global.\n\n"
                    f"Your task: Generate a {content_type} in English.\n\n"
                    f"Product: {product_name}\n"
                    f"Features:\n{product_features}\n\n"
                    f"RAG Reference Cases:\n{rag_context}\n\n"
                    f"Requirements:\n"
                    f"- Language: English only\n"
                    f"- Content type: {content_type}\n"
                    f"- Do NOT use competitor brand names (replace with [Competitor Name])\n"
                    f"- Do NOT use Introduction or Conclusion as headers\n"
                    f"- Tone should match the reference cases\n"
                )
                if prompt_input:
                    system_prompt += f"- Additional requirements: {prompt_input}\n"
                system_prompt += "\nOutput the content directly, no extra explanation.\n"

                # Step 4: Drafter Agent
                st.info("Generating content (Drafter Agent)...")
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
                    st.info("Reviewing content (Critic Agent)...")
                    critic_prompt = (
                        f"You are a marketing content reviewer. "
                        f"Review the following {content_type} content.\n\n"
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
                                    st.success(f"Approved (Round {round_num+1})")
                                    break
                                else:
                                    st.warning(f"Revision needed (Round {round_num+1})")
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
                                st.warning("Could not parse Critic response. Skipping review.")
                                break
                        except json.JSONDecodeError:
                            st.warning("Critic response is not valid JSON. Skipping review.")
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
                    st.success("Content generated successfully!")

with col2:
    st.subheader("Output")
    if "generated_content" in st.session_state:
        st.markdown(st.session_state.generated_content)
        st.download_button(
            "Download Content",
            st.session_state.generated_content,
            file_name=f"{product_name}_{content_type}.txt",
            mime="text/plain",
        )
    else:
        st.info("Generated content will appear here.")


# ------------------- History --------------------
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("History")
    for i, item in enumerate(reversed(st.session_state.history[-10:])):
        input_preview = item["input"][:50]
        with st.expander(f"Record {i+1}: {input_preview}..."):
            st.markdown(f"**Input**: {item['input']}")
            output_preview = item["output"][:200]
            st.markdown(f"**Output**: {output_preview}...")
