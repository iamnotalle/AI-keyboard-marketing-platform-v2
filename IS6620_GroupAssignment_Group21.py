import json
import os
import re
import time
from dataclasses import dataclass
from html import escape
from typing import Any

import streamlit as st

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled in UI
    OpenAI = None

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - handled in UI
    QdrantClient = None

try:
    from fastembed import TextEmbedding
except Exception:  # pragma: no cover - handled in UI
    TextEmbedding = None


APP_TITLE = "键盘出海营销内容生成器"
CASE_COLLECTION_NAME = "marketing_cases"
KNOWLEDGE_COLLECTION_NAME = "marketing_knowledge_base"

COMPETITOR_WORDS = [
    "Cherry",
    "Keychron",
    "Razer",
    "Logitech",
    "Corsair",
    "Ducky",
    "HHKB",
    "Leopold",
    "Akko",
    "Glorious",
    "SteelSeries",
    "HyperX",
    "Durgod",
    "Varmilo",
    "Filco",
]

DEMO_CASES = [
    {
        "title": "效率场景上市文案",
        "content": (
            "A high-performing keyboard launch story should connect tactile comfort "
            "with everyday productivity. Strong examples usually open with a work "
            "pain point, show how the typing experience reduces friction, then close "
            "with a clear trial or purchase call to action."
        ),
        "score": 0.91,
    },
    {
        "title": "创作者桌面 EDM",
        "content": (
            "Campaigns for creators work best when they balance specs with identity. "
            "Wireless setup, hot-swappable switches, and RGB should be framed as ways "
            "to keep the desk clean, flexible, and expressive."
        ),
        "score": 0.86,
    },
]

EVALUATION_DIMENSIONS = [
    "需求匹配",
    "品牌安全",
    "合规风险",
    "事实一致性",
    "CTA 清晰度",
    "内容完整度",
    "参考依据",
]

COMPLIANCE_RISK_PATTERNS = {
    "绝对化表达": [
        "best",
        "perfect",
        "ultimate",
        "guaranteed",
        "100%",
        "zero lag",
        "no lag",
        "risk-free",
    ],
    "无法证明的效果承诺": [
        "double your productivity",
        "triple your productivity",
        "increase productivity by",
        "eliminate fatigue",
        "cure",
        "prevent pain",
    ],
    "合规敏感词": [
        "medical grade",
        "FDA approved",
        "clinically proven",
    ],
}


@dataclass
class Brief:
    product_name: str
    platform: str
    content_type: str
    goal: str
    market: str
    audience: str
    tone: str
    length: str
    structure: str
    include_title: bool
    include_cta: bool
    compliance_check: bool
    features: str
    must_include: str
    additional_prompt: str
    avoid_terms: str
    user_need: str


def read_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, default)


def read_bool_secret(name: str, default: bool = False) -> bool:
    raw = read_secret(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def init_state() -> None:
    defaults = {
        "deepseek_api_key": read_secret("DEEPSEEK_API_KEY"),
        "qdrant_url": read_secret("QDRANT_URL"),
        "qdrant_api_key": read_secret("QDRANT_API_KEY"),
        "interview_mode": read_bool_secret("INTERVIEW_MODE", True),
        "show_api_settings": read_bool_secret("SHOW_API_SETTINGS", False),
        "history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.set_page_config(page_title=APP_TITLE, page_icon="⌨️", layout="wide")
init_state()

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    .stAppDeployButton { display: none !important; }
    .main .block-container { max-width: 1220px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    section[data-testid="stSidebar"] { background: #f7f8fa; border-right: 1px solid #e6e8ec; }
    .pm-hero {
        border-bottom: 1px solid #e6e8ec;
        padding: 0.2rem 0 1.1rem 0;
        margin-bottom: 1rem;
    }
    .pm-kicker {
        color: #5b6472;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }
    .pm-title {
        color: #111827;
        font-size: 2.1rem;
        font-weight: 760;
        line-height: 1.15;
        margin: 0;
    }
    .pm-subtitle {
        color: #4b5563;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-top: 0.55rem;
        max-width: 920px;
    }
    .summary-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.35rem 0 0.9rem 0;
    }
    .summary-card {
        background: #ffffff;
        border: 1px solid #e6e8ec;
        border-radius: 8px;
        padding: 0.72rem 0.85rem;
        min-height: 82px;
    }
    .summary-label {
        color: #6b7280;
        font-size: 0.78rem;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .summary-value {
        color: #111827;
        font-weight: 720;
        font-size: 1.22rem;
        line-height: 1.25;
        white-space: nowrap;
    }
    .summary-note {
        color: #6b7280;
        font-size: 0.75rem;
        line-height: 1.25;
        margin-top: 0.2rem;
    }
    .flow-step {
        background: #ffffff;
        border: 1px solid #e6e8ec;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 88px;
    }
    .flow-step strong { color: #111827; }
    .flow-step span { color: #4b5563; font-size: 0.9rem; }
    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
        min-height: 2.7rem;
    }
    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_deepseek_client(api_key: str):
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_deepseek(messages: list[dict[str, str]], api_key: str, temperature: float = 0.35) -> tuple[bool, str]:
    client = get_deepseek_client(api_key)
    if client is None:
        return False, "AI 服务尚未配置。"

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
            )
            return True, filter_think_tags(response.choices[0].message.content or "")
        except Exception as exc:
            if attempt == 1:
                return False, f"AI 生成失败：{exc}"
            time.sleep(1.2)
    return False, "AI 生成失败。"


@st.cache_resource(show_spinner=False)
def get_qdrant_resources(url: str, api_key: str):
    if not url or not api_key or QdrantClient is None:
        return None, None
    client = QdrantClient(url=url, api_key=api_key)
    embedding_model = None
    if TextEmbedding is not None:
        try:
            embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception:
            embedding_model = None
    return client, embedding_model


def payload_to_case(payload: dict[str, Any], score: float, fallback_title: str = "参考案例") -> dict[str, Any]:
    case_type = str(payload.get("type", "")).strip()
    style = str(payload.get("style", "")).strip()
    title = (
        payload.get("title")
        or payload.get("dimension")
        or " · ".join(item for item in [case_type, style] if item)
        or fallback_title
    )
    content = payload.get("content") or payload.get("text") or payload.get("body") or ""
    if not content:
        content = json.dumps(payload, ensure_ascii=False)
    return {"title": str(title), "content": str(content), "score": round(float(score), 4)}


def lexical_case_score(query: str, case: dict[str, Any], content_type: str) -> float:
    query_tokens = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    case_text = f"{case['title']} {case['content']}".lower()
    overlap = sum(1 for token in query_tokens if token in case_text)
    type_bonus = 2 if content_type.lower() and content_type.lower() in case_text else 0
    return overlap + type_bonus + min(len(case["content"]) / 1000, 1)


def scroll_qdrant_cases(
    client: Any,
    collection_name: str,
    query: str,
    content_type: str,
    top_k: int,
) -> list[dict[str, Any]]:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=80,
        with_payload=True,
        with_vectors=False,
    )
    ranked_cases = []
    for index, point in enumerate(points):
        payload = point.payload or {}
        payload_type = str(payload.get("type", "")).strip().lower()
        if payload_type and content_type and payload_type != content_type.lower():
            continue
        case = payload_to_case(payload, 0, fallback_title=f"Qdrant 案例 {index + 1}")
        case["score"] = lexical_case_score(query, case, content_type)
        ranked_cases.append(case)

    ranked_cases.sort(key=lambda item: item["score"], reverse=True)
    if not ranked_cases:
        return []

    best_score = max(case["score"] for case in ranked_cases) or 1
    selected = []
    for case in ranked_cases[:top_k]:
        normalized_score = max(0.55, min(0.98, case["score"] / best_score))
        selected.append({**case, "score": round(normalized_score, 4)})
    return selected


def merge_references(*groups: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for group in groups:
        for item in group:
            key = f"{item.get('title', '')}|{item.get('content', '')[:80]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def retrieve_cases(
    query: str,
    url: str,
    api_key: str,
    demo_mode: bool,
    content_type: str,
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], str]:
    if demo_mode:
        return DEMO_CASES[:top_k], "内置参考库"

    client, embedding_model = get_qdrant_resources(url, api_key)
    if client is None:
        return DEMO_CASES[:top_k], "内置参考库"

    vector_cases = []
    if embedding_model is not None:
        try:
            query_embedding = list(embedding_model.embed([query]))[0].tolist()
            results = client.query_points(
                collection_name=CASE_COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
            )
            vector_cases = [payload_to_case(hit.payload or {}, hit.score) for hit in results.points]
        except Exception:
            pass

    lexical_cases = []
    knowledge_cases = []
    try:
        lexical_cases = scroll_qdrant_cases(client, CASE_COLLECTION_NAME, query, content_type, top_k)
    except Exception:
        pass

    try:
        knowledge_cases = scroll_qdrant_cases(client, KNOWLEDGE_COLLECTION_NAME, query, content_type, 3)
    except Exception:
        pass

    combined = merge_references(vector_cases, lexical_cases, knowledge_cases, limit=6)
    if combined:
        if vector_cases and knowledge_cases:
            return combined, "Qdrant Cloud 向量+知识库检索"
        if lexical_cases and knowledge_cases:
            return combined, "Qdrant Cloud 案例+知识库检索"
        if vector_cases:
            return combined, "Qdrant Cloud 向量检索"
        return combined, "Qdrant Cloud 案例检索"

    return DEMO_CASES[:top_k], "内置参考库"


def filter_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def replace_competitor_words(text: str, avoid_terms: str = "") -> str:
    terms = COMPETITOR_WORDS + [term.strip() for term in avoid_terms.split(",") if term.strip()]
    result = text
    for word in terms:
        pattern = r"\b" + re.escape(word) + r"\b"
        result = re.sub(pattern, "[Competitor Name]", result, flags=re.IGNORECASE)
    return result


def prompt_value(value: str) -> str:
    mapping = {
        "曝光": "Awareness",
        "种草/考虑": "Consideration",
        "转化": "Conversion",
        "专业": "Professional",
        "有活力": "Energetic",
        "高端": "Premium",
        "友好": "Friendly",
        "技术向": "Technical",
        "短文案": "Short",
        "标准": "Standard",
        "长文案": "Long",
        "痛点-卖点-CTA": "Pain point - benefit - CTA",
        "AIDA": "AIDA",
        "PAS": "Problem - Agitate - Solution",
        "对比型": "Comparison-driven",
        "清单型": "Listicle",
    }
    return mapping.get(value, value)


def brief_to_prompt(brief: Brief, cases: list[dict[str, Any]]) -> str:
    references = "\n\n".join(
        f"Reference {index + 1}: {case['title']}\n{case['content']}"
        for index, case in enumerate(cases)
    )
    return f"""
You are a senior marketing content strategist for a keyboard brand going global.

Create one polished {prompt_value(brief.content_type)} for the {brief.platform} channel.

Product: {brief.product_name}
Market: {brief.market}
Audience: {brief.audience}
Campaign goal: {prompt_value(brief.goal)}
Tone: {prompt_value(brief.tone)}
Length: {prompt_value(brief.length)}
Structure: {prompt_value(brief.structure)}
Core features:
{brief.features}

Must include:
{brief.must_include or "No extra required points."}

Additional direction:
{brief.additional_prompt or "No extra direction."}

Avoid:
{brief.avoid_terms or "Competitor names, exaggerated claims, and unsupported promises."}

User brief:
{brief.user_need}

Reference cases from knowledge base:
{references}

Output requirements:
- Write in English.
- Use a practical marketing structure: hook, value proof, product benefit, and CTA.
- Do not mention competitor brand names.
- Make the content usable by a marketing operator without extra explanation.
- Do not include analysis notes.
- Include a title: {"yes" if brief.include_title else "no"}
- Include a clear CTA: {"yes" if brief.include_cta else "no"}
- Run compliance and brand-safety check before finalizing: {"yes" if brief.compliance_check else "no"}
- Avoid absolute or unverifiable claims such as guaranteed results, medical benefits, or 100% performance promises.
- If the content type is EDM, include a short unsubscribe or preference-management footer.
- If the content type is Blog, do not include email footer language such as "unsubscribe", "manage preferences", or "you received this email".
""".strip()


def build_demo_content(brief: Brief, cases: list[dict[str, Any]]) -> str:
    feature_lines = [line.strip() for line in brief.features.splitlines() if line.strip()]
    first_feature = feature_lines[0] if feature_lines else "a smoother typing experience"
    second_feature = feature_lines[1] if len(feature_lines) > 1 else "a cleaner desk setup"
    cta = "Try it in your next setup" if brief.goal in {"Awareness", "曝光"} else "Shop the new setup today"
    title_prefix = f"{brief.product_name}: " if brief.include_title else ""
    cta_line = f"\n\n{cta}." if brief.include_cta else ""

    if brief.content_type == "EDM":
        subject = f"Subject: {title_prefix}Turn every keystroke into a better workday\n\n" if brief.include_title else ""
        return f"""{subject}Hi there,

Meet {brief.product_name}, a keyboard built for {brief.audience.lower()} who want speed, comfort, and a desk that feels ready for deep work.

With {first_feature} and {second_feature}, it helps you move from idea to output with fewer distractions. The typing feel is stable, the setup is flexible, and the experience is easy to make your own.

{brief.must_include}
{cta_line}
"""

    title = f"{title_prefix}A Better Keyboard for the Way Modern Teams Work\n\n" if brief.include_title else ""
    return f"""{title}The best desk setup is not just about how it looks. It is about how quickly people can get into flow, stay focused, and move through the workday without friction.

{brief.product_name} is built for {brief.audience.lower()} in {brief.market}. Its {first_feature} gives each keystroke a stable feel, while {second_feature} supports a cleaner and more flexible setup. For users who switch between writing, planning, design, and communication, that combination turns the keyboard from a basic accessory into a daily productivity tool.

The product story is simple: better feel, fewer interruptions, and more confidence at the desk. {brief.must_include}
{cta_line or f"If your team is refreshing its work setup, {brief.product_name} is a practical place to start."}
"""


def generate_content(brief: Brief, cases: list[dict[str, Any]], api_key: str, demo_mode: bool) -> tuple[str, str]:
    prompt = brief_to_prompt(brief, cases)
    if demo_mode or not api_key:
        return replace_competitor_words(build_demo_content(brief, cases), brief.avoid_terms), "Demo Drafter"

    ok, output = call_deepseek(
        [
            {"role": "system", "content": "You create concise, conversion-aware marketing content."},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        temperature=0.65,
    )
    if not ok:
        st.warning(output)
        return replace_competitor_words(build_demo_content(brief, cases), brief.avoid_terms), "Demo fallback after API error"
    return replace_competitor_words(output, brief.avoid_terms), "DeepSeek Drafter"


def score_dimension(score: int, issue: str, suggestion: str) -> dict[str, Any]:
    return {"score": score, "issue": issue, "suggestion": suggestion}


def evaluate_content(content: str, brief: Brief, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lower_content = content.lower()
    issues = {}

    feature_hits = sum(
        1
        for line in brief.features.splitlines()
        if line.strip() and line.strip().lower().split()[0] in lower_content
    )
    if feature_hits >= 2:
        issues["需求匹配"] = score_dimension(9, "核心产品卖点已经体现。", "保留当前 Brief 结构，后续活动可以复用。")
    elif feature_hits == 1:
        issues["需求匹配"] = score_dimension(7, "只覆盖了部分核心卖点。", "为每个核心卖点补充一个明确的用户收益。")
    else:
        issues["需求匹配"] = score_dimension(5, "内容偏泛，产品细节不足。", "补充更具体的产品信息，并把功能转成用户收益。")

    banned_hits = [
        word
        for word in COMPETITOR_WORDS
        if re.search(r"\b" + re.escape(word) + r"\b", content, flags=re.IGNORECASE)
    ]
    if banned_hits:
        issues["品牌安全"] = score_dimension(4, f"检测到竞品词：{', '.join(banned_hits)}。", "发布前替换竞品词，并加入禁用词列表。")
    else:
        issues["品牌安全"] = score_dimension(9, "未检测到竞品品牌名。", "继续保留禁用词规则，降低误发风险。")

    compliance_hits = []
    for category, patterns in COMPLIANCE_RISK_PATTERNS.items():
        matched_patterns = [pattern for pattern in patterns if pattern in lower_content]
        if matched_patterns:
            compliance_hits.append(f"{category}: {', '.join(matched_patterns)}")

    edm_compliance_missing = (
        brief.content_type == "EDM"
        and not any(term in lower_content for term in ["unsubscribe", "opt out", "manage preferences"])
    )
    if compliance_hits:
        issues["合规风险"] = score_dimension(
            5,
            f"发现可能需要法务/合规复核的表达：{'; '.join(compliance_hits)}。",
            "把绝对化、疗效化或无法证明的承诺改成可验证、可解释的产品收益。",
        )
    elif edm_compliance_missing:
        issues["合规风险"] = score_dimension(
            6,
            "EDM 缺少退订或偏好管理提示。",
            "在邮件底部补充 unsubscribe / manage preferences 等合规入口。",
        )
    else:
        issues["合规风险"] = score_dimension(
            9,
            "未发现明显夸大承诺、敏感功效或 EDM 基础合规缺口。",
            "上线前仍建议按目标市场邮件法规和品牌法务清单做最终复核。",
        )

    unsupported_spec_patterns = [
        r"\b\d+\s*(?:mah|hours?|hrs?|ms|g|kg|million|m)\b",
        r"\b(?:waterproof|ip[0-9]{2}|bluetooth\s*[0-9.]+|2\.4ghz)\b",
        r"\b(?:free shipping|30-day|lifetime warranty|certified|award-winning)\b",
    ]
    unsupported_hits = []
    for pattern in unsupported_spec_patterns:
        unsupported_hits.extend(re.findall(pattern, lower_content, flags=re.IGNORECASE))
    if unsupported_hits:
        issues["事实一致性"] = score_dimension(
            6,
            f"出现可能需要产品资料确认的具体信息：{', '.join(sorted(set(unsupported_hits)))}。",
            "只保留 Brief 或产品事实库中已提供的信息；具体参数、保修、优惠和认证需补充来源后再发布。",
        )
    elif brief.product_name.lower() in lower_content:
        issues["事实一致性"] = score_dimension(
            9,
            "未发现明显编造参数，且产品名称使用一致。",
            "继续用产品事实库约束具体规格，避免临时生成未确认参数。",
        )
    else:
        issues["事实一致性"] = score_dimension(
            7,
            "未发现明显编造参数，但产品名称露出不够稳定。",
            "在标题、正文或 CTA 中明确产品名称，保证文案可直接用于该产品。",
        )

    cta_terms = ["shop", "try", "learn", "discover", "order", "start", "click", "buy"]
    if any(term in lower_content for term in cta_terms):
        issues["CTA 清晰度"] = score_dimension(8, "文案包含明确的下一步动作。", "如果营销目标变化，可以把 CTA 调整得更贴合渠道。")
    else:
        issues["CTA 清晰度"] = score_dimension(5, "下一步动作不够明确。", "增加一个与营销目标相关的明确 CTA。")

    blog_has_email_footer = brief.content_type == "Blog" and any(
        term in lower_content for term in ["unsubscribe", "manage preferences", "you received this email"]
    )
    if blog_has_email_footer:
        issues["内容完整度"] = score_dimension(
            6,
            "Blog 中混入了 EDM 页脚或退订语句。",
            "删除邮件页脚，保留 Blog 的标题、正文结构和 CTA。",
        )
    elif len(content.split()) < 80 and brief.content_type == "Blog":
        issues["内容完整度"] = score_dimension(6, "当前内容对该格式来说偏短。", "补充证明点、使用场景和卖点解释。")
    elif len(content.split()) > 260 and brief.content_type == "EDM":
        issues["内容完整度"] = score_dimension(6, "当前内容对该格式来说偏长。", "压缩篇幅，保留最关键的卖点和 CTA。")
    else:
        issues["内容完整度"] = score_dimension(8, "篇幅基本符合当前内容格式。", "继续按内容格式控制长度。")

    if cases:
        issues["参考依据"] = score_dimension(8, "已结合参考案例生成。", "保留最强的参考角度，删掉支撑较弱的表达。")
    else:
        issues["参考依据"] = score_dimension(5, "缺少参考案例支撑。", "发布前补充品牌案例或历史活动文案。")

    return issues


def parse_agent_scorecard(raw_text: str, fallback: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not json_match:
        return fallback

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        return fallback

    raw_scorecard = parsed.get("scorecard", parsed)
    normalized: dict[str, dict[str, Any]] = {}

    if isinstance(raw_scorecard, list):
        iterable = [
            (
                str(item.get("dimension", "")),
                item,
            )
            for item in raw_scorecard
            if isinstance(item, dict)
        ]
    elif isinstance(raw_scorecard, dict):
        iterable = list(raw_scorecard.items())
    else:
        return fallback

    for name, detail in iterable:
        if name not in EVALUATION_DIMENSIONS or not isinstance(detail, dict):
            continue
        try:
            score = int(round(float(detail.get("score", 0))))
        except (TypeError, ValueError):
            score = fallback.get(name, {}).get("score", 6)
        score = max(1, min(10, score))
        issue = str(detail.get("issue") or fallback.get(name, {}).get("issue", "需要进一步审核。"))
        suggestion = str(
            detail.get("suggestion") or fallback.get(name, {}).get("suggestion", "基于该维度继续优化。")
        )
        normalized[name] = score_dimension(score, issue, suggestion)

    if not normalized:
        return fallback

    return {
        name: normalized.get(name, fallback[name])
        for name in EVALUATION_DIMENSIONS
        if name in normalized or name in fallback
    }


def review_content_with_agent(
    content: str,
    brief: Brief,
    cases: list[dict[str, Any]],
    api_key: str,
    demo_mode: bool,
) -> tuple[dict[str, dict[str, Any]], str]:
    fallback = evaluate_content(content, brief, cases)
    if demo_mode or not api_key:
        return fallback, "规则审核 Agent"

    dimensions = "\n".join(f"- {name}" for name in EVALUATION_DIMENSIONS)
    prompt = f"""
You are a strict marketing content review agent.
Evaluate the generated content for a product manager demo.

Return ONLY valid JSON. Do not use markdown code fences.
Use these exact Chinese dimension names:
{dimensions}

JSON schema:
{{
  "scorecard": {{
    "需求匹配": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}},
    "品牌安全": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}},
    "合规风险": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}},
    "事实一致性": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}},
    "CTA 清晰度": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}},
    "内容完整度": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}},
    "参考依据": {{"score": 1-10, "issue": "中文问题判断", "suggestion": "中文迭代建议"}}
  }}
}}

Product: {brief.product_name}
Content type: {brief.content_type}
Channel: {brief.platform}
Goal: {brief.goal}
Audience: {brief.audience}
Features:
{brief.features}

Reference case count: {len(cases)}

Compliance review checklist:
- Check absolute or unverifiable claims, such as guaranteed results, best/perfect/100%, zero lag, medical or pain-prevention claims.
- Check competitor or trademark misuse separately under 品牌安全.
- Check whether the content invents unsupported specs, discounts, warranty promises, certifications or numerical claims under 事实一致性.
- For EDM, check unsubscribe / opt-out / preference-management footer expectations.
- This is a marketing-risk review, not legal advice; give practical rewrite suggestions.

Generated content:
{content}
""".strip()

    ok, output = call_deepseek(
        [
            {"role": "system", "content": "You are a precise content audit agent that returns valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        temperature=0.2,
    )
    if not ok:
        return fallback, "规则审核 Agent"

    return parse_agent_scorecard(output, fallback), "DeepSeek 审核 Agent"


def summarize_score(scorecard: dict[str, dict[str, Any]]) -> tuple[int, str]:
    if not scorecard:
        return 0, "Not reviewed"
    score = round(sum(item["score"] for item in scorecard.values()) / len(scorecard))
    if score >= 8:
        return score, "轻微修改后可用"
    if score >= 6:
        return score, "建议再迭代一轮"
    return score, "需要重点修改"


def build_iteration_plan(scorecard: dict[str, dict[str, Any]]) -> list[str]:
    sorted_items = sorted(scorecard.items(), key=lambda item: item[1]["score"])
    return [f"{name}: {detail['suggestion']}" for name, detail in sorted_items[:3]]


def render_output_summary(result: dict[str, Any] | None) -> None:
    if result:
        score, status_label = summarize_score(result["scorecard"])
        cards = [
            ("引用案例", f"{len(result['cases'])} 条", result.get("retrieval_source", "参考库")),
            ("审核维度", f"{len(result['scorecard'])} 项", result.get("reviewer_source", "审核 Agent")),
            ("综合评分", f"{score}/10", status_label),
        ]
    else:
        cards = [
            ("引用案例", "待检索", "生成后显示"),
            ("审核维度", f"{len(EVALUATION_DIMENSIONS)} 项", "固定审核 Agent"),
            ("综合评分", "待评分", "生成后显示"),
        ]

    card_html = "".join(
        '<div class="summary-card">'
        f'<div class="summary-label">{escape(label)}</div>'
        f'<div class="summary-value">{escape(value)}</div>'
        f'<div class="summary-note">{escape(note)}</div>'
        "</div>"
        for label, value, note in cards
    )
    st.markdown(f'<div class="summary-row">{card_html}</div>', unsafe_allow_html=True)


def save_history(brief: Brief, content: str, score: int, status: str) -> None:
    st.session_state.history.append(
        {
            "product": brief.product_name,
            "platform": brief.platform,
            "content_type": brief.content_type,
            "score": score,
            "status": status,
            "content": content,
        }
    )
    st.session_state.history = st.session_state.history[-8:]


with st.sidebar:
    has_live_key = bool(st.session_state.deepseek_api_key)
    show_api_settings = bool(st.session_state.show_api_settings)

    if show_api_settings:
        demo_mode = st.toggle("演示模式", value=not has_live_key, label_visibility="collapsed")

        with st.expander("API 设置", expanded=not demo_mode):
            st.session_state.deepseek_api_key = st.text_input(
                "DeepSeek API Key",
                value=st.session_state.deepseek_api_key,
                type="password",
                help="也可以通过 Streamlit Secrets 配置。",
            )
            st.session_state.qdrant_url = st.text_input(
                "Qdrant URL",
                value=st.session_state.qdrant_url,
                help="可选。未配置时会使用内置参考库。",
            )
            st.session_state.qdrant_api_key = st.text_input(
                "Qdrant API Key",
                value=st.session_state.qdrant_api_key,
                type="password",
            )
    else:
        demo_mode = not has_live_key

    st.subheader("产品与内容")
    product_name = st.text_input("产品名称", value="KeyX Pro")
    features = st.text_area(
        "产品特性（每行一个）",
        value="Gasket mount structure\nHot-swappable switches\nRGB backlighting\nWireless connectivity",
        height=140,
    )
    content_type_options = ["Blog", "EDM"]
    if "pending_template_type" in st.session_state:
        st.session_state.content_type = st.session_state.pop("pending_template_type")
    if "pending_user_need" in st.session_state:
        st.session_state.user_need = st.session_state.pop("pending_user_need")
    if st.session_state.get("content_type") not in content_type_options:
        st.session_state.content_type = "Blog"
    content_type = st.selectbox("内容类型", content_type_options, key="content_type")
    additional_prompt = st.text_area(
        "附加提示词（可选）",
        placeholder="例如：重点突出人体工学和生产力提升",
        height=140,
    )


st.markdown(
    """
    <div class="pm-hero" style="text-align:center;padding-top:3rem;">
        <h1 class="pm-title">键盘出海 AI 营销平台</h1>
        <div class="pm-subtitle" style="margin-left:auto;margin-right:auto;">
            DeepSeek + Qdrant Cloud — RAG 驱动的内容生成与审核
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 生成设置")
setting_col1, setting_col2, setting_col3, setting_col4 = st.columns(4)
with setting_col1:
    goal = st.selectbox("营销目标", ["曝光", "种草/考虑", "转化"], index=1)
with setting_col2:
    tone = st.selectbox("文案语气", ["专业", "有活力", "高端", "友好", "技术向"], index=1)
with setting_col3:
    length = st.selectbox("内容长度", ["短文案", "标准", "长文案"], index=1)
with setting_col4:
    structure = st.selectbox("内容结构", ["痛点-卖点-CTA", "AIDA", "PAS", "对比型", "清单型"], index=0)

include_title = True
include_cta = True
compliance_check = True
avoid_terms = st.text_input("避免出现的词", value="Cherry, Keychron, Razer")

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("输入")
    if "user_need" not in st.session_state:
        st.session_state.user_need = ""

    template_col1, template_col2 = st.columns(2)
    with template_col1:
        if st.button("Blog 模板", use_container_width=True):
            st.session_state.pending_template_type = "Blog"
            st.session_state.pending_user_need = "我们需要一篇 Blog，介绍新键盘如何提升远程办公用户的工作效率。"
            st.rerun()
    with template_col2:
        if st.button("EDM 模板", use_container_width=True):
            st.session_state.pending_template_type = "EDM"
            st.session_state.pending_user_need = "我们需要一封 EDM，向海外用户推广新品键盘，突出舒适手感和无线桌面。"
            st.rerun()

    user_input = st.text_area(
        "营销需求（支持中文输入）",
        key="user_need",
        height=260,
        placeholder="例如：我们需要一篇 Blog，介绍新键盘如何提升工作效率...",
    )

    platform = "Email" if content_type == "EDM" else "Blog"
    brief = Brief(
        product_name=product_name,
        platform=platform,
        content_type=content_type,
        goal=goal,
        market="海外键盘用户",
        audience="键盘爱好者和办公用户",
        tone=tone,
        length=length,
        structure=structure,
        include_title=include_title,
        include_cta=include_cta,
        compliance_check=compliance_check,
        features=features,
        must_include=additional_prompt,
        additional_prompt=additional_prompt,
        avoid_terms=avoid_terms,
        user_need=user_input,
    )

    run = st.button("生成营销内容", type="primary", use_container_width=True)

    if run:
        if not product_name.strip():
            st.error("请先填写产品名称。")
        elif not features.strip():
            st.error("请至少填写一个产品特性。")
        elif not user_input.strip():
            st.error("请先输入营销需求。")
        else:
            with st.status("正在生成...", expanded=True) as status:
                st.write("检索参考案例")
                retrieval_query = f"{brief.user_need}\n{brief.product_name}\n{brief.content_type}\n{brief.features}"
                cases, retrieval_source = retrieve_cases(
                    retrieval_query,
                    st.session_state.qdrant_url,
                    st.session_state.qdrant_api_key,
                    demo_mode=demo_mode,
                    content_type=brief.content_type,
                )

                st.write("生成内容")
                content, drafter_source = generate_content(
                    brief,
                    cases,
                    st.session_state.deepseek_api_key,
                    demo_mode=demo_mode,
                )

                st.write("审核 Agent 评估内容")
                scorecard, reviewer_source = review_content_with_agent(
                    content,
                    brief,
                    cases,
                    st.session_state.deepseek_api_key,
                    demo_mode=demo_mode,
                )
                score, status_label = summarize_score(scorecard)
                iteration_plan = build_iteration_plan(scorecard)
                status.update(label="生成完成", state="complete", expanded=False)

            st.session_state.result = {
                "brief": brief,
                "content": content,
                "cases": cases,
                "retrieval_source": retrieval_source,
                "drafter_source": drafter_source,
                "reviewer_source": reviewer_source,
                "scorecard": scorecard,
                "score": score,
                "status": status_label,
                "iteration_plan": iteration_plan,
            }
            save_history(brief, content, score, status_label)

with right:
    st.subheader("输出")
    result = st.session_state.get("result")
    render_output_summary(result)
    if not result:
        st.info("生成的内容将显示在此处。")
    else:
        visible_scorecard = result["scorecard"]
        st.caption(f"由 {result.get('reviewer_source', '审核 Agent')} 给出，用于判断是否需要继续迭代。")
        for name, detail in visible_scorecard.items():
            st.progress(detail["score"] / 10, text=f"{name}: {detail['score']}/10")
            st.caption(detail["issue"])

        iteration_plan = build_iteration_plan(visible_scorecard)
        if iteration_plan:
            with st.expander("迭代建议", expanded=True):
                for index, suggestion in enumerate(iteration_plan, start=1):
                    st.write(f"{index}. {suggestion}")

        st.divider()
        st.markdown(result["content"])
        st.download_button(
            "下载内容",
            data=result["content"],
            file_name=f"{result['brief'].product_name}_{result['brief'].content_type}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        with st.expander("查看审核结果与参考案例"):
            st.caption(
                f"生成来源：{result.get('drafter_source', '内容生成 Agent')}；"
                f"审核来源：{result.get('reviewer_source', '审核 Agent')}；"
                f"参考来源：{result.get('retrieval_source', '参考库')}"
            )
            for name, detail in result["scorecard"].items():
                st.progress(detail["score"] / 10, text=f"{name}: {detail['score']}/10")
                st.caption(detail["issue"])
            st.markdown("**参考案例**")
            for index, case in enumerate(result["cases"], start=1):
                st.write(f"{index}. {case['title']} · 匹配度 {case['score']}")

if st.session_state.history:
    with st.expander("历史记录", expanded=False):
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['product']} · {item['content_type']} · {item['score']}/10 · {item['status']}**")
            st.write(item["content"][:400] + ("..." if len(item["content"]) > 400 else ""))
