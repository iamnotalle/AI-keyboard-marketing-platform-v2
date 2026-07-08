import json
import os
import re
import time
from dataclasses import dataclass
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
    {
        "title": "远程办公场景 Blog",
        "content": (
            "Remote-work content should translate keyboard features into daily workflow "
            "benefits. Start from long typing sessions and desk clutter, then explain "
            "how a softer mount, flexible switches, and wireless connectivity support "
            "a calmer and more adaptable work setup."
        ),
        "score": 0.84,
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

OUTPUT_VIEWS = ["生成内容", "引用案例", "审核维度", "综合评分"]
REFERENCE_FEEDBACK_OPTIONS = ["采用", "不采用", "不相关"]
REFERENCE_FEEDBACK_SCORE = {"采用": 5, "不采用": 3, "不相关": 1}
LOW_SCORE_THRESHOLD = 7

QUALITY_GAP_ACTIONS = {
    "需求匹配": {
        "material": "补充目标用户、使用场景、核心卖点优先级，以及每个功能对应的用户收益。",
        "system_action": "下次生成时优先把产品特性转成场景化收益，并检查是否覆盖 Brief 中的核心诉求。",
    },
    "品牌安全": {
        "material": "补充品牌禁用词、竞品替代表达、品牌语气边界和允许使用的术语。",
        "system_action": "下次审核时扩大品牌安全词表，并在改稿阶段自动替换高风险竞品或不合适表达。",
    },
    "合规风险": {
        "material": "补充目标市场邮件合规要求、绝对化表达清单、医疗化/功效化禁用表达。",
        "system_action": "下次生成前先压低夸大承诺，改稿时把高风险表达替换成可验证、保守的产品收益。",
    },
    "事实一致性": {
        "material": "补充真实产品参数、认证、保修、价格、优惠、用户评价等可追溯来源。",
        "system_action": "下次生成时禁止新增未在 Brief 或知识库出现的参数、认证、保修和用户数据。",
    },
    "CTA 清晰度": {
        "material": "补充不同营销目标对应的 CTA 示例，例如了解更多、加入等待名单、立即购买或查看产品页。",
        "system_action": "下次按营销目标自动选择 CTA 强度，并检查正文结尾是否有明确下一步动作。",
    },
    "内容完整度": {
        "material": "补充 Blog / EDM 的结构模板、最低信息要求、标题/正文/CTA 示例和渠道长度标准。",
        "system_action": "下次按内容类型检查结构完整性，Blog 补足论证，EDM 控制篇幅并保留核心卖点。",
    },
    "参考依据": {
        "material": "补充更贴近该 Brief 的历史案例、产品事实、品牌规则和可检索标签。",
        "system_action": "下次优先优化 Qdrant 检索标签和 Top 3 引用质量，降低不相关案例进入生成上下文的概率。",
    },
}

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
        "reference_feedback": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.set_page_config(page_title=APP_TITLE, page_icon="⌨️", layout="wide", initial_sidebar_state="expanded")
init_state()

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] {
        background: transparent;
    }
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


def format_scorecard_for_prompt(scorecard: dict[str, dict[str, Any]]) -> str:
    if not scorecard:
        return "No review result."
    lines = []
    for name in EVALUATION_DIMENSIONS:
        detail = scorecard.get(name)
        if not detail:
            continue
        lines.append(
            f"- {name}: {detail['score']}/10 | 问题: {detail['issue']} | 修改建议: {detail['suggestion']}"
        )
    return "\n".join(lines)


def apply_rule_based_revision(
    draft_content: str,
    brief: Brief,
    draft_scorecard: dict[str, dict[str, Any]],
) -> str:
    revised = replace_competitor_words(draft_content, brief.avoid_terms)

    if brief.product_name.lower() not in revised.lower():
        revised = f"{brief.product_name}\n\n{revised}"

    lower_revised = revised.lower()
    if brief.content_type == "EDM" and not any(
        term in lower_revised for term in ["unsubscribe", "opt out", "manage preferences"]
    ):
        revised = revised.rstrip() + "\n\nYou can unsubscribe or manage preferences at any time."

    cta_terms = ["shop", "try", "learn", "discover", "order", "start", "click", "buy"]
    if brief.include_cta and not any(term in revised.lower() for term in cta_terms):
        revised = revised.rstrip() + f"\n\nLearn more about {brief.product_name} and find the setup that fits your day."

    lowest_items = sorted(draft_scorecard.items(), key=lambda item: item[1]["score"])[:2]
    if lowest_items and brief.content_type == "Blog":
        benefit_lines = [
            line.strip()
            for line in brief.features.splitlines()
            if line.strip() and line.strip().lower().split()[0] not in revised.lower()
        ][:2]
        if benefit_lines:
            revised = (
                revised.rstrip()
                + "\n\nIt also gives users practical upgrade options such as "
                + "; ".join(benefit_lines)
                + ", so the setup can adapt as their work style changes."
            )

    return clean_final_content(revised, brief)


def clean_final_content(content: str, brief: Brief) -> str:
    cleaned = replace_competitor_words(content, brief.avoid_terms)
    cleaned = re.sub(
        r"(?is)\n\s*(?:#+\s*)?(?:\*\*)?What users say(?:\*\*)?.*?(?=\n\s*(?:#+\s*)?(?:\*\*)?[A-Z][^\n]*(?:\*\*)?\n|$)",
        "\n",
        cleaned,
    )
    cleaned = re.sub(r"(?is)\s*Available with[^.\n]*(?:warranty|retailer|retailers)[^.\n]*\.", "", cleaned)
    cleaned = re.sub(
        r"(?im)^.*(?:warranty|free shipping|certified|award-winning|FDA approved|medical grade|clinically proven).*$\n?",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?im)^.*[\"“].*[\"”]\s*[—-].*(?:user|customer|worker).*$\n?", "", cleaned)
    replacements = {
        r"\bsignificantly enhances your experience\b": "can support a cleaner, more comfortable workflow",
        r"\bstable,\s*low-latency wireless connectivity\b": "flexible wireless connectivity",
        r"\blow-latency\b": "responsive",
        r"\bwithout worrying about lag or interruptions\b": "with fewer setup distractions",
        r"\bduring video calls or data transfers\b": "while moving between everyday tasks",
        r"\bensures smooth performance\b": "keeps the setup simple",
        r"\bnever lose your train of thought\b": "can stay closer to your flow",
        r"\breduce hand strain\b": "stay more comfortable",
        r"\bNo fatigue\.\s*No clutter\.\s*No friction\.": "Less clutter. Smoother switching. A calmer desk.",
        r"\bNo fatigue\b": "Less distraction",
        r"\bwork longer\b": "work with more comfort",
    }
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bperfect\b", "preferred", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def revise_content_with_agent(
    draft_content: str,
    brief: Brief,
    cases: list[dict[str, Any]],
    draft_scorecard: dict[str, dict[str, Any]],
    api_key: str,
    demo_mode: bool,
) -> tuple[str, str]:
    if demo_mode or not api_key:
        return apply_rule_based_revision(draft_content, brief, draft_scorecard), "规则迭代 Agent"

    prompt = f"""
You are a marketing content revision agent.
Revise the draft after the first review.

Return ONLY the revised final marketing content.
Do not include review notes, explanations, markdown fences, or scoring.

Product: {brief.product_name}
Content type: {brief.content_type}
Channel: {brief.platform}
Goal: {brief.goal}
Tone: {brief.tone}
Length: {brief.length}
Structure: {brief.structure}
Features:
{brief.features}

Must include:
{brief.must_include or "No extra required points."}

Avoid:
{brief.avoid_terms or "Competitor names, exaggerated claims, and unsupported promises."}

First review result:
{format_scorecard_for_prompt(draft_scorecard)}

Revision requirements:
- Fix the lowest-scoring review issues first.
- Preserve the useful parts of the draft.
- Keep the final output ready to use by a marketing operator.
- Do not mention competitor names.
- Avoid absolute, medical, warranty, discount, certification, or numerical claims that are not provided in the brief.
- Do not invent testimonials, user quotes, review durations, retailer availability, warranties, awards, certifications, or pricing offers.
- If the content type is EDM, include a short unsubscribe or preference-management footer.
- If the content type is Blog, do not include email footer language.

Draft content:
{draft_content}
""".strip()

    ok, output = call_deepseek(
        [
            {"role": "system", "content": "You revise marketing content based on audit feedback."},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        temperature=0.45,
    )
    if not ok:
        st.warning(output)
        return apply_rule_based_revision(draft_content, brief, draft_scorecard), "规则迭代 Agent"
    return clean_final_content(output, brief), "DeepSeek 迭代 Agent"


def score_dimension(score: int, issue: str, suggestion: str) -> dict[str, Any]:
    return {"score": score, "issue": issue, "suggestion": suggestion}


def clean_review_suggestion(name: str, suggestion: str) -> str:
    unsafe_patterns = [
        r"\b\d+\s*%",
        r"\b\d+\s*ms\b",
        r"<\s*\d+",
        r"用户测试",
        r"用户反馈",
        r"具体延迟",
        r"认证",
        r"warranty",
        r"保修",
        r"FDA",
        r"clinically",
        r"free shipping",
    ]
    if not any(re.search(pattern, suggestion, flags=re.IGNORECASE) for pattern in unsafe_patterns):
        return suggestion

    safe_defaults = {
        "事实一致性": "删除未在 Brief 或事实库出现的延迟、认证、保修、优惠、用户数据；如确需使用，先补充真实来源。",
        "合规风险": "把疲劳、疼痛、疗效或绝对化表达改成更稳妥的舒适体验描述，并避免量化承诺。",
        "参考依据": "只引用 Qdrant 已检索案例或产品事实库中的内容，不要补充未验证的用户数据。",
        "需求匹配": "围绕已提供产品特性补充用户收益，不要新增未经确认的参数或效果数据。",
    }
    return safe_defaults.get(name, "基于已提供 Brief 和知识库改写，不要新增未经验证的参数、认证或用户数据。")


def sanitize_scorecard(scorecard: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sanitized: dict[str, dict[str, Any]] = {}
    for name, detail in scorecard.items():
        sanitized[name] = {
            **detail,
            "suggestion": clean_review_suggestion(name, str(detail.get("suggestion", ""))),
        }
    return sanitized


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

    return sanitize_scorecard(issues)


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

    return sanitize_scorecard({
        name: normalized.get(name, fallback[name])
        for name in EVALUATION_DIMENSIONS
        if name in normalized or name in fallback
    })


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
- Do not suggest adding fake numbers, fake user tests, fake certifications, fake warranties, or fake testimonials.
- If evidence is missing, suggest deleting the claim or adding a real source to the knowledge base before publishing.

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


def build_human_review_notes(
    brief: Brief,
    final_scorecard: dict[str, dict[str, Any]],
    final_score: int,
) -> list[str]:
    notes: list[str] = []
    if final_score >= 8:
        notes.append("整体可以进入人工终审，重点确认产品参数、卖点表述和投放地区合规要求。")
    elif final_score >= 6:
        notes.append("建议人工复核后再发布，优先处理低分维度对应的表达风险。")
    else:
        notes.append("不建议直接发布，需要人工重审 Brief、产品事实和合规边界后再改稿。")

    low_dimensions = [
        name
        for name, detail in final_scorecard.items()
        if int(detail.get("score", 0)) <= 6
    ]
    if low_dimensions:
        notes.append(f"人工重点看：{', '.join(low_dimensions)}。")

    compliance_score = final_scorecard.get("合规风险", {}).get("score", 10)
    factual_score = final_scorecard.get("事实一致性", {}).get("score", 10)
    if compliance_score <= 7 or factual_score <= 7:
        notes.append("涉及合规或事实一致性的内容，发布前需要用产品资料、法务清单或地区邮件规则再确认。")

    if brief.content_type == "EDM":
        notes.append("EDM 发布前确认退订入口、发件人信息、目标市场邮件合规要求和最终落地页链接。")
    else:
        notes.append("Blog 发布前确认 SEO 标题、正文小标题、图片/链接素材和产品事实来源。")

    return notes[:4]


def select_output_view() -> str:
    if st.session_state.get("output_view") not in OUTPUT_VIEWS:
        st.session_state.output_view = OUTPUT_VIEWS[0]

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "输出视图",
            OUTPUT_VIEWS,
            key="output_view",
            label_visibility="collapsed",
        )
        return selected or OUTPUT_VIEWS[0]

    selected = st.radio(
        "输出视图",
        OUTPUT_VIEWS,
        key="output_view",
        horizontal=True,
        label_visibility="collapsed",
    )
    return selected or OUTPUT_VIEWS[0]


def render_empty_output(view: str) -> None:
    if view == "生成内容":
        st.info("填写左侧 Brief 后，点击“生成营销内容”，这里会显示二审后的最终稿。")
    elif view == "引用案例":
        st.info("生成后显示本次从 Qdrant / 案例库检索到的 Top 3 引用案例，并支持标记采用、不采用或不相关。")
    elif view == "审核维度":
        st.caption("审核维度由系统固定；生成后显示一审和终审结果。")
        for dimension in EVALUATION_DIMENSIONS:
            st.write(f"- {dimension}")
    elif view == "综合评分":
        st.info("生成后显示终审综合评分、迭代建议和人审意见。")


def scorecard_score(scorecard: dict[str, dict[str, Any]], name: str) -> int:
    try:
        return int(scorecard.get(name, {}).get("score", 0))
    except (TypeError, ValueError):
        return 0


def lowest_score_items(scorecard: dict[str, dict[str, Any]], limit: int = 3) -> list[tuple[str, dict[str, Any]]]:
    return sorted(scorecard.items(), key=lambda item: int(item[1].get("score", 0)))[:limit]


def build_score_improvements(
    draft_scorecard: dict[str, dict[str, Any]],
    final_scorecard: dict[str, dict[str, Any]],
) -> list[str]:
    improvements = []
    for name in EVALUATION_DIMENSIONS:
        draft_score = scorecard_score(draft_scorecard, name)
        final_score = scorecard_score(final_scorecard, name)
        if draft_score and final_score and final_score > draft_score:
            improvements.append(f"{name}: {draft_score} → {final_score}")
    return improvements


def render_revision_comparison(result: dict[str, Any]) -> None:
    draft_content = result.get("draft_content")
    if not draft_content:
        return

    result_id = result.get("result_id", "current")
    draft_scorecard = result.get("draft_scorecard", {})
    final_scorecard = result.get("scorecard", {})
    draft_score, draft_status = summarize_score(draft_scorecard)
    final_score, final_status = summarize_score(final_scorecard)
    improvements = build_score_improvements(draft_scorecard, final_scorecard)

    st.divider()
    st.markdown("**改前改后对比**")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("一审草稿", f"{draft_score}/10")
        st.caption(draft_status)
    with metric_col2:
        st.metric("终审稿", f"{final_score}/10")
        st.caption(final_status)
    with metric_col3:
        st.metric("提升维度", f"{len(improvements)} 个")
        st.caption("对比一审与终审评分")

    issue_col, suggestion_col = st.columns(2)
    with issue_col:
        st.markdown("**初稿问题**")
        for name, detail in lowest_score_items(draft_scorecard):
            st.write(f"- {name} {detail['score']}/10：{detail['issue']}")

    with suggestion_col:
        st.markdown("**一审建议**")
        for name, detail in lowest_score_items(draft_scorecard):
            st.write(f"- {name}：{detail['suggestion']}")

    if improvements:
        st.markdown("**本轮提升维度**")
        st.write("；".join(improvements))
    else:
        st.caption("本轮终审维度保持稳定，改稿重点是清理风险表达并提升可发布性。")

    with st.expander("查看初稿与修改后内容", expanded=False):
        before_col, after_col = st.columns(2)
        with before_col:
            st.text_area(
                "初稿",
                value=draft_content,
                height=360,
                disabled=True,
                key=f"draft_compare_{result_id}",
            )
        with after_col:
            st.text_area(
                "修改后内容",
                value=result.get("content", ""),
                height=360,
                disabled=True,
                key=f"final_compare_{result_id}",
            )


def render_generated_content(result: dict[str, Any]) -> None:
    brief = result["brief"]
    st.caption(
        f"最终稿来源：{result.get('revision_source', '迭代 Agent')}；"
        f"内容类型：{brief.content_type}"
    )
    st.markdown(result["content"])
    st.download_button(
        "下载内容",
        data=result["content"],
        file_name=f"{brief.product_name}_{brief.content_type}.txt",
        mime="text/plain",
        use_container_width=True,
        key="download_generated_content",
    )
    render_revision_comparison(result)


def reference_feedback_widget_key(result_id: str, index: int) -> str:
    return f"reference_feedback_{result_id}_{index}"


def build_reference_feedback_record(
    result: dict[str, Any],
    index: int,
    case: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    brief = result["brief"]
    return {
        "result_id": result.get("result_id", "current"),
        "content_type": brief.content_type,
        "product_name": brief.product_name,
        "reference_rank": index,
        "reference_title": case.get("title", "参考案例"),
        "retrieval_score": case.get("score", 0),
        "decision": decision,
        "human_relevance_score_1_5": REFERENCE_FEEDBACK_SCORE.get(decision, ""),
        "reference_excerpt": str(case.get("content", ""))[:220],
    }


def update_reference_feedback(
    result: dict[str, Any],
    index: int,
    case: dict[str, Any],
    decision: str | None,
) -> None:
    if not decision:
        return
    feedback_key = f"{result.get('result_id', 'current')}:{index}"
    st.session_state.reference_feedback[feedback_key] = build_reference_feedback_record(
        result,
        index,
        case,
        decision,
    )


def current_reference_feedback(result: dict[str, Any]) -> list[dict[str, Any]]:
    result_id = result.get("result_id", "current")
    rows = []
    for index, case in enumerate(result.get("cases", [])[:3], start=1):
        decision = st.session_state.get(reference_feedback_widget_key(result_id, index))
        if decision:
            rows.append(build_reference_feedback_record(result, index, case, decision))
    return rows


def render_reference_cases(result: dict[str, Any]) -> None:
    cases = result.get("cases", [])
    if not cases:
        st.info("本次没有检索到可用案例，生成内容使用了产品 Brief 和系统规则。")
        return

    visible_cases = cases[:3]
    st.caption(f"参考来源：知识库；展示 Top {len(visible_cases)}")
    st.caption("每条引用都可以标记为采用、不采用或不相关，用于沉淀人工引用相关性数据。")
    result_id = result.get("result_id", "current")
    for index, case in enumerate(visible_cases, start=1):
        title = case.get("title", "参考案例")
        score = case.get("score", 0)
        with st.expander(f"{index}. {title} · 匹配度 {score}", expanded=index == 1):
            st.write(case.get("content", ""))
            decision = st.segmented_control(
                "引用反馈",
                REFERENCE_FEEDBACK_OPTIONS,
                key=reference_feedback_widget_key(result_id, index),
            )
            update_reference_feedback(result, index, case, decision)

    feedback_rows = current_reference_feedback(result)
    if feedback_rows:
        used_count = sum(1 for row in feedback_rows if row["decision"] == "采用")
        irrelevant_count = sum(1 for row in feedback_rows if row["decision"] == "不相关")
        st.success(
            f"已记录 {len(feedback_rows)} 条引用反馈：采用 {used_count} 条，不相关 {irrelevant_count} 条。"
        )
        if irrelevant_count:
            st.warning("存在不相关引用，后续应进入知识库补充或检索标签优化。")
        st.download_button(
            "下载本次引用反馈",
            data=json.dumps(feedback_rows, ensure_ascii=False, indent=2),
            file_name=f"reference_feedback_{result_id}.json",
            mime="application/json",
            use_container_width=True,
            key=f"download_reference_feedback_{result_id}",
        )
    else:
        st.caption("还没有引用反馈。建议生成后至少标记 Top 3，方便后续做人工相关性评估。")


def build_low_score_explanations(scorecard: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, detail in sorted(scorecard.items(), key=lambda item: int(item[1].get("score", 0))):
        score = int(detail.get("score", 0))
        if score > LOW_SCORE_THRESHOLD:
            continue
        action = QUALITY_GAP_ACTIONS.get(
            name,
            {
                "material": "补充该维度对应的产品资料、案例或审核规则。",
                "system_action": "下次生成和审核时提高该维度权重，并沉淀为知识库优化待办。",
            },
        )
        rows.append(
            {
                "dimension": name,
                "score": score,
                "reason": detail.get("issue", "该维度需要进一步人工判断。"),
                "material": action["material"],
                "system_action": action["system_action"],
                "suggestion": detail.get("suggestion", "基于该维度继续优化。"),
            }
        )
    return rows


def render_low_score_explanations(scorecard: dict[str, dict[str, Any]]) -> None:
    low_score_rows = build_low_score_explanations(scorecard)
    st.markdown("**低分原因解释**")
    if not low_score_rows:
        st.success(f"没有低于或等于 {LOW_SCORE_THRESHOLD} 分的维度，本次内容主要进入人工终审确认。")
        return

    st.caption("系统把低分维度拆成原因、资料缺口和下一轮优化动作。")
    for row in low_score_rows:
        with st.expander(f"{row['dimension']} · {row['score']}/10", expanded=True):
            reason_col, material_col, action_col = st.columns(3)
            with reason_col:
                st.markdown("**为什么低**")
                st.write(row["reason"])
            with material_col:
                st.markdown("**需要补什么资料**")
                st.write(row["material"])
            with action_col:
                st.markdown("**系统下次怎么改**")
                st.write(row["system_action"])
            st.caption(f"本次改写建议：{row['suggestion']}")


def render_review_dimensions(result: dict[str, Any]) -> None:
    scorecard = result.get("scorecard", {})
    st.caption(
        f"终审由 {result.get('reviewer_source', '终审 Agent')} 给出；"
        "审核维度固定，避免用户为了拿高分手动筛选审核项。"
    )
    for name in EVALUATION_DIMENSIONS:
        detail = scorecard.get(name)
        if not detail:
            continue
        st.progress(detail["score"] / 10, text=f"{name}: {detail['score']}/10")
        st.caption(f"问题：{detail['issue']}")
        st.caption(f"建议：{detail['suggestion']}")

    draft_scorecard = result.get("draft_scorecard", {})
    if draft_scorecard:
        with st.expander("查看一审草稿审核", expanded=False):
            st.caption(f"一审由 {result.get('draft_reviewer_source', '一审 Agent')} 给出。")
            for name in EVALUATION_DIMENSIONS:
                detail = draft_scorecard.get(name)
                if not detail:
                    continue
                st.progress(detail["score"] / 10, text=f"{name}: {detail['score']}/10")
                st.caption(f"问题：{detail['issue']}")


def render_overall_score(result: dict[str, Any]) -> None:
    score = result.get("score")
    status_label = result.get("status")
    if score is None or status_label is None:
        score, status_label = summarize_score(result.get("scorecard", {}))

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("综合评分", f"{score}/10")
    metric_col2.metric("审核结论", status_label)
    draft_score, _ = summarize_score(result.get("draft_scorecard", {}))
    if draft_score:
        score_delta = score - draft_score
        delta_label = f"+{score_delta}" if score_delta > 0 else str(score_delta)
        metric_col3.metric("一审到终审", f"{draft_score} → {score}", delta=delta_label)
    else:
        metric_col3.metric("审核轮次", "双检双审")

    render_low_score_explanations(result.get("scorecard", {}))

    iteration_plan = result.get("iteration_plan") or build_iteration_plan(result.get("scorecard", {}))
    if iteration_plan:
        st.markdown("**优先迭代建议**")
        for index, suggestion in enumerate(iteration_plan, start=1):
            st.write(f"{index}. {suggestion}")

    human_review_notes = result.get("human_review_notes", [])
    if human_review_notes:
        st.markdown("**人审意见**")
        for index, note in enumerate(human_review_notes, start=1):
            st.write(f"{index}. {note}")


def render_output_panel(result: dict[str, Any] | None) -> None:
    selected_view = select_output_view()
    if not result:
        render_empty_output(selected_view)
        return

    if selected_view == "生成内容":
        render_generated_content(result)
    elif selected_view == "引用案例":
        render_reference_cases(result)
    elif selected_view == "审核维度":
        render_review_dimensions(result)
    elif selected_view == "综合评分":
        render_overall_score(result)


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

                st.write("草稿 Agent 生成初稿")
                draft_content, drafter_source = generate_content(
                    brief,
                    cases,
                    st.session_state.deepseek_api_key,
                    demo_mode=demo_mode,
                )

                st.write("一审 Agent 检查草稿")
                draft_scorecard, draft_reviewer_source = review_content_with_agent(
                    draft_content,
                    brief,
                    cases,
                    st.session_state.deepseek_api_key,
                    demo_mode=demo_mode,
                )

                st.write("迭代 Agent 根据一审意见修改")
                content, revision_source = revise_content_with_agent(
                    draft_content,
                    brief,
                    cases,
                    draft_scorecard,
                    st.session_state.deepseek_api_key,
                    demo_mode=demo_mode,
                )

                st.write("终审 Agent 评分并生成人审意见")
                scorecard, reviewer_source = review_content_with_agent(
                    content,
                    brief,
                    cases,
                    st.session_state.deepseek_api_key,
                    demo_mode=demo_mode,
                )
                score, status_label = summarize_score(scorecard)
                iteration_plan = build_iteration_plan(scorecard)
                human_review_notes = build_human_review_notes(brief, scorecard, score)
                status.update(label="生成完成", state="complete", expanded=False)

            result_id = str(int(time.time() * 1000))
            st.session_state.result = {
                "result_id": result_id,
                "brief": brief,
                "draft_content": draft_content,
                "content": content,
                "cases": cases,
                "retrieval_source": retrieval_source,
                "drafter_source": drafter_source,
                "draft_reviewer_source": draft_reviewer_source,
                "revision_source": revision_source,
                "reviewer_source": reviewer_source,
                "draft_scorecard": draft_scorecard,
                "scorecard": scorecard,
                "score": score,
                "status": status_label,
                "iteration_plan": iteration_plan,
                "human_review_notes": human_review_notes,
            }
            save_history(brief, content, score, status_label)

with right:
    st.subheader("输出")
    result = st.session_state.get("result")
    render_output_panel(result)

if st.session_state.history:
    with st.expander("历史记录", expanded=False):
        for item in reversed(st.session_state.history):
            st.markdown(f"**{item['product']} · {item['content_type']} · {item['score']}/10 · {item['status']}**")
            st.write(item["content"][:400] + ("..." if len(item["content"]) > 400 else ""))
