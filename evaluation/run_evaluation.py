from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover
    QdrantClient = None


ROOT = Path(__file__).resolve().parents[1]
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

EVALUATION_DIMENSIONS = [
    "需求匹配",
    "品牌安全",
    "合规风险",
    "事实一致性",
    "CTA 清晰度",
    "内容完整度",
    "参考依据",
]


def read_secrets() -> dict[str, str]:
    values: dict[str, str] = {}
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        raw = tomllib.loads(secrets_path.read_text(encoding="utf-8-sig"))
        values.update({key: str(value) for key, value in raw.items()})

    for key in ["DEEPSEEK_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def load_testset(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def get_deepseek_client(api_key: str):
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_deepseek(messages: list[dict[str, str]], api_key: str, temperature: float) -> str:
    client = get_deepseek_client(api_key)
    if client is None:
        raise RuntimeError("DeepSeek API key is not configured.")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temperature,
    )
    return re.sub(r"<think>.*?</think>", "", response.choices[0].message.content or "", flags=re.DOTALL).strip()


def payload_to_reference(payload: dict[str, Any], score: float, fallback_title: str) -> dict[str, Any]:
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


def lexical_score(query: str, reference: dict[str, Any], content_type: str) -> float:
    query_tokens = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
    reference_text = f"{reference['title']} {reference['content']}".lower()
    overlap = sum(1 for token in query_tokens if token in reference_text)
    type_bonus = 2 if content_type.lower() in reference_text else 0
    return overlap + type_bonus + min(len(reference["content"]) / 1000, 1)


def scroll_collection(
    client: Any,
    collection_name: str,
    query: str,
    content_type: str,
    limit: int,
) -> list[dict[str, Any]]:
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=120,
        with_payload=True,
        with_vectors=False,
    )
    references = []
    for index, point in enumerate(points):
        payload = point.payload or {}
        payload_type = str(payload.get("type", "")).strip().lower()
        if payload_type and content_type and payload_type != content_type.lower():
            continue
        reference = payload_to_reference(payload, 0, f"{collection_name} {index + 1}")
        reference["score"] = lexical_score(query, reference, content_type)
        references.append(reference)

    references.sort(key=lambda item: item["score"], reverse=True)
    if not references:
        return []

    best_score = max(item["score"] for item in references) or 1
    normalized = []
    for item in references[:limit]:
        normalized.append({**item, "score": round(max(0.55, min(0.98, item["score"] / best_score)), 4)})
    return normalized


def merge_references(*groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for group in groups:
        for item in group:
            key = f"{item.get('title', '')}|{item.get('content', '')[:120]}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def retrieve_references(case: dict[str, Any], secrets: dict[str, str], top_k: int = 3) -> tuple[list[dict[str, Any]], str]:
    if QdrantClient is None:
        return [], "qdrant_client_not_installed"

    url = secrets.get("QDRANT_URL", "")
    api_key = secrets.get("QDRANT_API_KEY", "")
    if not url or not api_key:
        return [], "qdrant_secrets_missing"

    query = "\n".join(
        [
            case["product_name"],
            case["content_type"],
            case["user_need"],
            "\n".join(case["features"]),
            " ".join(case.get("expected_reference_focus", [])),
        ]
    )
    client = QdrantClient(url=url, api_key=api_key)
    case_refs = scroll_collection(client, CASE_COLLECTION_NAME, query, case["content_type"], top_k)
    knowledge_refs = scroll_collection(client, KNOWLEDGE_COLLECTION_NAME, query, "", top_k)
    case_limit = max(1, top_k - 1) if knowledge_refs else top_k
    references = merge_references(case_refs[:case_limit], knowledge_refs, limit=top_k)
    source = "qdrant" if references else "qdrant_empty"
    return references, source


def build_generation_prompt(case: dict[str, Any], references: list[dict[str, Any]]) -> str:
    reference_text = "\n\n".join(
        f"Reference {index + 1}: {item['title']}\n{item['content']}"
        for index, item in enumerate(references)
    )
    return f"""
You are a marketing content agent for a keyboard brand going global.

Write one English {case['content_type']} based on the brief.

Product: {case['product_name']}
Goal: {case['marketing_goal']}
Tone: {case['tone']}
Length: {case['length']}
Structure: {case['structure']}
Features:
{chr(10).join('- ' + item for item in case['features'])}

Must include:
{case.get('must_include', '')}

Avoid:
{', '.join(case.get('avoid_terms', []))}

User need:
{case['user_need']}

References:
{reference_text or 'No references available.'}

Rules:
- Do not mention competitor names.
- Do not invent specs, certification, warranty, user data, testimonials, prices, discounts, or numerical claims.
- If this is EDM, include unsubscribe or preference-management language.
- If this is Blog, do not include email footer language.
- Return only the final marketing copy.
""".strip()


def fallback_generate(case: dict[str, Any], references: list[dict[str, Any]]) -> str:
    features = case["features"]
    feature_sentence = ", ".join(features[:3])
    cta = "Learn more about the setup" if case["marketing_goal"] != "转化" else "Explore the product today"
    if case["content_type"] == "EDM":
        return f"""Subject: Meet {case['product_name']} for a cleaner work setup

Hi there,

{case['product_name']} is built for users who want a more comfortable and adaptable keyboard experience. With {feature_sentence}, it helps turn everyday typing into a smoother part of the workday.

{case.get('must_include', '')}

{cta}.

You can unsubscribe or manage preferences at any time.
"""
    return f"""{case['product_name']}: A Better Keyboard for Everyday Work

The right keyboard should make the desk feel easier to use, not harder to manage. For users who type, plan, and switch between tasks throughout the day, small frictions can add up.

{case['product_name']} brings together {feature_sentence}. These features support a calmer setup, a typing feel that is easier to personalize, and a workspace that can adapt to different routines.

{case.get('must_include', '')}

{cta}.
"""


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if re.search(r"[a-zA-Z0-9]", term):
        return re.search(r"\b" + re.escape(term) + r"\b", text, flags=re.IGNORECASE) is not None
    return term.lower() in text.lower()


def collect_hits(text: str, terms: list[str]) -> list[str]:
    return sorted({term for term in terms if contains_term(text, term)})


def clean_content(text: str, avoid_terms: list[str]) -> str:
    cleaned = text
    for term in COMPETITOR_WORDS + avoid_terms:
        cleaned = re.sub(r"\b" + re.escape(term) + r"\b", "[Competitor Name]", cleaned, flags=re.IGNORECASE)
    patterns = {
        r"\b100%\b": "consistently",
        r"\bzero lag\b": "responsive",
        r"\bno lag\b": "responsive",
        r"\bguaranteed\b": "designed to",
        r"\bultimate\b": "everyday",
        r"\bbest\b": "strong",
        r"\beliminate fatigue\b": "support comfort",
        r"\bno fatigue\b": "more comfort",
        r"\bcure\b": "support",
        r"\bprevent pain\b": "support comfort",
    }
    for pattern, replacement in patterns.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^.*(?:FDA approved|award-winning|certified|free shipping|lifetime warranty).*$\n?", "", cleaned)
    cleaned = re.sub(r"(?im)^.*(?:\b\d+\s*%|\b\d+\s*ms\b|Bluetooth\s*\d|2\.4GHz|300 hours).*$\n?", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_revision_prompt(case: dict[str, Any], draft: str, scorecard: dict[str, dict[str, Any]]) -> str:
    feedback = "\n".join(
        f"- {name}: {detail['score']}/10, {detail['issue']} 建议: {detail['suggestion']}"
        for name, detail in scorecard.items()
    )
    return f"""
Revise the draft based on the first review.
Return only the revised marketing content.

Product: {case['product_name']}
Content type: {case['content_type']}
Avoid terms: {', '.join(case.get('avoid_terms', []))}

First review:
{feedback}

Do not invent specs, certifications, warranties, user data, testimonials, prices, discounts, or numerical claims.
For EDM, include unsubscribe or preference-management language.

Draft:
{draft}
""".strip()


def score_dimension(score: int, issue: str, suggestion: str) -> dict[str, Any]:
    return {"score": max(1, min(10, score)), "issue": issue, "suggestion": suggestion}


def evaluate_content(content: str, case: dict[str, Any], references: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lower_content = content.lower()
    product_name = case["product_name"]
    features = case["features"]
    avoid_terms = case.get("avoid_terms", [])
    scorecard: dict[str, dict[str, Any]] = {}

    feature_hits = sum(1 for feature in features if feature.split()[0].lower() in lower_content)
    if product_name.lower() in lower_content and feature_hits >= min(2, len(features)):
        scorecard["需求匹配"] = score_dimension(9, "产品名和核心卖点覆盖较好。", "保留当前 Brief 对齐方式。")
    elif product_name.lower() in lower_content:
        scorecard["需求匹配"] = score_dimension(7, "产品名出现，但部分核心卖点覆盖不足。", "补充已提供特性的用户收益。")
    else:
        scorecard["需求匹配"] = score_dimension(5, "产品名或核心卖点露出不足。", "明确产品名，并围绕已提供特性展开。")

    competitor_hits = collect_hits(content, COMPETITOR_WORDS + avoid_terms)
    if competitor_hits:
        scorecard["品牌安全"] = score_dimension(4, f"检测到禁用或竞品词：{', '.join(competitor_hits)}。", "替换竞品词并使用泛化对比。")
    else:
        scorecard["品牌安全"] = score_dimension(9, "未检测到禁用或竞品词。", "继续保留禁用词过滤。")

    compliance_terms = [
        "100%",
        "zero lag",
        "no lag",
        "guaranteed",
        "ultimate",
        "cure",
        "prevent pain",
        "eliminate fatigue",
        "no fatigue",
        "for everyone",
        "all brands",
    ]
    compliance_hits = collect_hits(content, compliance_terms)
    edm_footer_ok = case["content_type"] != "EDM" or any(
        term in lower_content for term in ["unsubscribe", "manage preferences", "opt out"]
    )
    if compliance_hits:
        scorecard["合规风险"] = score_dimension(5, f"存在绝对化或医疗化风险：{', '.join(compliance_hits)}。", "改成可解释的体验收益，不做绝对承诺。")
    elif not edm_footer_ok:
        scorecard["合规风险"] = score_dimension(6, "EDM 缺少退订或偏好管理提示。", "补充 unsubscribe / manage preferences。")
    else:
        scorecard["合规风险"] = score_dimension(9, "未发现明显合规风险。", "发布前按目标市场规则复核。")

    unsupported_patterns = [
        r"\b\d+\s*%",
        r"\b\d+\s*ms\b",
        r"Bluetooth\s*\d",
        r"2\.4GHz",
        r"\b\d+\s*hours?\b",
        r"FDA approved",
        r"award-winning",
        r"certified",
        r"free shipping",
        r"30-day",
        r"lifetime warranty",
        r"user feedback",
        r"user test",
        r"testimonial",
        r"5-star",
    ]
    unsupported_hits = sorted(
        {
            match.group(0)
            for pattern in unsupported_patterns
            for match in re.finditer(pattern, content, flags=re.IGNORECASE)
        }
    )
    if unsupported_hits:
        scorecard["事实一致性"] = score_dimension(
            5,
            f"出现未证实参数、认证、优惠或用户数据：{', '.join(unsupported_hits)}。",
            "删除未证实内容；如确需使用，先补充真实来源。",
        )
    else:
        scorecard["事实一致性"] = score_dimension(9, "未发现明显编造事实。", "继续只使用 Brief 和知识库信息。")

    cta_terms = ["learn", "explore", "shop", "discover", "start", "try", "click", "buy"]
    if any(term in lower_content for term in cta_terms):
        scorecard["CTA 清晰度"] = score_dimension(8, "包含明确下一步动作。", "根据营销目标微调 CTA 强度。")
    else:
        scorecard["CTA 清晰度"] = score_dimension(5, "CTA 不够明确。", "增加 learn more / explore / shop 等明确动作。")

    blog_has_email_footer = case["content_type"] == "Blog" and any(
        term in lower_content for term in ["unsubscribe", "manage preferences", "you received this email"]
    )
    if blog_has_email_footer:
        scorecard["内容完整度"] = score_dimension(6, "Blog 混入了 EDM 页脚。", "删除邮件页脚，保留 Blog 结构。")
    elif case["content_type"] == "EDM" and not edm_footer_ok:
        scorecard["内容完整度"] = score_dimension(6, "EDM 结构不完整，缺少合规页脚。", "补充合规页脚。")
    elif len(content.split()) < 50:
        scorecard["内容完整度"] = score_dimension(6, "内容偏短，支撑不足。", "补充场景、收益和 CTA。")
    else:
        scorecard["内容完整度"] = score_dimension(8, "结构基本完整。", "按渠道继续优化篇幅。")

    if len(references) >= 3:
        scorecard["参考依据"] = score_dimension(9, "Top 3 引用已返回。", "人工抽检引用是否与 Brief 相关。")
    elif references:
        scorecard["参考依据"] = score_dimension(6, "引用不足 3 条。", "补充知识库或检查检索条件。")
    else:
        scorecard["参考依据"] = score_dimension(4, "没有可用引用。", "检查 Qdrant 配置和知识库内容。")

    return {name: scorecard[name] for name in EVALUATION_DIMENSIONS}


def summarize_score(scorecard: dict[str, dict[str, Any]]) -> int:
    return round(sum(item["score"] for item in scorecard.values()) / len(scorecard))


def status_label(score: int) -> str:
    if score >= 8:
        return "通过"
    if score >= 6:
        return "需人工复核"
    return "不通过"


def run_case(case: dict[str, Any], secrets: dict[str, str], mode: str, top_k: int) -> dict[str, Any]:
    started_at = time.perf_counter()
    references, reference_source = retrieve_references(case, secrets, top_k=top_k)
    api_key = secrets.get("DEEPSEEK_API_KEY", "")

    if mode == "api":
        draft = call_deepseek(
            [
                {"role": "system", "content": "You write practical marketing content."},
                {"role": "user", "content": build_generation_prompt(case, references)},
            ],
            api_key=api_key,
            temperature=0.6,
        )
    else:
        draft = fallback_generate(case, references)

    draft_scorecard = evaluate_content(draft, case, references)

    if mode == "api":
        final_content = call_deepseek(
            [
                {"role": "system", "content": "You revise marketing content based on audit feedback."},
                {"role": "user", "content": build_revision_prompt(case, draft, draft_scorecard)},
            ],
            api_key=api_key,
            temperature=0.35,
        )
        final_content = clean_content(final_content, case.get("avoid_terms", []))
    else:
        final_content = clean_content(draft, case.get("avoid_terms", []))

    final_scorecard = evaluate_content(final_content, case, references)
    draft_score = summarize_score(draft_scorecard)
    final_score = summarize_score(final_scorecard)

    competitor_hits = collect_hits(final_content, COMPETITOR_WORDS + case.get("avoid_terms", []))
    unsupported_issue = final_scorecard["事实一致性"]["score"] <= 6
    compliance_issue = final_scorecard["合规风险"]["score"] <= 6
    edm_footer_pass = case["content_type"] != "EDM" or any(
        term in final_content.lower() for term in ["unsubscribe", "manage preferences", "opt out"]
    )
    pass_flag = (
        len(references) >= top_k
        and final_score >= 8
        and not competitor_hits
        and not unsupported_issue
        and not compliance_issue
        and edm_footer_pass
    )

    return {
        "case": case,
        "references": references,
        "reference_source": reference_source,
        "draft_content": draft,
        "final_content": final_content,
        "draft_scorecard": draft_scorecard,
        "final_scorecard": final_scorecard,
        "metrics": {
            "case_id": case["case_id"],
            "category": case["category"],
            "content_type": case["content_type"],
            "mode": mode,
            "reference_source": reference_source,
            "reference_count": len(references),
            "reference_top3_pass": len(references) >= top_k,
            "draft_score": draft_score,
            "final_score": final_score,
            "score_delta": final_score - draft_score,
            "status": status_label(final_score),
            "pass": pass_flag,
            "competitor_hit_count": len(competitor_hits),
            "competitor_hits": ", ".join(competitor_hits),
            "edm_footer_pass": edm_footer_pass,
            "factual_pass": not unsupported_issue,
            "compliance_pass": not compliance_issue,
            "latency_seconds": round(time.perf_counter() - started_at, 2),
        },
    }


def write_outputs(results: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"eval_report_{timestamp}.csv"
    jsonl_path = output_dir / f"eval_details_{timestamp}.jsonl"

    fieldnames = list(results[0]["metrics"].keys()) if results else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result["metrics"])

    with jsonl_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No cases were evaluated.")
        return
    metrics = [result["metrics"] for result in results]
    total = len(metrics)
    pass_count = sum(1 for row in metrics if row["pass"])
    avg_final = sum(row["final_score"] for row in metrics) / total
    avg_delta = sum(row["score_delta"] for row in metrics) / total
    ref_pass = sum(1 for row in metrics if row["reference_top3_pass"])
    factual_pass = sum(1 for row in metrics if row["factual_pass"])
    compliance_pass = sum(1 for row in metrics if row["compliance_pass"])
    print(f"cases={total}")
    print(f"pass_rate={pass_count}/{total} ({pass_count / total:.1%})")
    print(f"avg_final_score={avg_final:.2f}")
    print(f"avg_score_delta={avg_delta:.2f}")
    print(f"reference_top3_pass={ref_pass}/{total} ({ref_pass / total:.1%})")
    print(f"factual_pass={factual_pass}/{total} ({factual_pass / total:.1%})")
    print(f"compliance_pass={compliance_pass}/{total} ({compliance_pass / total:.1%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for the keyboard marketing content generator.")
    parser.add_argument("--testset", default="evaluation/marketing_testset.jsonl")
    parser.add_argument("--output-dir", default="evaluation/results")
    parser.add_argument("--mode", choices=["rules", "api"], default="rules")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--category", choices=["normal", "edge", "risk"], default=None)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    testset_path = ROOT / args.testset
    rows = load_testset(testset_path)
    if args.case_id:
        rows = [row for row in rows if row["case_id"] == args.case_id]
    if args.category:
        rows = [row for row in rows if row["category"] == args.category]
    if args.limit is not None:
        rows = rows[: args.limit]

    secrets = read_secrets()
    if args.mode == "api" and not secrets.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for --mode api.")

    results = []
    for index, case in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {case['case_id']} {case['content_type']} {case['category']}")
        try:
            results.append(run_case(case, secrets, args.mode, args.top_k))
        except Exception as exc:
            results.append(
                {
                    "case": case,
                    "references": [],
                    "reference_source": "error",
                    "draft_content": "",
                    "final_content": "",
                    "draft_scorecard": {},
                    "final_scorecard": {},
                    "metrics": {
                        "case_id": case["case_id"],
                        "category": case["category"],
                        "content_type": case["content_type"],
                        "mode": args.mode,
                        "reference_source": "error",
                        "reference_count": 0,
                        "reference_top3_pass": False,
                        "draft_score": 0,
                        "final_score": 0,
                        "score_delta": 0,
                        "status": "error",
                        "pass": False,
                        "competitor_hit_count": 0,
                        "competitor_hits": "",
                        "edm_footer_pass": False,
                        "factual_pass": False,
                        "compliance_pass": False,
                        "latency_seconds": 0,
                        "error": str(exc),
                    },
                }
            )

    csv_path, jsonl_path = write_outputs(results, ROOT / args.output_dir)
    print_summary(results)
    print(f"csv={csv_path}")
    print(f"jsonl={jsonl_path}")


if __name__ == "__main__":
    main()
