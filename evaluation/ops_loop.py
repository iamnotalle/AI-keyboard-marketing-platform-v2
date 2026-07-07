from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
REPORTS_DIR = EVALUATION_DIR / "reports"
HUMAN_REVIEW_DIR = EVALUATION_DIR / "human_review"
BACKLOG_DIR = EVALUATION_DIR / "knowledge_backlog"
TESTSET_DIR = EVALUATION_DIR / "testset_backlog"

DIMENSION_TO_KNOWLEDGE = {
    "需求匹配": ("product_fact / persona", "补充目标用户、使用场景、功能到收益映射。"),
    "品牌安全": ("brand_voice / banned_terms", "补充禁用词、竞品泛化表达和品牌语气边界。"),
    "合规风险": ("compliance_rule", "补充绝对化、医疗化、EDM 合规和地区规则。"),
    "事实一致性": ("product_fact / source_required", "补充参数来源、认证来源、保修政策和禁止虚构规则。"),
    "CTA 清晰度": ("channel_template", "补充 Blog / EDM 各目标阶段的 CTA 示例。"),
    "内容完整度": ("channel_template", "补充 Blog / EDM 结构模板和最低内容要求。"),
    "参考依据": ("case / retrieval_metadata", "补充可检索案例、标题、type、style 和关键词。"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def latest_details_file() -> Path:
    files = sorted(RESULTS_DIR.glob("eval_details_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No eval_details_*.jsonl found. Run evaluation/run_evaluation.py first.")
    return files[0]


def short_text(value: str, limit: int = 180) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def write_human_reference_scorecard(results: list[dict[str, Any]], output_dir: Path, stamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reference_relevance_scorecard_{stamp}.csv"
    fieldnames = [
        "case_id",
        "category",
        "content_type",
        "user_need",
        "expected_reference_focus",
        "reference_rank",
        "reference_title",
        "retrieval_score",
        "reference_excerpt",
        "human_relevance_score_1_5",
        "human_relevance_label",
        "human_note",
        "action",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            case = result["case"]
            for rank, reference in enumerate(result.get("references", [])[:3], start=1):
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "category": case["category"],
                        "content_type": case["content_type"],
                        "user_need": case["user_need"],
                        "expected_reference_focus": " | ".join(case.get("expected_reference_focus", [])),
                        "reference_rank": rank,
                        "reference_title": reference.get("title", ""),
                        "retrieval_score": reference.get("score", ""),
                        "reference_excerpt": short_text(reference.get("content", "")),
                        "human_relevance_score_1_5": "",
                        "human_relevance_label": "",
                        "human_note": "",
                        "action": "",
                    }
                )
    return output_path


def low_score_rows(results: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        case = result["case"]
        scorecard = result.get("final_scorecard", {})
        for dimension, detail in scorecard.items():
            score = int(detail.get("score", 0))
            if score <= threshold:
                knowledge_type, suggested_action = DIMENSION_TO_KNOWLEDGE.get(
                    dimension,
                    ("knowledge_base", "补充对应规则、案例或产品事实。"),
                )
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "category": case["category"],
                        "content_type": case["content_type"],
                        "dimension": dimension,
                        "score": score,
                        "issue": detail.get("issue", ""),
                        "suggestion": detail.get("suggestion", ""),
                        "knowledge_type": knowledge_type,
                        "suggested_action": suggested_action,
                        "owner": "",
                        "priority": "P1" if score <= 5 else "P2",
                        "status": "todo",
                    }
                )
    return rows


def write_knowledge_backlog(results: list[dict[str, Any]], output_dir: Path, stamp: str, threshold: int) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = low_score_rows(results, threshold)
    csv_path = output_dir / f"knowledge_gap_backlog_{stamp}.csv"
    fieldnames = [
        "case_id",
        "category",
        "content_type",
        "dimension",
        "score",
        "issue",
        "suggestion",
        "knowledge_type",
        "suggested_action",
        "owner",
        "priority",
        "status",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = output_dir / f"knowledge_gap_summary_{stamp}.md"
    counts = Counter(row["dimension"] for row in rows)
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# 低分维度知识库补充待办\n\n")
        file.write(f"生成时间：{stamp}\n\n")
        file.write(f"低分阈值：`<= {threshold}`\n\n")
        if not rows:
            file.write("本次测评没有低于阈值的维度。建议继续用 API 模式和线上真实失败案例扩充观察样本。\n\n")
        else:
            file.write("## 低分维度分布\n\n")
            file.write("| 维度 | 数量 |\n|---|---:|\n")
            for dimension, count in counts.most_common():
                file.write(f"| {dimension} | {count} |\n")
            file.write("\n## 处理机制\n\n")
            file.write("1. PM 根据 `knowledge_gap_backlog_*.csv` 分配 owner 和优先级。\n")
            file.write("2. 补充产品事实、案例、品牌规则或合规规则。\n")
            file.write("3. 更新 Qdrant 知识库 payload，确保 `type/title/content` 可检索。\n")
            file.write("4. 重新运行测评，比较低分维度是否改善。\n")
    return csv_path, md_path


def failure_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for result in results:
        metrics = result.get("metrics", {})
        if metrics.get("pass") is True:
            continue
        case = result["case"]
        failed_dimensions = [
            name
            for name, detail in result.get("final_scorecard", {}).items()
            if int(detail.get("score", 0)) <= 6
        ]
        candidates.append(
            {
                "case_id": f"ONLINE_FAIL_{case['case_id']}",
                "source_case_id": case["case_id"],
                "source": "evaluation_or_online_failure",
                "category": "risk",
                "content_type": case["content_type"],
                "product_name": case["product_name"],
                "features": case["features"],
                "marketing_goal": case["marketing_goal"],
                "tone": case["tone"],
                "length": case["length"],
                "structure": case["structure"],
                "user_need": case["user_need"],
                "must_include": case.get("must_include", ""),
                "avoid_terms": case.get("avoid_terms", []),
                "expected_reference_focus": case.get("expected_reference_focus", []),
                "risk_tags": sorted(set(case.get("risk_tags", []) + ["online_failure"] + failed_dimensions)),
                "failure_observation": {
                    "status": metrics.get("status", ""),
                    "final_score": metrics.get("final_score", 0),
                    "failed_dimensions": failed_dimensions,
                    "competitor_hits": metrics.get("competitor_hits", ""),
                },
                "human_acceptance_criteria": case.get("human_acceptance_criteria", []),
            }
        )
    return candidates


def write_failure_backlog(results: list[dict[str, Any]], output_dir: Path, stamp: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = failure_candidates(results)
    jsonl_path = output_dir / f"failure_testset_candidates_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    template_path = output_dir / "online_failure_case_template.json"
    template = {
        "case_id": "ONLINE_FAIL_YYYYMMDD_001",
        "source": "streamlit_user_feedback",
        "category": "risk",
        "content_type": "Blog or EDM",
        "product_name": "",
        "features": [],
        "marketing_goal": "",
        "tone": "",
        "length": "",
        "structure": "",
        "user_need": "",
        "must_include": "",
        "avoid_terms": [],
        "expected_reference_focus": [],
        "risk_tags": ["online_failure"],
        "failure_observation": {
            "user_report": "",
            "bad_output_excerpt": "",
            "failed_dimensions": [],
            "expected_behavior": "",
        },
        "human_acceptance_criteria": [],
    }
    template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return jsonl_path, template_path


def write_online_failure_candidates(cases: list[dict[str, Any]], output_dir: Path, stamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"online_failure_candidates_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for index, case in enumerate(cases, start=1):
            candidate = dict(case)
            case_id = str(candidate.get("case_id", "")).strip()
            if not case_id or case_id == "ONLINE_FAIL_YYYYMMDD_001":
                candidate["case_id"] = f"ONLINE_FAIL_{stamp[:8]}_{index:03d}"
            candidate.setdefault("source", "streamlit_user_feedback")
            candidate.setdefault("category", "risk")
            risk_tags = candidate.get("risk_tags") or []
            if "online_failure" not in risk_tags:
                risk_tags.append("online_failure")
            candidate["risk_tags"] = risk_tags
            candidate.setdefault("human_acceptance_criteria", [])
            file.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return jsonl_path


def write_api_sample_report(results: list[dict[str, Any]], output_dir: Path, stamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"api_sample_report_{stamp}.md"
    metrics = [result["metrics"] for result in results]
    total = len(metrics)
    pass_count = sum(1 for row in metrics if row.get("pass") is True)
    avg_final = sum(float(row.get("final_score", 0)) for row in metrics) / total if total else 0
    avg_delta = sum(float(row.get("score_delta", 0)) for row in metrics) / total if total else 0
    ref_pass = sum(1 for row in metrics if row.get("reference_top3_pass") is True)
    factual_pass = sum(1 for row in metrics if row.get("factual_pass") is True)
    compliance_pass = sum(1 for row in metrics if row.get("compliance_pass") is True)

    with path.open("w", encoding="utf-8") as file:
        file.write("# API 模式小样本测评报告\n\n")
        file.write(f"生成时间：{stamp}\n\n")
        file.write("## 总览\n\n")
        file.write("| 指标 | 结果 |\n|---|---:|\n")
        file.write(f"| 用例数 | {total} |\n")
        file.write(f"| 通过率 | {pass_count}/{total} ({pass_count / total:.1%}) |\n" if total else "| 通过率 | 0 |\n")
        file.write(f"| 平均终审分 | {avg_final:.2f} |\n")
        file.write(f"| 平均一审到终审变化 | {avg_delta:.2f} |\n")
        file.write(f"| Top 3 引用返回率 | {ref_pass}/{total} |\n")
        file.write(f"| 事实一致性通过率 | {factual_pass}/{total} |\n")
        file.write(f"| 合规风险通过率 | {compliance_pass}/{total} |\n\n")
        file.write("## 明细\n\n")
        file.write("| case_id | 类型 | 内容类型 | 终审分 | 结论 | Top3 引用 | 事实一致性 | 合规 |\n")
        file.write("|---|---|---|---:|---|---|---|---|\n")
        for row in metrics:
            file.write(
                f"| {row['case_id']} | {row['category']} | {row['content_type']} | {row['final_score']} | "
                f"{row['status']} | {row['reference_top3_pass']} | {row['factual_pass']} | {row['compliance_pass']} |\n"
            )
        file.write("\n## 使用说明\n\n")
        file.write("API 模式会真实调用 DeepSeek，建议每次抽取 3-5 条 normal / edge / risk 混合用例即可。\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PM ops loop artifacts from evaluation details.")
    parser.add_argument("--details", default=None, help="Path to eval_details_*.jsonl. Defaults to latest.")
    parser.add_argument("--low-score-threshold", type=int, default=7)
    parser.add_argument("--api-sample-report", action="store_true")
    parser.add_argument("--online-failure", default=None, help="Path to a filled online failure .json or .jsonl file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details_path = ROOT / args.details if args.details else latest_details_file()
    results = load_jsonl(details_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reference_path = write_human_reference_scorecard(results, HUMAN_REVIEW_DIR, stamp)
    backlog_csv, backlog_md = write_knowledge_backlog(results, BACKLOG_DIR, stamp, args.low_score_threshold)
    failure_jsonl, failure_template = write_failure_backlog(results, TESTSET_DIR, stamp)

    print(f"details={details_path}")
    print(f"human_reference_scorecard={reference_path}")
    print(f"knowledge_backlog_csv={backlog_csv}")
    print(f"knowledge_backlog_md={backlog_md}")
    print(f"failure_candidates={failure_jsonl}")
    print(f"failure_template={failure_template}")

    if args.online_failure:
        online_failure_path = ROOT / args.online_failure
        online_cases = load_json_or_jsonl(online_failure_path)
        online_jsonl = write_online_failure_candidates(online_cases, TESTSET_DIR, stamp)
        print(f"online_failure_candidates={online_jsonl}")

    if args.api_sample_report:
        report_path = write_api_sample_report(results, REPORTS_DIR, stamp)
        print(f"api_sample_report={report_path}")


if __name__ == "__main__":
    main()
