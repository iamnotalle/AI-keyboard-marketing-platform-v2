# 线上失败案例回流说明

这个目录用于把真实用户体验中发现的问题沉淀为回归测试用例。

## 使用流程

1. 用户或面试官发现生成失败，例如引用不相关、事实编造、合规风险、内容不完整。
2. 按 `online_failure_case_template.json` 记录原始 Brief、错误输出片段、失败维度和期望行为。
3. PM 确认这个问题不是偶发网络/API 问题，而是产品能力问题。
4. 将确认后的失败案例转成 JSONL，用 `ONLINE_FAIL_YYYYMMDD_序号` 作为 `case_id`。
5. 追加到 `evaluation/marketing_testset.jsonl`。
6. 重新运行 rules 全量测评和 API 小样本测评，确认同类问题被修复。

## 回流标准

- 只回流可复现、可判断、可验收的问题。
- 每条失败案例必须写清楚 `failed_dimensions` 和 `expected_behavior`。
- 涉及产品参数、认证、保修、折扣、用户评价的数据，必须有可信来源后才能进入知识库。
