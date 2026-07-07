# 营销内容生成测试集

这个目录用于评估键盘出海 AI 营销平台的生成质量、RAG 引用质量和双检双审效果。

## 文件

- `marketing_testset.jsonl`: 30 条结构化测试用例。
- `run_evaluation.py`: 批量测评脚本，支持 rules / api 两种模式。
- `ops_loop.py`: 根据测评明细生成评估运营闭环产物。
- `reports/`: 固化后的测评报告，适合放进面试材料。
- `human_review/`: 人工引用相关性打分表。
- `knowledge_backlog/`: 低分维度对应的知识库补充待办。
- `testset_backlog/`: 线上失败案例回流模板和候选测试用例。

## 用例分层

- `normal`: 常规营销需求，验证基础生成能力。
- `edge`: 边界输入，验证短文案、模糊需求、单一特性、语气控制等能力。
- `risk`: 风险输入，验证竞品词、绝对化表达、医疗化表达、虚构用户数据、虚构参数、虚构认证和虚构优惠是否被拦截。

## 建议记录指标

- 终审平均分
- 终审分数 `>= 8` 的通过率
- 一审到终审的平均分数变化
- Top 3 引用案例相关率
- 竞品词命中率
- 虚构事实命中率
- EDM 退订提示通过率
- 人工复核通过率
- 平均生成耗时
- API fallback 率

## 人工抽检方式

每条用例跑完整链路：

1. 输入 Brief。
2. 检查知识库 Top 3 引用是否相关。
3. 检查草稿 Agent 输出。
4. 检查一审 Agent 是否发现关键问题。
5. 检查迭代 Agent 是否修正问题。
6. 检查终审 Agent 分数、人审意见和最终稿。

人工验收时优先看 `human_acceptance_criteria` 字段。

## 运行测评

快速规则测评，不消耗 DeepSeek API：

```bash
python evaluation/run_evaluation.py --mode rules
```

先抽 3 条做 smoke test：

```bash
python evaluation/run_evaluation.py --mode rules --limit 3
```

真实调用 DeepSeek 跑草稿 Agent 和迭代 Agent：

```bash
python evaluation/run_evaluation.py --mode api --limit 5
```

只抽风险用例做 API 小样本：

```bash
python evaluation/run_evaluation.py --mode api --category risk --limit 3
```

测评结果会输出到 `evaluation/results/`：

- `eval_report_*.csv`: 指标表，适合放进汇报或 Excel。
- `eval_details_*.jsonl`: 每条用例的引用、草稿、最终稿和评分详情。

## 评估运营闭环

测评完成后，用 `eval_details_*.jsonl` 生成四类闭环产物：

```bash
python evaluation/ops_loop.py --details evaluation/results/<eval_details_file>.jsonl
```

如果这次是 API 模式小样本，同时生成面试用报告：

```bash
python evaluation/ops_loop.py --details evaluation/results/<eval_details_file>.jsonl --api-sample-report
```

产物说明：

- `human_review/reference_relevance_scorecard_*.csv`: 人工引用相关性打分表，逐条检查 Top 3 引用是否匹配 Brief。
- `reports/api_sample_report_*.md`: API 模式小样本报告，说明真实 LLM 链路是否可用。
- `knowledge_backlog/knowledge_gap_backlog_*.csv`: 低分维度补库待办，包含维度、问题、建议、知识类型、owner、优先级和状态。
- `testset_backlog/online_failure_case_template.json`: 线上失败案例记录模板，人工确认后可回流到测试集。

人工引用相关性建议按 1-5 分记录：

| 分数 | 含义 |
|---:|---|
| 5 | 引用与 Brief、内容类型、产品特性高度匹配，可直接作为生成依据 |
| 4 | 基本相关，只有少量场景或表达不完全贴合 |
| 3 | 部分相关，只能提供通用结构或语气参考 |
| 2 | 弱相关，容易误导生成方向 |
| 1 | 不相关或与需求冲突，需要进入知识库补充或检索优化 |

低分维度默认阈值是 `<= 7`。PM 需要根据待办表补充产品事实、案例、品牌规范或合规规则，更新 Qdrant 后重新运行测评。

线上真实失败案例先填入 `online_failure_case_template.json`，标记失败维度和期望行为；人工确认后复制为新的 JSONL 用例，追加进 `marketing_testset.jsonl`，作为后续回归测试的一部分。

如果已经填好一个失败案例 JSON，可以先转成候选 JSONL：

```bash
python evaluation/ops_loop.py --details evaluation/results/<eval_details_file>.jsonl --online-failure evaluation/testset_backlog/<filled_failure_case>.json
```

面试展示时建议先跑 `rules` 全量，再挑 3-5 条风险用例用 `api` 模式跑真实链路。
