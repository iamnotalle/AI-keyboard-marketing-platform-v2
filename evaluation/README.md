# 营销内容生成测试集

这个目录用于评估键盘出海 AI 营销平台的生成质量、RAG 引用质量和双检双审效果。

## 文件

- `marketing_testset.jsonl`: 30 条结构化测试用例。

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

测评结果会输出到 `evaluation/results/`：

- `eval_report_*.csv`: 指标表，适合放进汇报或 Excel。
- `eval_details_*.jsonl`: 每条用例的引用、草稿、最终稿和评分详情。

面试展示时建议先跑 `rules` 全量，再挑 3-5 条风险用例用 `api` 模式跑真实链路。
