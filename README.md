# AI 键盘出海营销内容生成器

这是一个面向产品经理面试展示的 Streamlit 项目，用来模拟“键盘品牌出海营销团队”的真实内容生产流程。

用户输入产品名称、产品特性和营销需求后，系统会结合 Qdrant Cloud 中的参考案例与知识库，调用 DeepSeek 生成英文 Blog 或 EDM 文案，并由审核 Agent 给出固定维度的质量评估和迭代建议。

## 产品定位

**目标用户：** 键盘品牌的海外营销运营人员。

**用户问题：** 营销团队需要快速产出英文内容初稿，但不能只依赖黑盒生成；他们还需要确认文案是否符合产品事实、品牌安全、合规要求和转化目标。

**产品目标：** 把 AI 内容生成变成一个可控、可解释、可迭代的工作流，而不是只返回一段文案。

## 核心能力

1. **产品 Brief 输入**
   - 支持填写产品名称、产品特性、内容类型、营销目标、文案语气、内容长度、内容结构和禁用词。
   - 当前保留 Blog 和 EDM 两种内容类型，因为 Qdrant 案例库中这两类案例最完整。

2. **RAG 检索**
   - 从 Qdrant Cloud 检索 Blog / EDM 参考案例。
   - 同时读取产品事实、品牌规范、合规规则、用户画像和内容结构规则。
   - Top 3 引用支持用户标记“采用 / 不采用 / 不相关”，用于沉淀人工引用相关性反馈。
   - 如果检索或 embedding 失败，会使用内置参考库兜底，降低演示风险。

3. **内容生成 Agent**
   - 使用 DeepSeek 生成英文营销内容。
   - 根据用户选择的目标、语气、结构和产品特性生成可直接阅读的 Blog 或 EDM 初稿。
   - 对 Blog 和 EDM 做不同约束，例如 EDM 允许退订提示，Blog 不加入邮件页脚。

4. **审核 Agent**
   - 审核维度固定展示，不让用户自行选择，避免增加操作负担。
   - 当前固定维度包括：
     - 需求匹配
     - 品牌安全
     - 合规风险
     - 事实一致性
     - CTA 清晰度
     - 内容完整度
     - 参考依据
   - 审核 Agent 会输出评分、问题判断和具体迭代建议。
   - 对低分维度解释风险点、发布前处理建议和需要补充/确认的资料。
   - 评分用于判断是否需要继续修改，不代表真实投放效果预测。

5. **迭代闭环**
   - 生成结果旁边直接展示审核分数和最低分维度。
   - 展示初稿问题、一审建议、终稿和提升维度，让用户看清楚 Agent 改了什么。
   - 将低分维度转成“风险在哪里、发布前怎么处理、要补什么资料”的质量管理信息。
   - 系统给出可执行的改写方向，帮助 PM 说明“如何基于评估继续迭代产品”。

## 为什么这是产品经理项目

- 不是单纯调用大模型，而是围绕真实用户任务设计流程。
- 从“营销人员如何写出可发布内容”出发，而不是从技术展示出发。
- 把 RAG、生成 Agent、审核 Agent、合规审核和迭代建议串成完整闭环。
- 对面试官隐藏 API 配置，让对方能直接体验产品。
- 用固定审核维度降低用户选择成本，体现产品设计取舍。

## AI 产品经理交付物

| 文档 | 用途 |
|---|---|
| [PRD](docs/PRD.md) | 说明目标用户、用户问题、产品目标、功能范围和成功指标 |
| [用户流程](docs/USER_FLOW.md) | 展示从 Brief 到 RAG、双检双审和最终输出的完整链路 |
| [产品与技术架构](docs/ARCHITECTURE.md) | 说明 Streamlit、Qdrant、DeepSeek、Agent 和安全规则如何协作 |
| [指标与监控](docs/METRICS_AND_MONITORING.md) | 定义北极星指标、质量指标、体验指标和上线监控 |
| [风险控制](docs/RISK_AND_GOVERNANCE.md) | 说明事实一致性、合规、品牌安全和人机协同边界 |
| [路线图](docs/ROADMAP.md) | 说明从基础生成器到 RAG、双检双审、测评和运营闭环的迭代路径 |
| [测评报告](evaluation/reports/rules_eval_summary_20260706.md) | 基于 30 条测试集的规则测评摘要 |
| [API 小样本报告](evaluation/reports/api_sample_report_20260707_100638.md) | 基于真实 API 链路抽取风险用例验证生成质量 |
| [人工引用打分说明](evaluation/human_review/reference_relevance_scoring_guide.md) | 定义 Top 3 知识库引用的人工相关性评分口径 |
| [知识库补充待办](evaluation/knowledge_backlog/knowledge_gap_summary_20260707_100515.md) | 将低分维度转成可执行的补库动作 |
| [失败案例回流说明](evaluation/testset_backlog/README.md) | 将线上真实失败沉淀为后续回归测试集 |

## 技术结构

- 前端与产品界面：Streamlit
- 内容生成：DeepSeek API
- RAG 知识库：Qdrant Cloud
- 检索内容：
  - Blog / EDM 历史案例
  - KeyX Pro 产品事实
  - 功能到用户收益映射
  - 用户画像与购买阻力
  - 品牌语气与术语规范
  - 合规规则与审核规则

## Streamlit Secrets 配置

不要把真实 API key 写进 GitHub，也不要让面试官或访问者填写 key。

正确做法是：由 app owner 在 Streamlit Cloud 的 Secrets 后台配置一次，访问者打开页面后直接体验产品。

```toml
INTERVIEW_MODE = true
SHOW_API_SETTINGS = false
DEEPSEEK_API_KEY = "your-deepseek-api-key"

QDRANT_URL = "your-qdrant-cluster-url"
QDRANT_API_KEY = "your-qdrant-api-key"
```

当 `SHOW_API_SETTINGS = false` 时：

- 页面不会展示 API 输入框；
- 访问者不需要、也不能在页面里填写 key；
- 如果 `DEEPSEEK_API_KEY` 存在，系统自动使用真实 AI 生成；
- 如果 key 缺失，系统自动进入 Demo Mode，避免页面不可用。

只有本地调试时才建议临时设置 `SHOW_API_SETTINGS = true`，用于手动检查 API 配置。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run IS6620_GroupAssignment_Group21.py
```

本地真实 API 测试：

1. 复制 `.streamlit/secrets.example.toml` 为 `.streamlit/secrets.toml`。
2. 填入真实的 DeepSeek 和 Qdrant 配置。
3. 不要提交 `.streamlit/secrets.toml`，该文件已被 `.gitignore` 忽略。

## Streamlit Cloud 部署

1. 将代码推送到 GitHub。
2. 打开 Streamlit Community Cloud。
3. 选择该 GitHub 仓库创建 app。
4. Main file 设置为：

```text
IS6620_GroupAssignment_Group21.py
```

5. 在 Streamlit Cloud 的 Secrets 页面填入上面的配置。
6. 部署后将链接发给面试官体验。

## 面试讲解重点

- 我把原来的“生成文案工具”改成了“AI 营销内容工作流产品”。
- 用户不是来调参数的，所以审核维度不让用户选，而是固定输出。
- Blog 和 EDM 是当前案例库最完整的场景，所以先聚焦这两个内容类型。
- RAG 不只是拿案例，也包含产品事实、合规规则和品牌规范。
- 引用案例不是只展示，用户可以反馈采用、不采用或不相关，用于反哺知识库。
- 双检双审不是黑盒流程，页面会展示初稿问题、一审建议、终稿和提升维度。
- 综合评分不是打分器，会把低分原因拆解成发布前处理建议和资料缺口。
- 审核 Agent 不是预测投放效果，而是帮助判断内容是否可发布、是否需要继续迭代。
- 后续可以加入真实内容表现数据，把审核建议和实际转化率、打开率、点击率做闭环验证。
