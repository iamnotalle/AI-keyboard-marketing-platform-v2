# 知识库资料包

这个目录存放可入库的营销知识资料，用来支撑 Qdrant RAG 检索。

## 文件

- `keyboard_marketing_knowledge_v1.jsonl`: 产品事实、功能收益映射、品牌规范、合规规则和 Blog / EDM 优秀案例。

## 设计原则

- 产品事实只写当前项目可确认的信息，不编造价格、保修、认证、销量、用户评价或具体延迟参数。
- 合规资料引用公开官方来源，作为风险初筛规则，不替代法务意见。
- Blog / EDM 案例是内部 approved examples，用于提供结构和表达参考，不冒充真实客户案例。
- 入库 payload 保留 `source_type`、`source_name`、`source_url`、`verified_status` 和 `last_reviewed` 字段，方便后续治理。

## 入库方式

```bash
python scripts/seed_qdrant_knowledge.py
```

脚本会读取 `.streamlit/secrets.toml` 或环境变量中的 Qdrant 配置，将资料写入：

- `marketing_knowledge_base`
- `marketing_cases`

如果没有配置 Qdrant，脚本会停止，不会打印任何 API key。
