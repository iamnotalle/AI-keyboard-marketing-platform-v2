# AI Keyboard Marketing Studio

An AI marketing content workflow for global keyboard brands.

This Streamlit app is designed as a product-manager portfolio project. It helps a marketing operator turn a rough campaign brief into English marketing copy, then reviews the output with a lightweight content quality scorecard.

## Product Positioning

**Target user:** overseas marketing operator for a keyboard brand
**User problem:** content teams need fast first drafts, but they also need confidence that the copy is on-brand, channel-fit, and safe to publish.
**Product goal:** make the AI workflow visible and controllable instead of returning a black-box generation result.

## Core Workflow

1. **Campaign Brief**
   - Product, channel, audience, market, goal, tone, core features, required points, and blocked terms.

2. **RAG Retrieval**
   - Pulls reference cases from Qdrant Cloud when configured.
   - Falls back to a built-in demo knowledge base when Qdrant is not configured.

3. **Drafter Agent**
   - Uses DeepSeek to generate English marketing content.
   - Supports Blog and EDM formats, matching the current reference case library.

4. **Quality Review**
   - Scores the draft on brief match, brand safety, compliance risk, fact consistency, CTA clarity, content depth, and evidence use.
   - Returns concrete iteration suggestions instead of only showing a final score.

5. **PM Notes**
   - Explains user problem, product decision, and next success metrics.

## Why This Is a PM-Oriented AI Project

- It starts from a real user workflow rather than a model demo.
- It structures vague user needs into a campaign brief.
- It exposes retrieval evidence so users can understand why the content was generated.
- It includes a review and iteration loop, which is essential for AI content products.
- It supports an interviewer-friendly live demo without exposing API keys in the UI.

## Interview Demo Mode

For an interview, do **not** hardcode API keys in the GitHub repo.

Use Streamlit Secrets instead:

```toml
INTERVIEW_MODE = true
DEEPSEEK_API_KEY = "your-deepseek-api-key"

# Optional. If omitted, the app uses demo reference cases.
QDRANT_URL = "your-qdrant-cluster-url"
QDRANT_API_KEY = "your-qdrant-api-key"
```

When `INTERVIEW_MODE = true` and `DEEPSEEK_API_KEY` exists:

- the app automatically uses Live AI;
- the API input fields are hidden from the sidebar;
- the interviewer can directly generate and review content.

If the key is missing, the app automatically falls back to Demo Mode.

## Local Setup

```bash
pip install -r requirements.txt
streamlit run IS6620_GroupAssignment_Group21.py
```

For local live testing:

1. Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`.
2. Fill in your actual API keys.
3. Keep `.streamlit/secrets.toml` private. It is already ignored by `.gitignore`.

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io).
3. Create a new app from this repository.
4. Set the main file to:

```text
IS6620_GroupAssignment_Group21.py
```

5. Add the Secrets shown above in Streamlit Cloud settings.
6. Deploy and send the app link to the interviewer.

## Suggested Interview Talking Points

- I redesigned the app around the marketing operator's task flow.
- I separated live AI configuration from the user interface so the interviewer can experience it directly.
- I added Demo Mode as a fallback to reduce demo risk.
- I made the AI output reviewable with a scorecard and iteration plan.
- I avoided committing API keys and used Streamlit Secrets for production-like deployment.
