# 📚 AI Study Assistant

Streamlit app with login, PDF Q&A (RAG), voice assistant, practice quizzes, smart study planner,
progress analytics and repeated-question detection. Powered by the Google Gemini API.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then put your real key inside .env
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a **new GitHub repository** and upload every file in this folder.
   **Do NOT upload `.env`, `users.db`, `study_ai.db` or `kb_data/`** (the `.gitignore` already blocks them).
2. Go to https://share.streamlit.io -> **Create app** -> pick your repo, branch `main`, main file `app.py`.
3. Open **Advanced settings** and:
   - choose Python **3.11** or **3.12**
   - paste into **Secrets**:
     ```toml
     GEMINI_API_KEY = "your-real-key"
     GEMINI_MODEL = "gemini-3.8-flash"
     ```
4. Click **Deploy**.

## If something fails
Open the sidebar -> **Diagnostics -> Check AI connection**. It tests the chat and embedding models and
tells you exactly what is wrong (bad key, quota, wrong model name).
The app also shows the real error message instead of a generic one.

| Error | Fix |
|---|---|
| 429 / RESOURCE_EXHAUSTED | Quota hit. Wait, enable billing, or set `GEMINI_MODEL = "gemini-2.5-flash"` |
| 400 / 403 / API key | Create a new key in Google AI Studio, update Secrets |
| 404 / not found | Fix the model name in Secrets |
| "upload documents first" | The knowledge base was wiped by a restart. Re-upload PDFs and click **Build knowledge base** |

## Important: data is temporary on free hosting
Streamlit Cloud's disk is wiped on restart/redeploy. That clears `users.db` (accounts), `study_ai.db`
(quiz history, plans) and `kb_data/` (indexed PDFs). Fine for demos; for real users move accounts and data
to an external database (e.g. Supabase or Turso).

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (tabs) |
| `auth.py` | Sign up / log in / password reset |
| `rag.py` | PDF reading, Gemini embeddings, search, answers, quizzes |
| `db.py` | SQLite storage (quizzes, plans, reviews, rate limits) |
| `planner.py` | Study plan + spaced repetition (SM-2) |
| `voice.py` | Speech-to-text, intent detection, text-to-speech |
