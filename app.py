"""
AI Study Assistant — Streamlit UI with login, RAG Q&A, voice assistant, smart planner, analytics.
Run:  python -m streamlit run app.py
Key is read from .env (local) or Streamlit secrets (deployed): GEMINI_API_KEY
"""
import hashlib
import logging
import os
import shutil
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit call
st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("study_ai")

import auth
import db
import planner
import rag
import voice

db.init_db()

MAX_FILES = int(os.getenv("MAX_UPLOAD_FILES", 10))
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", 15))
MAX_PAGES = int(os.getenv("MAX_PDF_PAGES", 300))
LLM_PER_HOUR = int(os.getenv("LLM_CALLS_PER_HOUR", 60))

# ---------- Login gate: nothing below runs until the user is signed in ----------
user = auth.require_login()

st.title("📚 AI Study Assistant")

# ---------- Per-user knowledge base ----------
kb_folder = f"kb_data/{user}"  # username is restricted to a-z, 0-9, _ so this is a safe path

if st.session_state.get("kb_owner") != user:
    kb, questions = rag.load_kb(kb_folder)
    st.session_state.kb = kb
    st.session_state.questions = questions
    st.session_state.mcqs = []
    st.session_state.mcq_topic = ""
    st.session_state.kb_owner = user


# ---------- Helpers ----------
def within_limit():
    """Per-user hourly cap on AI calls (protects your API bill)."""
    if db.allow(user, "llm", LLM_PER_HOUR):
        return True
    st.error(f"Hourly limit of {LLM_PER_HOUR} AI requests reached. Please try again later.")
    return False


def record_quiz(topic, correct, total):
    """Save a quiz score and update the spaced-repetition schedule for that topic."""
    db.add_quiz_result(user, topic, correct, total)
    r = db.get_review(user, topic) or {"ease": 2.5, "interval": 0, "reps": 0}
    ease, interval, reps = planner.sm2(r["ease"], r["interval"], r["reps"], round(5 * correct / total))
    db.save_review(user, topic, ease, interval, reps, (date.today() + timedelta(days=interval)).isoformat())


def handle_voice(text):
    intent = voice.detect_intent(text)
    if intent == "plan_today":
        tasks = [t for t in db.get_tasks(user, date.today().isoformat()) if not t["done"]]
        if not tasks:
            return "You have nothing left to study today."
        return "Today you should study: " + "; ".join(
            f"{t['topic']} for {t['minutes']} minutes ({t['kind']})" for t in tasks) + "."
    if intent == "progress":
        acc = db.topic_accuracy(user)
        if not acc:
            return "You haven't saved any quiz scores yet."
        return f"Your weakest topic is {acc[0]['topic']} at {acc[0]['accuracy']:.0%}. Focus on that first."
    answer, _ = rag.answer_question(st.session_state.kb, text)
    return answer


def valid_uploads(files):
    ok = []
    for f in files:
        if f.size > MAX_MB * 1024 * 1024:
            st.warning(f"Skipped {f.name}: larger than {MAX_MB} MB.")
        else:
            ok.append(f)
    return ok


# ---------- Sidebar: account + upload ----------
with st.sidebar:
    auth.sidebar_header(user)

    st.header("1. Upload material")
    notes = st.file_uploader("Syllabus / notes (PDF)", type="pdf", accept_multiple_files=True)
    papers = st.file_uploader("Past papers (PDF)", type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 100)

    if st.button("Build knowledge base", type="primary"):
        notes_ok, papers_ok = valid_uploads(notes or []), valid_uploads(papers or [])
        if len(notes_ok) + len(papers_ok) > MAX_FILES:
            st.error(f"Please upload at most {MAX_FILES} files at a time.")
        elif not notes_ok and not papers_ok:
            st.error("Upload at least one valid PDF.")
        else:
            kb = rag.KnowledgeBase()
            questions = []
            try:
                with st.spinner("Reading and indexing..."):
                    for f in notes_ok + papers_ok:
                        pages = rag.read_pdf(f)
                        if len(pages) > MAX_PAGES:
                            st.warning(f"Skipped {f.name}: more than {MAX_PAGES} pages.")
                            continue
                        kb.add(rag.chunk_pages(pages, f.name, chunk_size=chunk_size))
                        if f in papers_ok:
                            questions += rag.extract_questions(pages)
                st.session_state.kb, st.session_state.questions = kb, questions
                rag.save_kb(kb, questions, kb_folder)
                st.success(f"Indexed {len(kb.chunks)} chunks, found {len(questions)} past-paper questions.")
            except Exception as e:
                log.exception("indexing failed")
                st.error(f"Indexing failed: {rag.explain_error(e)}")

    with st.expander("Privacy"):
        confirm = st.checkbox("I want to delete all my study data")
        if st.button("Delete my data", disabled=not confirm):
            db.delete_user_data(user)
            shutil.rmtree(kb_folder, ignore_errors=True)
            st.session_state.kb, st.session_state.questions = rag.KnowledgeBase(), []
            st.success("Your study data was deleted.")

    with st.expander("Diagnostics"):
        if st.button("Check AI connection"):
            with st.spinner("Testing..."):
                for label, ok, msg in rag.diagnose():
                    (st.success if ok else st.error)(f"{label}: {msg}")
    st.caption("Note: on free hosting, saved files can reset when the app restarts. "
               "If answers say 'upload documents first', rebuild the knowledge base.")

tab_ask, tab_voice, tab_quiz, tab_plan, tab_prog, tab_repeat = st.tabs(
    ["💬 Ask", "🎙️ Voice", "📝 Practice quiz", "🗓️ Study planner", "📈 Progress", "🔁 Repeated questions"])

# ---------- Tab 1: Q&A ----------
with tab_ask:
    q = st.text_input("Ask a question from your syllabus:")
    if st.button("Get answer") and q.strip() and within_limit():
        try:
            with st.spinner("Thinking..."):
                answer, hits = rag.answer_question(st.session_state.kb, q)
            db.add_history(user, q, answer)
            st.markdown(answer)
            with st.expander("Sources used"):
                for h in hits:
                    st.caption(f"{h['source']} — page {h['page']} (similarity {h['score']:.2f})")
                    st.write(h["text"])
        except Exception as e:
            log.exception("answer_question failed")
            st.error(f"Could not get an answer: {rag.explain_error(e)}")
    with st.expander("Recent questions"):
        for h in db.get_history(user, 10):
            st.markdown(f"**{h['question']}**")
            st.caption(h["created"])
            st.write(h["answer"])

# ---------- Tab 2: Voice assistant ----------
with tab_voice:
    st.write("Speak a question or a command, e.g. *“What's my plan for today?”*, "
             "*“How is my progress?”*, or any syllabus question.")
    audio = st.audio_input("Record")
    if audio is not None:
        data = audio.getvalue()
        key = hashlib.md5(data).hexdigest()
        if st.session_state.get("voice_key") != key and within_limit():  # no repeat API calls on rerun
            try:
                with st.spinner("Listening..."):
                    text = voice.transcribe(rag.client, rag.LLM_MODEL, data)
                    reply = handle_voice(text) if text else "I couldn't hear anything. Please try again."
                    st.session_state.voice_result = (text, reply, voice.speak(reply))
                    st.session_state.voice_key = key
            except Exception as e:
                log.exception("voice pipeline failed")
                st.error(f"Voice request failed: {rag.explain_error(e)}")
    res = st.session_state.get("voice_result")
    if res:
        st.markdown(f"**You said:** {res[0]}")
        st.markdown(res[1])
        if res[2]:
            st.audio(res[2], format="audio/mp3", autoplay=True)

# ---------- Tab 3: MCQs ----------
with tab_quiz:
    topic = st.text_input("Topic for quiz (e.g. 'normalization in DBMS'):")
    n = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate quiz") and topic.strip() and within_limit():
        try:
            with st.spinner("Writing questions..."):
                st.session_state.mcqs = rag.generate_mcqs(st.session_state.kb, topic, n)
            st.session_state.mcq_topic = topic.strip()
            for i in range(10):  # clear previous answers
                st.session_state.pop(f"mcq{i}", None)
        except Exception as e:
            log.exception("generate_mcqs failed")
            st.session_state.mcqs = []
            st.error(f"Could not generate questions: {rag.explain_error(e)}")
        else:
            if not st.session_state.mcqs:
                st.error("No material found for that topic. Upload PDFs and click "
                         "'Build knowledge base' first.")
    answered = correct = 0
    for i, m in enumerate(st.session_state.mcqs):
        st.markdown(f"**Q{i + 1}. {m['question']}**")
        choice = st.radio("Your answer", m["options"], key=f"mcq{i}", index=None)
        if choice:
            answered += 1
            if choice.strip().startswith(m["answer"]):
                correct += 1
                st.success("Correct!")
            else:
                st.error(f"Wrong. Correct answer: {m['answer']}")
            st.info(m.get("explanation", ""))
    if st.session_state.mcqs and answered == len(st.session_state.mcqs):
        st.metric("Score", f"{correct}/{answered}")
        if st.button("Save score to my progress"):
            record_quiz(st.session_state.mcq_topic, correct, answered)
            st.success("Saved. Your planner and revision schedule will use this.")

# ---------- Tab 4: Smart study planner ----------
with tab_plan:
    st.write("Enter topics as `Topic, difficulty(1-5)`, one per line. Weak topics from your quizzes get extra time.")
    topics_text = st.text_area("Topics", "Normalization, 4\nTransactions, 3\nSQL joins, 2", height=120)
    c1, c2, c3 = st.columns(3)
    exam = c1.date_input("Exam date", min_value=date.today())
    hours = c2.slider("Hours per day", 1.0, 8.0, 2.0, 0.5)
    boost = c3.checkbox("Boost weak topics", value=True)
    if st.button("Generate plan", type="primary"):
        weak = {a["topic"].lower(): 1 - a["accuracy"] for a in db.topic_accuracy(user)} if boost else {}
        ts = planner.parse_topics(topics_text)
        for t in ts:
            t["weakness"] = weak.get(t["name"].lower(), 0.0)
        try:
            db.replace_plan(user, planner.generate_plan(ts, exam, hours))
            st.success("Plan created.")
        except ValueError as e:
            st.error(str(e))

    due = db.due_reviews(user, date.today().isoformat())
    if due:
        st.warning("🔁 Revise today (spaced repetition): " + ", ".join(r["topic"] for r in due))

    tasks = db.get_tasks(user)
    if tasks:
        done_n = sum(t["done"] for t in tasks)
        st.progress(done_n / len(tasks), text=f"{done_n}/{len(tasks)} sessions completed")
        current = None
        for t in tasks:
            if t["due"] != current:
                current = t["due"]
                st.subheader(current + ("  (today)" if current == date.today().isoformat() else ""))
            label = f"{t['topic']} — {t['minutes']} min ({t['kind']})"
            if st.checkbox(label, value=bool(t["done"]), key=f"task{t['id']}") != bool(t["done"]):
                db.set_done(user, t["id"], not t["done"])
                st.rerun()

# ---------- Tab 5: Progress analytics ----------
with tab_prog:
    acc = db.topic_accuracy(user)
    if not acc:
        st.info("Take a quiz and save your score to see progress here.")
    else:
        df = pd.DataFrame(acc)
        st.bar_chart(df.set_index("topic")["accuracy"])
        weak_df = df[df["accuracy"] < 0.6]
        if len(weak_df):
            st.warning("Weak topics (<60%): " + ", ".join(weak_df["topic"]))
        st.dataframe(df, hide_index=True)

# ---------- Tab 6: repeated questions ----------
with tab_repeat:
    st.write("Questions that appear (in similar form) across your uploaded past papers.")
    if st.button("Analyze past papers"):
        if not st.session_state.questions:
            st.info("No questions found. Upload past papers and rebuild the knowledge base.")
        else:
            for rep, count in rag.cluster_questions(st.session_state.questions)[:15]:
                st.write(f"**{count}×** — {rep}")
