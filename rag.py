"""Core RAG logic: PDF reading, chunking, Gemini embeddings, vector search, LLM answers, MCQs.

Deployment-friendly: no PyTorch / sentence-transformers / FAISS. Embeddings come from the
Gemini API and similarity search is plain NumPy, so the app installs fast and uses little RAM.
"""
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

# Load .env from the same folder as this file (local development only)
load_dotenv(Path(__file__).parent / ".env", override=True)

log = logging.getLogger("study_ai.rag")


def _get_setting(name, default=None):
    """Read a setting from environment/.env first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


api_key = _get_setting("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

client = genai.Client(api_key=api_key)
LLM_MODEL = _get_setting("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_MODEL = _get_setting("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = 768  # smaller vectors = smaller files, still accurate for study notes


# ---------- Error helpers ----------
def _retry(call, retries=3):
    """Run a Gemini API call, retrying on server errors and rate limits."""
    for attempt in range(retries):
        try:
            return call()
        except errors.ServerError as e:  # 5xx
            if attempt == retries - 1:
                raise RuntimeError(f"Gemini server error {e.code}: {e.message}") from e
            time.sleep(2 ** attempt)
        except errors.ClientError as e:  # 4xx
            if e.code == 429 and attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini error {e.code}: {e.message}") from e


def explain_error(e):
    """Turn an exception into a short message with a hint about the likely fix."""
    msg = str(e)[:500]
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        hint = ("Quota or rate limit reached. Wait a minute, enable billing for your key, "
                "or set GEMINI_MODEL to a model your quota allows (e.g. gemini-2.5-flash).")
    elif "404" in msg or "not_found" in low:
        hint = "Model not found. Check GEMINI_MODEL / GEMINI_EMBED_MODEL in your Secrets."
    elif "api key" in low or "api_key" in low or "permission_denied" in low or "403" in msg:
        hint = "API key problem. Create a new key in Google AI Studio and update your Secrets."
    elif "empty response" in low:
        hint = "The model returned nothing. Try again."
    else:
        hint = ""
    return f"{msg}\n\n{hint}".strip()


def diagnose():
    """Test the chat model and the embedding model. Returns [(label, ok, message)]."""
    results = []
    try:
        client.models.generate_content(model=LLM_MODEL, contents="Say OK.")
        results.append((f"Chat model ({LLM_MODEL})", True, "working"))
    except Exception as e:
        results.append((f"Chat model ({LLM_MODEL})", False, explain_error(e)))
    try:
        embed(["test"])
        results.append((f"Embedding model ({EMBED_MODEL})", True, "working"))
    except Exception as e:
        results.append((f"Embedding model ({EMBED_MODEL})", False, explain_error(e)))
    return results


# ---------- Embeddings ----------
def embed(texts, task="RETRIEVAL_DOCUMENT"):
    """Return L2-normalised float32 vectors, one row per text."""
    rows = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        resp = _retry(lambda b=batch: client.models.embed_content(
            model=EMBED_MODEL,
            contents=b,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM),
        ))
        rows.extend(e.values for e in resp.embeddings)
    vecs = np.array(rows, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# ---------- PDF + chunking ----------
def read_pdf(file):
    """Return list of (page_number, text)."""
    reader = PdfReader(file)
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def chunk_pages(pages, source, chunk_size=800, overlap=150):
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"
    chunks = []
    for page_no, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_size]
            if len(piece.strip()) > 50:
                chunks.append({"text": piece, "source": source, "page": page_no})
            start += chunk_size - overlap
    return chunks


# ---------- Vector store ----------
class KnowledgeBase:
    def __init__(self, embed_model=EMBED_MODEL):
        self.embed_model = embed_model
        self.chunks = []
        self.vecs = None  # numpy array, shape (n_chunks, EMBED_DIM)

    def add(self, chunks):
        if not chunks:
            return
        vecs = embed([c["text"] for c in chunks])
        self.vecs = vecs if self.vecs is None else np.vstack([self.vecs, vecs])
        self.chunks.extend(chunks)

    def search(self, query, k=4):
        if self.vecs is None or not self.chunks:
            return []
        q = embed([query], task="RETRIEVAL_QUERY")[0]
        scores = self.vecs @ q  # cosine similarity (vectors are normalised)
        top = np.argsort(-scores)[:min(k, len(self.chunks))]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


# ---------- LLM ----------
def ask_llm(system, user, max_tokens=8000, json_mode=False):
    """Call Gemini. Raises RuntimeError with a readable message on failure."""
    config = {"system_instruction": system, "max_output_tokens": max_tokens}
    if json_mode:
        config["response_mime_type"] = "application/json"
    resp = _retry(lambda: client.models.generate_content(
        model=LLM_MODEL, contents=user, config=config))
    if not resp.text:
        reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
        raise RuntimeError(f"Empty response from model (finish_reason={reason})")
    return resp.text


def answer_question(kb, question, k=4):
    hits = kb.search(question, k)
    if not hits:
        return "Please upload and index some documents first.", []
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits)
    system = (
        "You are a study assistant. Answer ONLY using the provided context. "
        "Cite sources inline like [file.pdf p.3]. If the context does not contain "
        "the answer, say you could not find it in the uploaded material."
    )
    answer = ask_llm(system, f"Context:\n{context}\n\nQuestion: {question}")
    return answer, hits


def generate_mcqs(kb, topic, n=5):
    """Return a list of MCQ dicts. [] if there is no material; raises on API/parse failure."""
    hits = kb.search(topic, k=6)
    if not hits:
        return []
    context = "\n\n".join(h["text"] for h in hits)
    system = "You write exam MCQs. Respond with ONLY valid JSON, no markdown fences."
    prompt = (
        f"Using only this material, write {n} multiple-choice questions on '{topic}'.\n"
        'Format: [{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], '
        '"answer": "B", "explanation": "..."}]\n\n'
        f"Material:\n{context}"
    )
    raw = ask_llm(system, prompt, max_tokens=8000, json_mode=True)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("MCQ JSON parse failed: %s", raw[:300])
        raise ValueError("The model returned invalid JSON. Please try again.") from e
    if isinstance(data, dict):  # some replies wrap the list in an object
        data = next((v for v in data.values() if isinstance(v, list)), [])
    ok = []
    for m in data:
        if (isinstance(m, dict) and {"question", "options", "answer"} <= m.keys()
                and isinstance(m["options"], list) and len(m["options"]) >= 2):
            m["answer"] = str(m["answer"]).strip()[:1].upper()
            ok.append(m)
    return ok


# ---------- Past paper analysis ----------
def extract_questions(pages):
    """Pull sentences ending in '?' from past-paper pages."""
    qs = []
    for _, text in pages:
        text = re.sub(r"\s+", " ", text)
        qs += [q.strip() for q in re.findall(r"[A-Z][^?]{15,250}\?", text)]
    return qs


def cluster_questions(questions, threshold=0.85):
    """Group near-duplicate questions; return [(representative, count)] sorted by count."""
    if not questions:
        return []
    vecs = embed(questions, task="SEMANTIC_SIMILARITY")
    clusters = []  # each: {"rep_idx": int, "members": [idx]}
    for i, v in enumerate(vecs):
        for c in clusters:
            if float(vecs[c["rep_idx"]] @ v) >= threshold:
                c["members"].append(i)
                break
        else:
            clusters.append({"rep_idx": i, "members": [i]})
    out = [(questions[c["rep_idx"]], len(c["members"])) for c in clusters]
    return sorted(out, key=lambda x: -x[1])


# ---------- Persistence: knowledge base survives restarts (while the disk lasts) ----------
def save_kb(kb, questions, folder):
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    if kb.vecs is not None:
        np.save(p / "vectors.npy", kb.vecs)
    elif (p / "vectors.npy").exists():
        (p / "vectors.npy").unlink()
    (p / "data.json").write_text(json.dumps(
        {"embed_model": kb.embed_model, "chunks": kb.chunks, "questions": questions}),
        encoding="utf-8")


def load_kb(folder):
    """Return (KnowledgeBase, questions); an empty pair if nothing saved or files are corrupt."""
    p = Path(folder)
    try:
        if (p / "data.json").exists() and (p / "vectors.npy").exists():
            data = json.loads((p / "data.json").read_text(encoding="utf-8"))
            if data.get("embed_model") != EMBED_MODEL:
                return KnowledgeBase(), []  # built with a different embedding model: rebuild
            kb = KnowledgeBase(EMBED_MODEL)
            kb.chunks = data["chunks"]
            kb.vecs = np.load(p / "vectors.npy")
            if len(kb.chunks) != len(kb.vecs):
                return KnowledgeBase(), []
            return kb, data.get("questions", [])
    except Exception:
        log.exception("Could not load saved knowledge base from %s", folder)
    return KnowledgeBase(), []
