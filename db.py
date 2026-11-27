"""SQLite persistence layer (quiz history, study plan tasks, spaced-repetition state)."""
import logging
import os
import sqlite3
import time
from contextlib import contextmanager

log = logging.getLogger("study_ai.db")
DB_PATH = os.getenv("STUDY_DB_PATH", "study_ai.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS quiz_results(
    id INTEGER PRIMARY KEY, user TEXT NOT NULL, topic TEXT NOT NULL,
    correct INTEGER NOT NULL, total INTEGER NOT NULL,
    created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS tasks(
    id INTEGER PRIMARY KEY, user TEXT NOT NULL, topic TEXT NOT NULL,
    due TEXT NOT NULL, minutes INTEGER NOT NULL, kind TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS reviews(
    user TEXT NOT NULL, topic TEXT NOT NULL, ease REAL NOT NULL DEFAULT 2.5,
    interval INTEGER NOT NULL DEFAULT 0, reps INTEGER NOT NULL DEFAULT 0,
    next_due TEXT NOT NULL, PRIMARY KEY(user, topic));
CREATE TABLE IF NOT EXISTS usage(user TEXT NOT NULL, action TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS qa_history(
    id INTEGER PRIMARY KEY, user TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
    created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_usage ON usage(user, action, ts);
CREATE INDEX IF NOT EXISTS idx_quiz_user ON quiz_results(user);
CREATE INDEX IF NOT EXISTS idx_tasks_user_due ON tasks(user, due);
"""


@contextmanager
def connect(path=None):
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("DB error")
        raise
    finally:
        conn.close()


def init_db(path=None):
    with connect(path) as c:
        c.executescript(SCHEMA)


def add_quiz_result(user, topic, correct, total, path=None):
    with connect(path) as c:
        c.execute("INSERT INTO quiz_results(user,topic,correct,total) VALUES(?,?,?,?)",
                  (user, topic.strip(), correct, total))


def topic_accuracy(user, path=None):
    """[{topic, correct, total, accuracy}] sorted weakest first."""
    with connect(path) as c:
        rows = c.execute(
            "SELECT topic, SUM(correct) c, SUM(total) t FROM quiz_results "
            "WHERE user=? GROUP BY lower(topic)", (user,)).fetchall()
    out = [{"topic": r["topic"], "correct": r["c"], "total": r["t"],
            "accuracy": r["c"] / r["t"] if r["t"] else 0.0} for r in rows]
    return sorted(out, key=lambda x: x["accuracy"])


def replace_plan(user, tasks, path=None):
    with connect(path) as c:
        c.execute("DELETE FROM tasks WHERE user=?", (user,))
        c.executemany("INSERT INTO tasks(user,topic,due,minutes,kind) VALUES(?,?,?,?,?)",
                      [(user, t["topic"], t["due"], t["minutes"], t["kind"]) for t in tasks])


def get_tasks(user, due=None, path=None):
    sql, args = "SELECT * FROM tasks WHERE user=?", [user]
    if due:
        sql += " AND due=?"
        args.append(due)
    with connect(path) as c:
        return [dict(r) for r in c.execute(sql + " ORDER BY due, id", args)]


def set_done(user, task_id, done, path=None):
    with connect(path) as c:  # user check prevents editing other users' tasks
        c.execute("UPDATE tasks SET done=? WHERE id=? AND user=?", (int(done), task_id, user))


def get_review(user, topic, path=None):
    with connect(path) as c:
        r = c.execute("SELECT * FROM reviews WHERE user=? AND topic=?", (user, topic)).fetchone()
    return dict(r) if r else None


def save_review(user, topic, ease, interval, reps, next_due, path=None):
    with connect(path) as c:
        c.execute("INSERT OR REPLACE INTO reviews VALUES(?,?,?,?,?,?)",
                  (user, topic, ease, interval, reps, next_due))


def due_reviews(user, today, path=None):
    with connect(path) as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM reviews WHERE user=? AND next_due<=? ORDER BY next_due", (user, today))]


def allow(user, action, limit, window=3600, path=None):
    """Sliding-window rate limit. True (and records the call) if under `limit` per `window` seconds."""
    now = time.time()
    with connect(path) as c:
        n = c.execute("SELECT COUNT(*) FROM usage WHERE user=? AND action=? AND ts>?",
                      (user, action, now - window)).fetchone()[0]
        if n >= limit:
            return False
        c.execute("INSERT INTO usage VALUES(?,?,?)", (user, action, now))
        c.execute("DELETE FROM usage WHERE ts<?", (now - 86400,))
    return True


def add_history(user, question, answer, path=None):
    with connect(path) as c:
        c.execute("INSERT INTO qa_history(user,question,answer) VALUES(?,?,?)", (user, question, answer))


def get_history(user, limit=10, path=None):
    with connect(path) as c:
        return [dict(r) for r in c.execute(
            "SELECT question, answer, created FROM qa_history WHERE user=? ORDER BY id DESC LIMIT ?",
            (user, limit))]


def delete_user_data(user, path=None):
    """Privacy: erase everything stored about a user (the login account itself is in users.db)."""
    with connect(path) as c:
        for t in ("quiz_results", "tasks", "reviews", "usage", "qa_history"):
            c.execute(f"DELETE FROM {t} WHERE user=?", (user,))
