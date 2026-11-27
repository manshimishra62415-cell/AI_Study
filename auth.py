"""Username/password authentication for Streamlit, backed by SQLite.

Passwords and recovery codes are never stored in plain text: each is salted
and hashed with PBKDF2-SHA256.
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "users.db"
LOGO_PATH = BASE_DIR / "logo.svg"

ITERATIONS = 200_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
USERNAME_RE = re.compile(r"^[a-z0-9_]{3,20}$")
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I to avoid mix-ups


# ---------- Database ----------
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username   TEXT PRIMARY KEY,
               salt       BLOB NOT NULL,
               pw_hash    BLOB NOT NULL,
               created_at TEXT DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    # add recovery columns to databases created by the earlier version
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    for col in ("rec_salt", "rec_hash"):
        if col not in cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} BLOB")
    return conn


def _hash(secret_text, salt):
    return hashlib.pbkdf2_hmac("sha256", secret_text.encode("utf-8"), salt, ITERATIONS)


def _new_recovery_code():
    raw = "".join(secrets.choice(CODE_ALPHABET) for _ in range(12))
    return "-".join(raw[i:i + 4] for i in range(0, 12, 4))


def _normalise_code(code):
    return re.sub(r"[^A-Za-z0-9]", "", code).upper()


def _check_password_rules(password, confirm):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password != confirm:
        return "Passwords do not match."
    return None


def register_user(username, password, confirm):
    """Create an account. Returns (ok, message, recovery_code)."""
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-20 characters: letters, numbers or underscore.", None
    problem = _check_password_rules(password, confirm)
    if problem:
        return False, problem, None

    salt, rec_salt = os.urandom(16), os.urandom(16)
    code = _new_recovery_code()
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO users (username, salt, pw_hash, rec_salt, rec_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, salt, _hash(password, salt), rec_salt,
                 _hash(_normalise_code(code), rec_salt)),
            )
    except sqlite3.IntegrityError:
        return False, "That username is taken. Choose another one.", None
    return True, "Account created.", code


def verify_user(username, password):
    """Return True if the username and password match."""
    username = username.strip().lower()
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT salt, pw_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    if row is None:
        _hash(password, b"\x00" * 16)  # keep timing similar for unknown users
        return False
    return hmac.compare_digest(_hash(password, row[0]), row[1])


def reset_password(username, code, new_password, confirm):
    """Reset a password using the recovery code. Returns (ok, message, new_code)."""
    username = username.strip().lower()
    problem = _check_password_rules(new_password, confirm)
    if problem:
        return False, problem, None

    generic_error = "Username or recovery code is incorrect."
    with closing(_connect()) as conn, conn:
        row = conn.execute(
            "SELECT rec_salt, rec_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            _hash(_normalise_code(code), b"\x00" * 16)
            return False, generic_error, None
        if not hmac.compare_digest(_hash(_normalise_code(code), row[0]), row[1]):
            return False, generic_error, None

        # code is valid: set the new password and rotate the recovery code
        salt, rec_salt = os.urandom(16), os.urandom(16)
        new_code = _new_recovery_code()
        conn.execute(
            "UPDATE users SET salt=?, pw_hash=?, rec_salt=?, rec_hash=? WHERE username=?",
            (salt, _hash(new_password, salt), rec_salt,
             _hash(_normalise_code(new_code), rec_salt), username),
        )
    return True, "Password updated. You can log in now.", new_code


# ---------- UI helpers ----------
def logo_html(width=88):
    svg = LOGO_PATH.read_text(encoding="utf-8") if LOGO_PATH.exists() else "📚"
    return f'<div style="width:{width}px;margin:0 auto 0.75rem auto;">{svg}</div>'


def _locked_seconds_left():
    return max(0, int(st.session_state.get("locked_until", 0) - time.time()))


def _is_locked():
    wait = _locked_seconds_left()
    if wait:
        st.error(f"Too many failed attempts. Try again in {wait} seconds.")
        return True
    return False


def _record_failure(message):
    fails = st.session_state.get("failed_attempts", 0) + 1
    st.session_state["failed_attempts"] = fails
    if fails >= MAX_FAILED_ATTEMPTS:
        st.session_state["locked_until"] = time.time() + LOCKOUT_SECONDS
        st.session_state["failed_attempts"] = 0
        st.error(f"Too many failed attempts. Try again in {LOCKOUT_SECONDS} seconds.")
    else:
        st.error(message)


def _show_recovery_code(code, heading):
    st.success(heading)
    st.warning(
        "Save this recovery code now. It is the only way to reset your password, "
        "and it is shown just once."
    )
    st.code(code, language=None)


def _handle_login(username, password):
    if _is_locked():
        return
    if not username.strip() or not password:
        st.error("Enter your username and password.")
        return
    if verify_user(username, password):
        st.session_state["failed_attempts"] = 0
        st.session_state["user"] = username.strip().lower()
        st.rerun()
    else:
        _record_failure("Incorrect username or password.")


def _handle_reset(username, code, new_pass, confirm):
    if _is_locked():
        return
    if not username.strip() or not code.strip():
        st.error("Enter your username and recovery code.")
        return
    ok, msg, new_code = reset_password(username, code, new_pass, confirm)
    if ok:
        st.session_state["failed_attempts"] = 0
        _show_recovery_code(new_code, msg + " Your old recovery code no longer works.")
    elif msg.startswith("Username or recovery"):
        _record_failure(msg)
    else:
        st.error(msg)


def login_page():
    _, centre, _ = st.columns([1, 1.3, 1])
    with centre:
        st.markdown(logo_html(), unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center;margin-bottom:0'>AI Study Assistant</h2>"
            "<p style='text-align:center;opacity:.7'>Ask questions, practise with quizzes, "
            "and spot repeated exam questions from your own PDFs.</p>",
            unsafe_allow_html=True,
        )

        tab_login, tab_signup, tab_forgot = st.tabs(
            ["Log in", "Create account", "Forgot password"]
        )

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Log in", type="primary")
            if submitted:
                _handle_login(username, password)
            st.caption("Forgot your password? Use the Forgot password tab.")

        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("Choose a username", key="signup_user")
                new_pass = st.text_input("Password (8+ characters)", type="password", key="signup_pass")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                created = st.form_submit_button("Create account", type="primary")
            if created:
                ok, msg, code = register_user(new_user, new_pass, confirm)
                if ok:
                    _show_recovery_code(code, msg + " You can log in now.")
                else:
                    st.error(msg)

        with tab_forgot:
            st.write("Enter your username and the recovery code you saved when you created your account.")
            with st.form("reset_form"):
                r_user = st.text_input("Username", key="reset_user")
                r_code = st.text_input("Recovery code", key="reset_code", placeholder="XXXX-XXXX-XXXX")
                r_pass = st.text_input("New password (8+ characters)", type="password", key="reset_pass")
                r_conf = st.text_input("Confirm new password", type="password", key="reset_confirm")
                reset = st.form_submit_button("Reset password", type="primary")
            if reset:
                _handle_reset(r_user, r_code, r_pass, r_conf)
            st.caption(
                "Lost your recovery code too? It can't be recovered. "
                "Create a new account instead."
            )


def require_login():
    """Show the login page and stop the script until the user is signed in."""
    if st.session_state.get("user"):
        return st.session_state["user"]
    login_page()
    st.stop()


def sidebar_header(user):
    """Logo, signed-in name and log out button for the sidebar."""
    st.markdown(logo_html(56), unsafe_allow_html=True)
    st.caption(f"Signed in as **{user}**")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    st.divider()