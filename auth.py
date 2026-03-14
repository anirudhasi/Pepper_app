"""
auth.py — Authentication module for Pepper Price Forecasting App.

User registry is stored in users.json (never committed with plain-text passwords).
Passwords are hashed with SHA-256 + per-user salt.

Admin can add / remove / reset users by editing users.json or via the
admin panel built into the app (admin role only).
"""

import hashlib
import hmac
import json
import os
import secrets
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
USERS_FILE = Path(__file__).parent / "users.json"

# Session timeout in minutes (0 = no timeout)
SESSION_TIMEOUT_MINUTES = 120


# ─────────────────────────────────────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: str) -> str:
    """SHA-256 hash of password + salt. Returns hex string."""
    key = (salt + password).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def _new_salt() -> str:
    return secrets.token_hex(16)


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ─────────────────────────────────────────────────────────────────────────────
# User registry (users.json)
# ─────────────────────────────────────────────────────────────────────────────

def _load_users() -> dict:
    if not USERS_FILE.exists():
        # Bootstrap with a default admin account on first run
        _bootstrap_users()
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _bootstrap_users():
    """Create users.json with a default admin account."""
    salt = _new_salt()
    users = {
        "admin": {
            "name":       "Administrator",
            "role":       "admin",
            "salt":       salt,
            "password_hash": _hash_password("Admin@1234", salt),
            "created_at": datetime.utcnow().isoformat(),
            "active":     True,
        }
    }
    _save_users(users)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_all_users() -> dict:
    return _load_users()


def user_exists(username: str) -> bool:
    return username.lower() in _load_users()


def add_user(username: str, name: str, password: str, role: str = "viewer") -> tuple[bool, str]:
    """Add a new user. Returns (success, message)."""
    users = _load_users()
    uname = username.strip().lower()
    if not uname:
        return False, "Username cannot be empty."
    if uname in users:
        return False, f"Username '{uname}' already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = _new_salt()
    users[uname] = {
        "name":          name.strip(),
        "role":          role,
        "salt":          salt,
        "password_hash": _hash_password(password, salt),
        "created_at":    datetime.utcnow().isoformat(),
        "active":        True,
    }
    _save_users(users)
    return True, f"User '{uname}' created successfully."


def remove_user(username: str) -> tuple[bool, str]:
    users = _load_users()
    uname = username.lower()
    if uname not in users:
        return False, f"User '{uname}' not found."
    if uname == "admin":
        return False, "Cannot delete the admin account."
    del users[uname]
    _save_users(users)
    return True, f"User '{uname}' removed."


def set_user_active(username: str, active: bool) -> tuple[bool, str]:
    users = _load_users()
    uname = username.lower()
    if uname not in users:
        return False, "User not found."
    users[uname]["active"] = active
    _save_users(users)
    state = "enabled" if active else "disabled"
    return True, f"User '{uname}' {state}."


def reset_password(username: str, new_password: str) -> tuple[bool, str]:
    users = _load_users()
    uname = username.lower()
    if uname not in users:
        return False, "User not found."
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    salt = _new_salt()
    users[uname]["salt"]          = salt
    users[uname]["password_hash"] = _hash_password(new_password, salt)
    _save_users(users)
    return True, f"Password for '{uname}' reset successfully."


def authenticate(username: str, password: str) -> tuple[bool, dict | None, str]:
    """
    Try to log in. Returns (success, user_dict_or_None, message).
    """
    users = _load_users()
    uname = username.strip().lower()
    if uname not in users:
        return False, None, "Invalid username or password."
    user = users[uname]
    if not user.get("active", True):
        return False, None, "Your account has been disabled. Contact the administrator."
    if not verify_password(password, user["salt"], user["password_hash"]):
        return False, None, "Invalid username or password."
    return True, user, "Login successful."


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit session helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    """Initialise auth-related session_state keys."""
    for key, default in [
        ("authenticated", False),
        ("username",      ""),
        ("user_name",     ""),
        ("user_role",     ""),
        ("login_time",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def is_authenticated() -> bool:
    init_session()
    if not st.session_state.authenticated:
        return False
    # Session timeout check
    if SESSION_TIMEOUT_MINUTES > 0 and st.session_state.login_time:
        elapsed = datetime.utcnow() - st.session_state.login_time
        if elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            logout()
            st.warning("⏱ Session expired. Please log in again.")
            return False
    return True


def login_user(username: str, user: dict):
    st.session_state.authenticated = True
    st.session_state.username      = username
    st.session_state.user_name     = user["name"]
    st.session_state.user_role     = user["role"]
    st.session_state.login_time    = datetime.utcnow()


def logout():
    for key in ["authenticated", "username", "user_name", "user_role", "login_time"]:
        st.session_state[key] = False if key == "authenticated" else (None if key == "login_time" else "")


def current_user() -> dict:
    return {
        "username": st.session_state.get("username", ""),
        "name":     st.session_state.get("user_name", ""),
        "role":     st.session_state.get("user_role", ""),
    }


def require_admin() -> bool:
    return st.session_state.get("user_role") == "admin"


# ─────────────────────────────────────────────────────────────────────────────
# Login page UI
# ─────────────────────────────────────────────────────────────────────────────

def render_login_page():
    """Full-page login UI. Returns True if user just logged in successfully."""

    # Centre the login card
    st.markdown("""
<style>
/* Hide default header/footer on login page */
[data-testid="stHeader"]  { display: none; }
[data-testid="stToolbar"] { display: none; }
footer { display: none; }

/* Full-page dark background */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1117 50%, #0a1628 100%) !important;
}
[data-testid="stMain"] > div { padding-top: 0 !important; }

.login-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 92vh;
}
.login-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 18px;
    padding: 48px 44px 40px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.login-logo {
    text-align: center;
    margin-bottom: 8px;
    font-size: 3rem;
}
.login-title {
    text-align: center;
    color: #4fc3f7;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
}
.login-subtitle {
    text-align: center;
    color: #546e7a;
    font-size: 0.82rem;
    margin-bottom: 32px;
}
/* Input labels */
[data-testid="stTextInput"] label {
    color: #90caf9 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
}
/* Login button */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #1565c0, #4fc3f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px !important;
    width: 100% !important;
    margin-top: 8px;
    transition: opacity 0.2s;
}
[data-testid="stFormSubmitButton"] button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

    # Three-column centering trick
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown('<div class="login-logo">🌿</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Pepper Price Forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Karnataka APMC Market Intelligence · Secure Access</div>',
                    unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password.")
                return False
            success, user, msg = authenticate(username, password)
            if success:
                login_user(username.strip().lower(), user)
                st.success(f"Welcome, {user['name']}!")
                st.rerun()
                return True
            else:
                st.error(f"🔒 {msg}")
                return False

        st.markdown("""
<div style='text-align:center;margin-top:24px;color:#37474f;font-size:0.72rem;'>
Access is restricted to authorised users only.<br>
Contact your administrator to request access.
</div>
""", unsafe_allow_html=True)

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Admin panel UI
# ─────────────────────────────────────────────────────────────────────────────

def render_admin_panel():
    """Admin user management panel — only visible to admin role."""
    if not require_admin():
        st.error("⛔ Access denied — admin only.")
        return

    st.markdown("## 🔐 User Management")
    st.markdown("---")

    users = get_all_users()

    # ── Current users table ───────────────────────────────────────────────────
    st.markdown("### 👥 Registered Users")
    import pandas as pd
    rows = []
    for uname, u in users.items():
        rows.append({
            "Username":   uname,
            "Full Name":  u["name"],
            "Role":       u["role"],
            "Status":     "✅ Active" if u.get("active", True) else "🚫 Disabled",
            "Created":    u.get("created_at", "—")[:10],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Add user ──────────────────────────────────────────────────────────────
    with col1:
        st.markdown("### ➕ Add New User")
        with st.form("add_user_form"):
            new_uname = st.text_input("Username", placeholder="e.g. john.doe")
            new_name  = st.text_input("Full Name", placeholder="e.g. John Doe")
            new_pass  = st.text_input("Password", type="password", placeholder="Min 6 characters")
            new_role  = st.selectbox("Role", ["viewer", "admin"])
            if st.form_submit_button("Add User", use_container_width=True):
                ok, msg = add_user(new_uname, new_name, new_pass, new_role)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # ── Reset password ────────────────────────────────────────────────────────
    with col2:
        st.markdown("### 🔑 Reset Password")
        with st.form("reset_pass_form"):
            r_uname  = st.selectbox("Select User", list(users.keys()), key="reset_sel")
            r_pass   = st.text_input("New Password", type="password", placeholder="Min 6 characters")
            if st.form_submit_button("Reset Password", use_container_width=True):
                ok, msg = reset_password(r_uname, r_pass)
                st.success(msg) if ok else st.error(msg)

    st.markdown("---")
    col3, col4 = st.columns(2)

    # ── Enable / disable ──────────────────────────────────────────────────────
    with col3:
        st.markdown("### 🔄 Enable / Disable User")
        with st.form("toggle_form"):
            t_uname = st.selectbox("Select User",
                                   [u for u in users if u != "admin"], key="toggle_sel")
            t_action = st.radio("Action", ["Enable", "Disable"], horizontal=True)
            if st.form_submit_button("Apply", use_container_width=True):
                ok, msg = set_user_active(t_uname, t_action == "Enable")
                st.success(msg) if ok else st.error(msg)
                st.rerun()

    # ── Remove user ───────────────────────────────────────────────────────────
    with col4:
        st.markdown("### 🗑️ Remove User")
        with st.form("remove_form"):
            rm_uname = st.selectbox("Select User",
                                    [u for u in users if u != "admin"], key="remove_sel")
            st.warning(f"This permanently deletes user **{rm_uname}**.")
            confirm = st.checkbox("I confirm I want to delete this user")
            if st.form_submit_button("Remove User", use_container_width=True):
                if confirm:
                    ok, msg = remove_user(rm_uname)
                    st.success(msg) if ok else st.error(msg)
                    st.rerun()
                else:
                    st.error("Please tick the confirmation checkbox first.")
