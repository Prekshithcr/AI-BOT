# studybuddy_streamlit.py
"""
StudyBuddy — Simple Streamlit intake app using DeepSeek for suggestions.

Environment variables expected (set in .env):
  - DEEPSEEK_API_KEY        (required)
  - DEEPSEEK_BASE_URL       (optional, default: https://api.deepseek.com/v1)
  - DEEPSEEK_MODEL          (optional, default: deepseek-chat)
  - STUDYBUDDY_ADMIN_PW     (optional, default: adminpass)
  - CALENDLY_EMBED_LINK     (optional; default demo link)
"""

import os
import sqlite3
import uuid
import datetime
import requests
import streamlit as st
from typing import Dict
from dotenv import load_dotenv

# ----------------------
# Load environment
# ----------------------
load_dotenv()  # reads .env in the same folder

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
ADMIN_PASSWORD = os.getenv("STUDYBUDDY_ADMIN_PW", "adminpass")
CALENDLY_EMBED_LINK = os.getenv(
    "CALENDLY_EMBED_LINK",
    "https://calendly.com/your-organization/30min"
)

DB_PATH = "students.db"


# ----------------------
# Database helpers
# ----------------------
def init_db(db_path: str = DB_PATH):
    """Create the students table if it does not exist."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            full_name TEXT,
            email TEXT,
            phone TEXT,
            country_of_origin TEXT,
            preferred_cities TEXT,
            program_interest TEXT,
            current_qualification TEXT,
            target_intake TEXT,
            budget_estimate TEXT,
            preferred_contact_method TEXT,
            consent INTEGER,
            created_at TEXT,
            suggestion_text TEXT
        )
        """
    )
    conn.commit()
    return conn


conn = init_db()


# ----------------------
# DeepSeek interaction
# ----------------------
def call_deepseek(messages, max_tokens: int = 350, temperature: float = 0.2) -> str:
    """Low-level helper to call DeepSeek with an array of messages."""
    if not DEEPSEEK_API_KEY:
        return "DEEPSEEK_API_KEY not set. Add it to your .env file."

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()

        if "choices" in j and len(j["choices"]) > 0:
            choice = j["choices"][0]
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content", "")
                return content.strip()
            if "text" in choice:
                return choice["text"].strip()

        return f"(Unexpected DeepSeek response format) {j}"

    except requests.exceptions.HTTPError as he:
        try:
            err_json = resp.json()
            return f"DeepSeek API HTTP error: {err_json}"
        except Exception:
            return f"DeepSeek API HTTP error: {he}"
    except Exception as e:
        return f"Error contacting DeepSeek API: {e}"


def ask_deepseek_suggestions(payload: Dict) -> str:
    """Generate 3 city suggestions + next step based on intake form."""
    prompt = (
        f"You are StudyBuddy, a concise professional education consultant.\n"
        f"Student info:\n"
        f"- Name: {payload.get('full_name')}\n"
        f"- Preferred city(s): {payload.get('preferred_cities')}\n"
        f"- Program: {payload.get('program_interest')}\n"
        f"- Country: {payload.get('country_of_origin')}\n"
        f"- Qualification: {payload.get('current_qualification')}\n\n"
        "Suggest exactly 3 study cities (City — one-line reason each) "
        "and one clear next step (one sentence). Keep it concise and actionable."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a concise, friendly professional education consultant named StudyBuddy.",
        },
        {"role": "user", "content": prompt},
    ]
    return call_deepseek(messages, max_tokens=350, temperature=0.2)


def ask_deepseek_chat(question: str, payload: Dict | None) -> str:
    """
    Chat-style follow-up: answer student's question conversationally.
    Uses the intake info (if available) as context.
    """
    context = ""
    if payload:
        context = (
            f"Student profile:\n"
            f"- Name: {payload.get('full_name')}\n"
            f"- Preferred cities: {payload.get('preferred_cities')}\n"
            f"- Program: {payload.get('program_interest')}\n"
            f"- Country: {payload.get('country_of_origin')}\n"
            f"- Qualification: {payload.get('current_qualification')}\n"
            f"- Target intake: {payload.get('target_intake')}\n"
            f"- Budget: {payload.get('budget_estimate')}\n\n"
        )
    prompt = (
        context
        + "You are StudyBuddy, a calm, supportive, realistic study-abroad consultant. "
          "Answer the student's question in 2–3 short sentences. Be helpful but avoid legal/visa guarantees.\n\n"
          f"Student question: {question}"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly, concise study-abroad assistant named StudyBuddy.",
        },
        {"role": "user", "content": prompt},
    ]
    return call_deepseek(messages, max_tokens=220, temperature=0.4)


# ----------------------
# Streamlit UI CONFIG
# ----------------------
st.set_page_config(page_title="StudyBuddy (DeepSeek)", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background: #f3f4f6;
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1100px;
        margin: auto;
    }
    h1.big-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: left;
        margin-bottom: 0.25rem;
        color: #111827;
    }
    p.subtitle {
        margin-top: 0;
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Card shells */
    .card {
        background: #ffffff;
        border-radius: 24px;
        padding: 1.4rem 1.4rem 1.5rem 1.4rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    }

    /* Inputs */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    textarea {
        background-color: #f9fafb !important;
        border-radius: 999px !important;
        border: 1px solid #e5e7eb !important;
    }
    textarea {
        border-radius: 16px !important;
    }
    .stTextInput>label,
    .stSelectbox>label,
    .stTextArea>label {
        font-size: 0.8rem;
        color: #4b5563;
        font-weight: 500;
    }
    .stCheckbox>label {
        font-size: 0.8rem;
        color: #4b5563;
    }

    /* Primary button */
    .stButton>button {
        width: 100%;
        border-radius: 999px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.5rem 0;
        box-shadow: 0 14px 30px rgba(129, 140, 248, 0.4);
    }
    .stButton>button:hover {
        filter: brightness(1.05);
    }

    .small-label {
        font-size: 0.8rem;
        color: #6b7280;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Chat area */
    .chat-column-card {
        background: #ffffff;
        border-radius: 24px;
        padding: 0.9rem 1rem;
        border: 1px solid #e5e7eb;
        height: 420px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    /* Left conversation list (fake) */
    .chat-list-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #4b5563;
        font-weight: 600;
    }
    .chat-list-search {
        border-radius: 999px;
        background: #f3f4f6;
        padding: 0.35rem 0.75rem;
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }
    .chat-list-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.4rem 0.3rem;
        border-radius: 12px;
        cursor: default;
        font-size: 0.8rem;
    }
    .chat-list-item.active {
        background: #eef2ff;
    }
    .chat-avatar {
        width: 28px;
        height: 28px;
        border-radius: 999px;
        background: #e5e7eb;
    }
    .chat-list-name {
        font-weight: 600;
        color: #111827;
        font-size: 0.8rem;
    }
    .chat-list-preview {
        color: #6b7280;
        font-size: 0.75rem;
    }

    /* Chat bubbles */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 0.3rem 0.1rem 0.4rem 0.1rem;
        margin-bottom: 0.25rem;
    }
    .chat-bubble {
        padding: 0.45rem 0.75rem;
        border-radius: 16px;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        max-width: 85%;
        line-height: 1.4;
    }
    .chat-user {
        background: #4f46e5;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .chat-bot {
        background: #f3f4ff;
        color: #111827;
        border-bottom-left-radius: 4px;
        border: 1px solid #e5e7ff;
        margin-right: auto;
    }

    .chat-input-wrapper {
        border-radius: 999px;
        background: #f3f4f6;
        padding: 0.25rem 0.4rem;
        border: 1px solid #e5e7eb;
    }

    /* Right profile */
    .profile-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .profile-avatar {
        width: 40px;
        height: 40px;
        border-radius: 999px;
        background: linear-gradient(135deg, #a855f7, #6366f1);
    }
    .profile-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #111827;
    }
    .profile-tag {
        font-size: 0.75rem;
        color: #6b7280;
    }
    .profile-pill {
        display: inline-block;
        padding: 0.1rem 0.6rem;
        border-radius: 999px;
        background: #ecfdf3;
        color: #166534;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.15rem;
    }
    .profile-section-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7280;
        margin-top: 0.55rem;
        margin-bottom: 0.25rem;
    }
    .profile-item {
        font-size: 0.78rem;
        color: #4b5563;
        margin-bottom: 0.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Title
        # --- Chat section in 3-column theme ---
        st.markdown("---")
        st.markdown("#### Live chat")

        col_chat_left, col_chat_mid, col_chat_right = st.columns([1.1, 1.8, 1.3])

        # LEFT: conversation list (dummy UI just for look)
        with col_chat_left:
            st.markdown(
                """
                <div class="chat-column-card">
                    <div class="chat-list-header">
                        <span>All conversations</span>
                        <span style="font-size:0.75rem;color:#9ca3af;">PRO</span>
                    </div>
                    <div class="chat-list-search">Search</div>

                    <div class="chat-list-item active">
                        <div class="chat-avatar"></div>
                        <div>
                            <div class="chat-list-name">You</div>
                            <div class="chat-list-preview">Study plans & budget</div>
                        </div>
                    </div>
                    <div class="chat-list-item">
                        <div class="chat-avatar"></div>
                        <div>
                            <div class="chat-list-name">Demo student</div>
                            <div class="chat-list-preview">Looking for Masters...</div>
                        </div>
                    </div>
                    <div class="chat-list-item">
                        <div class="chat-avatar"></div>
                        <div>
                            <div class="chat-list-name">Sample chat</div>
                            <div class="chat-list-preview">Scholarship options</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # MIDDLE: actual chatbot
        with col_chat_mid:
            st.markdown('<div class="chat-column-card">', unsafe_allow_html=True)

            # history
            chat_container = st.container()
            with chat_container:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    css_class = "chat-bot" if msg["role"] == "assistant" else "chat-user"
                    st.markdown(
                        f"<div class='chat-bubble {css_class}'>{msg['content']}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # input row
            chat_input_col, chat_button_col = st.columns([4, 1])
            with chat_input_col:
                user_msg = st.text_input(
                    "",
                    placeholder="Type your message…",
                    key="chat_input",
                    label_visibility="collapsed",
                )
            with chat_button_col:
                send = st.button("Send", key="chat_send")

            if send and user_msg.strip():
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_msg.strip()}
                )
                with st.spinner("StudyBuddy is replying..."):
                    reply = ask_deepseek_chat(
                        user_msg.strip(), st.session_state.student_payload
                    )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )
                st.session_state.chat_input = ""

            st.markdown("</div>", unsafe_allow_html=True)

        # RIGHT: student profile / info
        with col_chat_right:
            st.markdown('<div class="chat-column-card">', unsafe_allow_html=True)

            payload = st.session_state.student_payload or {}
            name = payload.get("full_name") or "New student"
            email = payload.get("email") or "Not provided"
            country = payload.get("country_of_origin") or "—"
            program = payload.get("program_interest") or "—"
            intake = payload.get("target_intake") or "—"
            budget = payload.get("budget_estimate") or "—"

            st.markdown(
                f"""
                <div class="profile-header">
                    <div class="profile-avatar"></div>
                    <div>
                        <div class="profile-name">{name}</div>
                        <div class="profile-tag">{email}</div>
                        <div class="profile-pill">Active student</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="profile-section-title">General info</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="profile-item"><strong>Country:</strong> {country}</div>
                <div class="profile-item"><strong>Program:</strong> {program}</div>
                <div class="profile-item"><strong>Target intake:</strong> {intake}</div>
                <div class="profile-item"><strong>Budget:</strong> {budget}</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="profile-section-title">Notes</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="profile-item">
                    Short chat summary will appear here based on the latest messages.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)


# ----------------------
# Admin Mode (compact)
# ----------------------
elif mode == "Admin":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Admin — submissions overview")

        pw = st.text_input("Admin password", type="password")
        if pw != ADMIN_PASSWORD:
            st.warning("Enter the correct admin password to view records.")
        else:
            st.success("Admin access granted")

            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, full_name, email, phone, preferred_cities,
                       program_interest, created_at, suggestion_text
                FROM students
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()

            if not rows:
                st.info("No submissions yet.")
            else:
                # Show as short table first
                st.caption("Recent submissions")
                preview = [
                    {
                        "Name": r[1],
                        "Email": r[2],
                        "Program": r[5],
                        "Cities": r[4],
                        "Created": r[6],
                    }
                    for r in rows[:50]
                ]
                st.dataframe(preview, use_container_width=True, height=250)

                st.markdown("---")
                st.caption("Tap a record for full details")

                for r in rows:
                    sid, name, email, phone, pref, prog, created_at, suggestion_text = r
                    with st.expander(f"{name} — {prog} — {created_at}"):
                        st.write("**Email:**", email)
                        st.write("**Phone:**", phone)
                        st.write("**Preferred cities:**", pref)
                        st.write("**AI Suggestion:**")
                        st.write(suggestion_text)
                        st.code(sid, language="text")

        st.markdown("</div>", unsafe_allow_html=True)

