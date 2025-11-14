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
        background: radial-gradient(circle at top, #151821 0, #050509 55%);
        color: #F5F5F7;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 900px;
        margin: auto;
    }
    h1.big-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    p.subtitle {
        text-align: center;
        margin-top: 0;
        color: #9CA3AF;
        font-size: 0.9rem;
    }
    .card {
        background: rgba(17, 24, 39, 0.92);
        border-radius: 18px;
        padding: 1.2rem 1.2rem 1.3rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
        backdrop-filter: blur(18px);
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, textarea {
        background-color: #050712 !important;
        border-radius: 999px !important;
    }
    textarea {
        border-radius: 16px !important;
    }
    .stTextInput>label, .stSelectbox>label, .stTextArea>label {
        font-size: 0.8rem;
        color: #E5E7EB;
    }
    .stCheckbox>label {
        font-size: 0.8rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 999px;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.45rem 0;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
    }
    .small-label {
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Chat bubbles */
    .chat-container {
        max-height: 260px;
        overflow-y: auto;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.5rem;
    }
    .chat-bubble {
        padding: 0.45rem 0.75rem;
        border-radius: 16px;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        max-width: 92%;
        line-height: 1.35;
    }
    .chat-user {
        background: #2563EB;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .chat-bot {
        background: #111827;
        color: #E5E7EB;
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<h1 class='big-title'>StudyBuddy</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Smart, quick study-abroad assistant — fill once, get city suggestions & chat follow-up.</p>",
    unsafe_allow_html=True,
)

mode = st.sidebar.selectbox("Mode", ["Apply (Student)", "Admin"])

# Keep student context in session_state for chat
if "student_payload" not in st.session_state:
    st.session_state.student_payload = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hi! I’m StudyBuddy. After you fill your details, you can ask me follow-up questions here.",
        }
    ]


# ----------------------
# Student Apply Mode (with chat box)
# ----------------------
if mode == "Apply (Student)":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### Quick intake form")

        col_left, col_right = st.columns(2)

        with st.form("student_form"):
            with col_left:
                full_name = st.text_input("Full name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                country = st.text_input("Country of origin")
                preferred_cities = st.text_input(
                    "Preferred city / cities",
                    placeholder="e.g. Toronto, Melbourne (or leave blank)",
                )

            with col_right:
                program = st.selectbox(
                    "Program interest", ["Masters", "Bachelors", "PhD", "Language", "Other"]
                )
                qualification = st.text_input(
                    "Current qualification",
                    placeholder="e.g. B.Tech CSE, 8.1 CGPA",
                )
                intake = st.text_input("Target intake", placeholder="e.g. 2026-09")
                budget = st.text_input("Budget (approx)", placeholder="e.g. 20,00,000 INR")
                contact_method = st.selectbox(
                    "Preferred contact", ["Email", "Phone / WhatsApp"]
                )

            consent = st.checkbox(
                "I consent to sharing my info for counselling purposes.",
                value=True,
            )
            submitted = st.form_submit_button("Get city suggestions")

        if submitted:
            if not (full_name and email and consent):
                st.error("Please provide at least your name, email, and consent.")
            else:
                sid = str(uuid.uuid4())
                now = datetime.datetime.utcnow().isoformat()

                payload = {
                    "full_name": full_name,
                    "email": email,
                    "phone": phone,
                    "country_of_origin": country,
                    "preferred_cities": preferred_cities,
                    "program_interest": program,
                    "current_qualification": qualification,
                    "target_intake": intake,
                    "budget_estimate": budget,
                    "preferred_contact_method": contact_method,
                    "consent": int(consent),
                }
                st.session_state.student_payload = payload  # save for chat context

                with st.spinner("Thinking about the best cities for you..."):
                    suggestion = ask_deepseek_suggestions(payload)

                # Save to DB
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO students (
                        id, full_name, email, phone, country_of_origin, preferred_cities,
                        program_interest, current_qualification, target_intake, budget_estimate,
                        preferred_contact_method, consent, created_at, suggestion_text
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        sid,
                        full_name,
                        email,
                        phone,
                        country,
                        preferred_cities,
                        program,
                        qualification,
                        intake,
                        budget,
                        contact_method,
                        int(consent),
                        now,
                        suggestion,
                    ),
                )
                conn.commit()

                st.success("Got it! Here’s what I recommend:")

                st.markdown("#### Personalized suggestions")
                st.write(suggestion)

                st.markdown("---")
                st.markdown("#### Book a quick mock interview")
                st.markdown(
                    f"<p class='small-label'>Pick a time that suits you — a counselor will join you on the call.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"[Open Calendly to schedule →]({CALENDLY_EMBED_LINK})",
                    unsafe_allow_html=True,
                )

        # --- Chat section ---
        st.markdown("---")
        st.markdown("#### Chat with StudyBuddy")

        # Render chat history
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

        # Chat input
        chat_col1, chat_col2 = st.columns([4, 1])
        with chat_col1:
            user_msg = st.text_input(
                "Ask anything about studying abroad",
                placeholder="e.g. Is my budget enough for Canada?",
                key="chat_input",
            )
        with chat_col2:
            send = st.button("Send", key="chat_send")

        if send and user_msg.strip():
            # Add user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_msg.strip()}
            )
            with st.spinner("StudyBuddy is replying..."):
                reply = ask_deepseek_chat(user_msg.strip(), st.session_state.student_payload)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )
            # Clear input for next message
            st.session_state.chat_input = ""

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

