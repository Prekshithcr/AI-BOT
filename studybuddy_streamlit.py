
# studybuddy_streamlit.py
"""
StudyBuddy — Simple Streamlit intake app using DeepSeek for suggestions.

Environment variables expected (set in .env):
  - DEEPSEEK_API_KEY        (required)
  - DEEPSEEK_BASE_URL       (optional, default: https://api.deepseek.com/v1)
  - DEEPSEEK_MODEL          (optional, default: deepseek-chat)
  - STUDYBUDDY_ADMIN_PW     (optional, default: adminpass)
  - CALENDLY_EMBED_LINK     (optional; default demo link)

The app:
  - Shows a student intake form
  - Calls DeepSeek to generate 3 city suggestions + next step
  - Saves everything into a local SQLite DB (students.db)
  - Provides a simple admin view of all submissions
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

# SQLite DB path (local file)
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
def ask_deepseek_suggestions(payload: Dict) -> str:
    """
    Call DeepSeek's chat completions endpoint and return the assistant's text.
    Returns a friendly error message if something goes wrong.
    """
    if not DEEPSEEK_API_KEY:
        return "DEEPSEEK_API_KEY not set. Add it to your .env file."

    # Compose prompt for the model
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

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise professional education consultant named StudyBuddy."
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 350,
        "temperature": 0.2,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()

        # DeepSeek uses OpenAI-compatible format
        if "choices" in j and len(j["choices"]) > 0:
            choice = j["choices"][0]
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content", "")
                return content.strip()
            if "text" in choice:  # fallback if text field used
                return choice["text"].strip()

        return f"(Unexpected DeepSeek response format) {j}"

    except requests.exceptions.HTTPError as he:
        # Try to decode error body
        try:
            err_json = resp.json()
            return f"DeepSeek API HTTP error: {err_json}"
        except Exception:
            return f"DeepSeek API HTTP error: {he}"
    except Exception as e:
        return f"Error contacting DeepSeek API: {e}"

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="StudyBuddy (DeepSeek)", layout="centered")
st.title("StudyBuddy — Student Intake Assistant (DeepSeek)")

mode = st.sidebar.selectbox("Mode", ["Apply (Student)", "Admin"])

# ----------------------
# Student Apply Mode
# ----------------------
if mode == "Apply (Student)":
    st.header("Quick intake form")

    with st.form("student_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
        country = st.text_input("Country of origin")
        preferred_cities = st.text_input(
            "Preferred city or leave blank for suggestions (comma separated)"
        )
        program = st.selectbox(
            "Program interest", ["Masters", "Bachelors", "PhD", "Language", "Other"]
        )
        qualification = st.text_input(
            "Current qualification (e.g., B.Tech, GPA/Percentage)"
        )
        intake = st.text_input("Target intake (e.g., 2026-09)")
        budget = st.text_input("Budget estimate (approx)")
        contact_method = st.selectbox(
            "Preferred contact", ["Email", "Phone / WhatsApp"]
        )
        consent = st.checkbox(
            "I consent to sharing my info with the consultancy"
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not (full_name and email and consent):
            st.error("Please provide your name, email, and consent to proceed.")
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

            with st.spinner("Generating personalized suggestions..."):
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

            st.success("Thanks — your details were saved.")
            st.subheader("Personalized suggestions")
            st.write(suggestion)

            st.markdown("---")
            st.subheader("Schedule a mock interview")
            st.markdown(
                f"[Schedule mock interview →]({CALENDLY_EMBED_LINK})"
            )
            st.info(
                "After booking in Calendly, our team will contact you with the next steps."
            )

# ----------------------
# Admin Mode
# ----------------------
elif mode == "Admin":
    st.header("Admin — view submissions")
    pw = st.text_input("Enter admin password", type="password")

    if pw != ADMIN_PASSWORD:
        st.warning("Enter correct admin password to view records.")
    else:
        st.success("Admin access granted")
        st.subheader("Submissions (most recent first)")

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
            for r in rows:
                sid, name, email, phone, pref, prog, created_at, suggestion_text = r
                with st.expander(f"{name} — {prog} — {created_at}"):
                    st.write("Email:", email)
                    st.write("Phone:", phone)
                    st.write("Preferred cities:", pref)
                    st.write("AI Suggestion:")
                    st.write(suggestion_text)
                    st.code(sid, language="text")
