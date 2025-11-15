# studybuddy_streamlit.py
"""
StudyBuddy — Streamlit intake app using Google Gemini API for study abroad suggestions.

Environment variables expected:
  - GEMINI_API_KEY
  - GEMINI_MODEL                (optional, default: "gemini-pro")
  - STUDYBUDDY_ADMIN_PW         (optional, default: "adminpass")
  - CALENDLY_EMBED_LINK         (optional, default test link)
"""

import os
import sqlite3
import uuid
import datetime
import requests
import streamlit as st
from typing import Dict
from dotenv import load_dotenv
import google.generativeai as genai

# -------------------------------------
# Load environment variables
# -------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")
ADMIN_PASSWORD = os.getenv("STUDYBUDDY_ADMIN_PW", "adminpass")
CALENDLY_EMBED_LINK = os.getenv(
    "CALENDLY_EMBED_LINK",
    "https://calendly.com/your-organization/30min"
)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# SQLite DB path
DB_PATH = "students.db"


# -------------------------------------
# Database Setup
# -------------------------------------
def init_db(db_path: str = DB_PATH):
    """Initialize SQLite database and create table if not exists."""
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


# -------------------------------------
# Gemini AI Suggestion Function
# -------------------------------------
def ask_deepseek_suggestions(payload: Dict) -> str:
    """
    Generate city suggestions using Google Gemini.
    (The function name is kept same for compatibility)
    """

    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY not set. Add it to your .env file."

    prompt = (
        f"You are StudyBuddy, a concise professional education consultant.\n"
        f"Student info:\n"
        f"- Name: {payload.get('full_name')}\n"
        f"- Preferred city(s): {payload.get('preferred_cities')}\n"
        f"- Program: {payload.get('program_interest')}\n"
        f"- Country: {payload.get('country_of_origin')}\n"
        f"- Qualification: {payload.get('current_qualification')}\n\n"
        "Suggest exactly 3 study cities (City — one-line reason each) and "
        "one clear next step (one sentence). Be short and actionable."
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        canned = (
            "City suggestions (fallback):\n"
            "1. Toronto — strong CS and business programs.\n"
            "2. Melbourne — high-quality universities and research.\n"
            "3. Berlin — affordable education and tech exposure.\n\n"
            "Next step: Schedule a mock interview using Calendly."
        )
        return f"(Gemini error — using fallback)\n\n{canned}"


# -------------------------------------
# Streamlit UI Setup
# -------------------------------------
st.set_page_config(page_title="StudyBuddy", layout="centered")
st.title("StudyBuddy")


mode = st.sidebar.selectbox("Mode", ["Apply (Student)", "Admin"])


# -------------------------------------
# Student Intake Mode
# -------------------------------------
if mode == "Apply (Student)":

    st.header("Quick intake form")

    with st.form("student_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
        country = st.text_input("Country of origin")
        preferred_cities = st.text_input("Preferred city or leave blank for suggestions")
        program = st.selectbox("Program interest", ["Masters", "Bachelors", "PhD", "Language", "Other"])
        qualification = st.text_input("Current qualification (e.g. B.Tech, GPA/Percentage)")
        intake = st.text_input("Target intake (e.g., 2026-09)")
        budget = st.text_input("Budget estimate (approx)")
        contact_method = st.selectbox("Preferred contact", ["Email", "Phone / WhatsApp"])
        consent = st.checkbox("I consent to sharing my info with the consultancy")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not (full_name and email and consent):
            st.error("Please provide your name, email and consent.")
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
                )
            )
            conn.commit()

            st.success("Your details were saved!")
            st.subheader("Personalized suggestions")
            st.write(suggestion)

            st.markdown("---")
            st.subheader("Schedule a mock interview")
            st.markdown(f"[Click here to schedule →]({CALENDLY_EMBED_LINK})")


# -------------------------------------
# Admin Mode
# -------------------------------------
elif mode == "Admin":

    st.header("Admin — View Submissions")

    pw = st.text_input("Enter admin password", type="password")

    if pw != ADMIN_PASSWORD:
        st.warning("Incorrect password.")
    else:
        st.success("Admin access granted")

        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, full_name, email, phone, preferred_cities, program_interest,
                   created_at, suggestion_text
            FROM students
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()

        if not rows:
            st.info("No submissions yet.")
        else:
            for r in rows:
                sid, name, email, phone, pref, prog, created_at, suggestion = r
                with st.expander(f"{name} — {prog} — {created_at}"):
                    st.write("Email:", email)
                    st.write("Phone:", phone)
                    st.write("Preferred cities:", pref)
                    st.write("AI Suggestion:")
                    st.write(suggestion)
                    st.code(sid, language="text")

