# studybuddy_streamlit.py
"""
StudyBuddy — Streamlit intake app using Google Gemini API.

Features:
- Student intake form
- Pre-interview questions + scoring
- Gemini-generated city suggestions
- SQLite storage
- Email notification (optional via SMTP)
- PDF report download per student
- Multi-counselor login + assignment
- Dashboard with analytics
- Chatbot mode
- CSV export
"""

import os
import sqlite3
import uuid
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO

from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai

# -------------------------------------
# Load environment variables
# -------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro")

ADMIN_PASSWORD = os.getenv("STUDYBUDDY_ADMIN_PW", "adminpass")
COUNSELOR_PASSWORD = os.getenv("COUNSELOR_PASSWORD", "counselorpass")

CALENDLY_EMBED_LINK = os.getenv(
    "CALENDLY_EMBED_LINK",
    "https://calendly.com/your-organization/30min"
)

EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER or "")

# Simple counselor list (can be configured as you like)
COUNSELORS: List[str] = ["Counselor A", "Counselor B", "Counselor C"]

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
            pre_interview_answers TEXT,
            pre_interview_score INTEGER,
            counselor TEXT,
            status TEXT,
            created_at TEXT,
            suggestion_text TEXT
        )
        """
    )
    conn.commit()
    return conn


conn = init_db()


# -------------------------------------
# Utility functions
# -------------------------------------
def calculate_pre_interview_score(answers: Dict) -> int:
    """Simple scoring based on IELTS, work experience, and motivation length."""
    score = 0

    # IELTS score
    try:
        ielts = float(answers.get("ielts_score") or 0)
    except ValueError:
        ielts = 0
    if ielts >= 7.0:
        score += 30
    elif ielts >= 6.0:
        score += 20
    elif ielts > 0:
        score += 10

    # Work experience
    try:
        work_years = float(answers.get("work_experience_years") or 0)
    except ValueError:
        work_years = 0
    if work_years >= 3:
        score += 25
    elif work_years >= 1:
        score += 15

    # Motivation length
    motivation = (answers.get("motivation") or "").strip()
    if len(motivation) > 400:
        score += 25
    elif len(motivation) > 200:
        score += 15
    elif len(motivation) > 50:
        score += 5

    # Budget (rough)
    try:
        budget = float(answers.get("budget_estimate") or 0)
    except ValueError:
        budget = 0
    if budget >= 20000:
        score += 10

    # Cap at 100
    return min(score, 100)


def classify_score(score: int) -> str:
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    return "Low"


def send_email_notification(to_email: str, subject: str, body: str):
    if not EMAIL_ENABLED:
        return "Email disabled; skipping."
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD and EMAIL_FROM):
        return "SMTP not fully configured; skipping."

    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return "Email sent."
    except Exception as e:
        return f"Error sending email: {e}"


def generate_pdf_report(row: Dict) -> bytes:
    """Generate a simple PDF report for a student using FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "StudyBuddy Student Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    lines = [
        f"ID: {row.get('id')}",
        f"Name: {row.get('full_name')}",
        f"Email: {row.get('email')}",
        f"Phone: {row.get('phone')}",
        f"Country: {row.get('country_of_origin')}",
        f"Preferred cities: {row.get('preferred_cities')}",
        f"Program: {row.get('program_interest')}",
        f"Qualification: {row.get('current_qualification')}",
        f"Target intake: {row.get('target_intake')}",
        f"Budget estimate: {row.get('budget_estimate')}",
        f"Preferred contact: {row.get('preferred_contact_method')}",
        f"Consent: {bool(row.get('consent'))}",
        f"Pre-interview score: {row.get('pre_interview_score')} ({classify_score(row.get('pre_interview_score') or 0)})",
        f"Counselor: {row.get('counselor')}",
        f"Status: {row.get('status')}",
        "",
        "AI Suggestions:",
        row.get("suggestion_text") or "",
    ]
    for line in lines:
        pdf.multi_cell(0, 8, line)

    buffer = BytesIO()
    pdf.output(buffer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# -------------------------------------
# Gemini AI Suggestion Function (intake)
# -------------------------------------
def ask_deepseek_suggestions(payload: Dict, pre_answers: Dict, score: int) -> str:
    """
    Generate city suggestions using Google Gemini.
    (Function name kept for compatibility)
    """

    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY not set. Add it to your .env file."

    score_band = classify_score(score)

    prompt = (
        f"You are StudyBuddy, a concise professional education consultant.\n"
        f"Student profile:\n"
        f"- Name: {payload.get('full_name')}\n"
        f"- Country: {payload.get('country_of_origin')}\n"
        f"- Program interest: {payload.get('program_interest')}\n"
        f"- Preferred cities: {payload.get('preferred_cities')}\n"
        f"- Qualification: {payload.get('current_qualification')}\n"
        f"- Target intake: {payload.get('target_intake')}\n"
        f"- Budget: {payload.get('budget_estimate')}\n"
        f"- Motivation: {pre_answers.get('motivation')}\n"
        f"- IELTS score: {pre_answers.get('ielts_score')}\n"
        f"- Work experience (years): {pre_answers.get('work_experience_years')}\n"
        f"- Pre-interview score: {score} ({score_band})\n\n"
        "Task:\n"
        "1. Suggest exactly 3 study cities (format: City — one-line reason each).\n"
        "2. Give one short next step (one sentence).\n"
        "Be practical and honest. Do not oversell if score is low."
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        canned = (
            "City suggestions (fallback):\n"
            "1. Toronto — strong CS and business programs.\n"
            "2. Melbourne — high-quality universities.\n"
            "3. Berlin — affordable and tech-focused.\n\n"
            "Next step: Schedule a mock interview using the Calendly link."
        )
        return f"(Gemini error — using fallback)\n\n{canned}"


# -------------------------------------
# Gemini Chatbot Helper
# -------------------------------------
def call_gemini_chat(history: List[Dict[str, str]]) -> str:
    """
    history: list of {"role": "user"|"assistant", "content": str}
    Returns StudyBuddy's reply.
    """
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not set. Add it to your .env file."

    system_prompt = (
        "You are StudyBuddy, a professional, friendly and concise study-abroad advisor. "
        "You help students with university selection, country comparison, exam planning, "
        "SOP/LOI tips, visa awareness, and realistic guidance about budgets and timelines. "
        "Avoid overpromising. Be specific, practical, and structured."
    )

    model = genai.GenerativeModel(GEMINI_MODEL)

    gemini_history = [{"role": "user", "parts": [system_prompt]}]
    for msg in history:
        if msg["role"] == "user":
            gemini_history.append({"role": "user", "parts": [msg["content"]]})
        else:
            gemini_history.append({"role": "model", "parts": [msg["content"]]})

    try:
        response = model.generate_content(gemini_history)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error from Gemini: {e}"


# -------------------------------------
# Streamlit UI Setup + Custom Styling
# -------------------------------------
st.set_page_config(
    page_title="StudyBuddy",
    page_icon="🎓",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* ============ GLOBAL ============ */
.stApp {
    background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #000 100%);
    color: #e5e7eb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.7);
    border-radius: 999px;
}

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.98);
    border-right: 1px solid rgba(148,163,184,0.3);
}
section[data-testid="stSidebar"] .css-1d391kg, 
section[data-testid="stSidebar"] .css-12oz5g7 {
    padding-top: 1rem !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e5e7eb !important;
}
.stSidebar .sidebar-title {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 0.25rem;
}
.stSidebar .sidebar-subtitle {
    font-size: 0.8rem;
    color: #6b7280;
}

/* Sidebar selectbox styling */
.stSidebar .stSelectbox > div > div {
    background: rgba(15,23,42,0.9) !important;
    border-radius: 999px !important;
    border: 1px solid rgba(148,163,184,0.6) !important;
}

/* ============ CARDS & CONTAINERS ============ */
.block-card {
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(15,23,42,0.85));
    border-radius: 30px;
    padding: 24px 26px;
    border: 1px solid rgba(148, 163, 184, 0.55);
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(22px);
}

/* Slim secondary card */
.sub-card {
    background: rgba(15,23,42,0.9);
    border-radius: 20px;
    padding: 16px 18px;
    border: 1px solid rgba(148,163,184,0.35);
}

/* ============ HERO SECTION ============ */
.hero-wrapper {
    padding: 22px 26px;
    border-radius: 28px;
    background: radial-gradient(circle at top left, #38bdf8 0, #0f172a 35%, #020617 100%);
    border: 1px solid rgba(148,163,184,0.4);
    box-shadow: 0 24px 70px rgba(15,23,42,0.9);
    margin-bottom: 1.4rem;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 750;
    letter-spacing: 0.02em;
    color: #f9fafb;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #e5e7eb;
    max-width: 520px;
}
.hero-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.7rem;
}
.hero-chip {
    font-size: 0.78rem;
    padding: 0.24rem 0.7rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(148,163,184,0.5);
    color: #e5e7eb;
}

/* CTA button in hero */
.hero-cta {
    margin-top: 1rem;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.55rem 1.2rem;
    border-radius: 999px;
    border: none;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: #022c22;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
}
.hero-cta span {
    font-size: 1.1rem;
}

/* Right side highlight */
.hero-highlight {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    font-size: 0.82rem;
    color: #d1d5db;
}
.hero-metric-pill {
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.4);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

/* ============ FORM FIELDS ============ */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div > select {
    background-color: rgba(15, 23, 42, 0.95) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.7) !important;
    color: #e5e7eb !important;
    font-size: 0.9rem;
}
.stTextArea textarea {
    min-height: 120px;
}
.stTextInput > label,
.stTextArea > label,
.stSelectbox > label {
    font-size: 0.84rem;
    color: #cbd5f5 !important;
}

/* Checkbox */
.stCheckbox > label {
    font-size: 0.85rem;
}

/* ============ BUTTONS ============ */
.stButton button {
    border-radius: 999px;
    padding: 0.45rem 1.4rem;
    border: none;
    font-weight: 600;
    font-size: 0.9rem;
    background: linear-gradient(135deg, #4f46e5, #22c55e);
    color: #f9fafb;
    box-shadow: 0 10px 25px rgba(15,23,42,0.9);
}
.stButton button:hover {
    filter: brightness(1.03);
    transform: translateY(-1px);
}

/* ============ METRICS ============ */
[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    color: #e5e7eb !important;
}
[data-testid="stMetricLabel"] {
    color: #9ca3af !important;
}

/* ============ CHATBOT ============ */
.chat-bubble-user {
    background: linear-gradient(135deg, #1e40af, #22c55e);
    color: white;
    padding: 10px 14px;
    border-radius: 18px 18px 4px 18px;
    max-width: 100%;
    font-size: 0.95rem;
}
.chat-bubble-assistant {
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid rgba(148, 163, 184, 0.5);
    padding: 10px 14px;
    border-radius: 18px 18px 18px 4px;
    max-width: 100%;
    font-size: 0.95rem;
}

/* Chat "pill" hint */
.chat-hint-pill {
    display: inline-flex;
    gap: 0.35rem;
    align-items: center;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    font-size: 0.76rem;
    color: #9ca3af;
}

/* ============ TABLE / EXPANDERS ============ */
[data-testid="stExpander"] div[role="button"] {
    background: rgba(15,23,42,0.9);
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.45);
}
[data-testid="stExpander"] {
    border-radius: 18px;
    margin-bottom: 0.6rem;
}

/* Links */
a {
    color: #38bdf8;
}
a:hover {
    text-decoration: none;
}

/* Divider text */
.section-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #6b7280;
}

/* Small tag line */
.text-soft {
    font-size: 0.8rem;
    color: #9ca3af;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Hero header
# Hero / landing header
col_hero_left, col_hero_right = st.columns([2, 1], gap="large")

with col_hero_left:
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-title">
                StudyBuddy — AI Student Advisory Platform
            </div>
            <div class="hero-subtitle">
                Collect student profiles, auto-score them, generate AI city suggestions,
                and manage everything from one clean dashboard with counselor views and a built-in chatbot.
            </div>
            <div class="hero-chip-row">
                <div class="hero-chip">AI city suggestions</div>
                <div class="hero-chip">Pre-interview scoring</div>
                <div class="hero-chip">Counselor workspace</div>
                <div class="hero-chip">PDF reports</div>
                <div class="hero-chip">Chatbot powered by Gemini</div>
            </div>
            <button class="hero-cta">
                <span>🎓</span> Start with student intake
            </button>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_hero_right:
    st.markdown(
        """
        <div class="hero-wrapper" style="padding: 16px 18px; min-height: 100%;">
            <div class="hero-highlight">
                <div class="hero-metric-pill">
                    <span>⚡</span>
                    <span>Instant AI city suggestions per profile</span>
                </div>
                <div class="hero-metric-pill">
                    <span>📊</span>
                    <span>Pre-interview scores out of 100</span>
                </div>
                <div class="hero-metric-pill">
                    <span>👥</span>
                    <span>Admin & counselor views separated</span>
                </div>
                <div class="hero-metric-pill">
                    <span>🤖</span>
                    <span>Chatbot for study-abroad questions</span>
                </div>
                <div class="text-soft">
                    Use the sidebar to switch between modes: 
                    <strong>Apply, Dashboard, Admin, Counselor, Chatbot.</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar mode selection
mode = st.sidebar.selectbox(
    "Mode",
    ["Apply (Student)", "Dashboard", "Admin", "Counselor", "Chatbot"]
)

# -------------------------------------
# Student Intake Mode
# -------------------------------------
if mode == "Apply (Student)":
    with st.container():
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Quick intake")

        col1, col2 = st.columns(2)
        with st.form("student_form"):
            with col1:
                full_name = st.text_input("Full name")
                email = st.text_input("Email")
                phone = st.text_input("Phone (optional)")
                country = st.text_input("Country of origin")
                preferred_cities = st.text_input("Preferred city or leave blank for suggestions")
                program = st.selectbox("Program interest", ["Masters", "Bachelors", "PhD", "Language", "Other"])
                qualification = st.text_input("Current qualification (e.g. B.Tech, GPA/Percentage)")
            with col2:
                intake = st.text_input("Target intake (e.g., 2026-09)")
                budget = st.text_input("Budget estimate (approx, in USD)")
                contact_method = st.selectbox("Preferred contact", ["Email", "Phone / WhatsApp"])
                motivation = st.text_area("Why do you want to study abroad?")
                ielts_score = st.text_input("IELTS/TOEFL score (if any)")
                work_experience_years = st.text_input("Work experience (years, can be 0)")
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

                pre_answers = {
                    "motivation": motivation,
                    "ielts_score": ielts_score,
                    "work_experience_years": work_experience_years,
                    "budget_estimate": budget,
                }

                score = calculate_pre_interview_score(pre_answers)

                with st.spinner(f"Generating personalized suggestions... (Score: {score})"):
                    suggestion = ask_deepseek_suggestions(payload, pre_answers, score)

                # Save to DB
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO students (
                        id, full_name, email, phone, country_of_origin, preferred_cities,
                        program_interest, current_qualification, target_intake, budget_estimate,
                        preferred_contact_method, consent, pre_interview_answers, pre_interview_score,
                        counselor, status, created_at, suggestion_text
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        str(pre_answers),
                        score,
                        None,
                        "New",
                        now,
                        suggestion,
                    )
                )
                conn.commit()

                # Email notification (optional)
                status_msg = send_email_notification(
                    email,
                    "Your StudyBuddy suggestions",
                    f"Hi {full_name},\n\nHere are your suggestions:\n\n{suggestion}\n\nThank you,\nStudyBuddy"
                )
                if EMAIL_ENABLED:
                    st.info(status_msg)

                st.success(f"Your details were saved! Pre-interview score: {score} ({classify_score(score)})")
                st.subheader("Personalized suggestions")
                st.write(suggestion)

                st.markdown("---")
                st.subheader("Schedule a mock interview")
                st.markdown(f"[Click here to schedule →]({CALENDLY_EMBED_LINK})")

        st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------
# Dashboard with analytics
# -------------------------------------
elif mode == "Dashboard":
    with st.container():
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Dashboard — Analytics")

        df = pd.read_sql_query("SELECT * FROM students", conn)
        if df.empty:
            st.info("No data yet.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total students", len(df))
            col2.metric("Avg pre-interview score", f"{df['pre_interview_score'].fillna(0).mean():.1f}")
            col3.metric("High-score students (>=75)", (df["pre_interview_score"] >= 75).sum())

            st.markdown("---")
            st.subheader("Program distribution")
            prog_counts = df["program_interest"].value_counts()
            st.bar_chart(prog_counts)

            st.subheader("Country distribution")
            country_counts = df["country_of_origin"].value_counts()
            st.bar_chart(country_counts)

            st.subheader("Score distribution")
            st.bar_chart(df["pre_interview_score"].fillna(0))

        st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------
# Admin Mode
# -------------------------------------
elif mode == "Admin":
    with st.container():
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Admin — View & Manage Submissions")

        pw = st.text_input("Enter admin password", type="password")
        if pw != ADMIN_PASSWORD:
            st.warning("Incorrect password.")
        else:
            st.success("Admin access granted")

            df = pd.read_sql_query("SELECT * FROM students ORDER BY created_at DESC", conn)
            if df.empty:
                st.info("No submissions yet.")
            else:
                # CSV export
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download all as CSV",
                    data=csv_data,
                    file_name="students_export.csv",
                    mime="text/csv",
                )

                for _, row in df.iterrows():
                    sid = row["id"]
                    name = row["full_name"]
                    prog = row["program_interest"]
                    created = row["created_at"]
                    with st.expander(f"{name} — {prog} — {created}"):
                        st.write("Email:", row["email"])
                        st.write("Phone:", row["phone"])
                        st.write("Country:", row["country_of_origin"])
                        st.write("Preferred cities:", row["preferred_cities"])
                        st.write("Program:", row["program_interest"])
                        st.write("Qualification:", row["current_qualification"])
                        st.write("Score:", f"{row['pre_interview_score']} ({classify_score(row['pre_interview_score'] or 0)})")
                        st.write("Counselor:", row["counselor"])
                        st.write("Status:", row["status"])
                        st.write("AI Suggestion:")
                        st.write(row["suggestion_text"])

                        # Update counselor and status
                        new_counselor = st.selectbox(
                            "Assign counselor",
                            options=[""] + COUNSELORS,
                            index=([""] + COUNSELORS).index(row["counselor"]) if row["counselor"] in COUNSELORS else 0,
                            key=f"couns_{sid}",
                        )
                        new_status = st.selectbox(
                            "Status",
                            options=["New", "In Progress", "Closed"],
                            index=["New", "In Progress", "Closed"].index(row["status"] or "New"),
                            key=f"status_{sid}",
                        )
                        if st.button("Save changes", key=f"save_{sid}"):
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE students SET counselor = ?, status = ? WHERE id = ?",
                                (new_counselor or None, new_status, sid),
                            )
                            conn.commit()
                            st.success("Updated.")

                        # PDF download
                        if st.button("Generate PDF report", key=f"pdf_{sid}"):
                            pdf_bytes = generate_pdf_report(row.to_dict())
                            st.download_button(
                                "Download PDF",
                                data=pdf_bytes,
                                file_name=f"studybuddy_{name}_{sid}.pdf",
                                mime="application/pdf",
                                key=f"pdf_dl_{sid}",
                            )

        st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------
# Counselor Mode
# -------------------------------------
elif mode == "Counselor":
    with st.container():
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("Counselor — My Students")

        counselor_name = st.selectbox("Choose your name", COUNSELORS)
        pw = st.text_input("Counselor password", type="password")

        if pw != COUNSELOR_PASSWORD:
            st.warning("Enter correct counselor password.")
        else:
            st.success(f"Welcome, {counselor_name}!")
            df = pd.read_sql_query(
                "SELECT * FROM students WHERE counselor = ? ORDER BY created_at DESC",
                conn,
                params=(counselor_name,),
            )
            if df.empty:
                st.info("No students assigned to you yet.")
            else:
                for _, row in df.iterrows():
                    sid = row["id"]
                    name = row["full_name"]
                    score = row["pre_interview_score"]
                    created = row["created_at"]
                    with st.expander(f"{name} — Score: {score} — {created}"):
                        st.write("Email:", row["email"])
                        st.write("Phone:", row["phone"])
                        st.write("Country:", row["country_of_origin"])
                        st.write("Preferred cities:", row["preferred_cities"])
                        st.write("Program:", row["program_interest"])
                        st.write("Qualification:", row["current_qualification"])
                        st.write("AI Suggestion:")
                        st.write(row["suggestion_text"])
                        st.code(sid, language="text")

        st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------
# Chatbot Mode (new professional UI)
# -------------------------------------
elif mode == "Chatbot":
    with st.container():
        st.markdown('<div class="block-card">', unsafe_allow_html=True)
        st.subheader("StudyBuddy Chatbot ")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                "Use this for **counselor-grade answers** on countries, universities, SOPs, timelines, and visas."
            )
        with col2:
            if st.button("🧹 New conversation", use_container_width=True):
                st.session_state.chat_history = []

        st.markdown("---")

        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-assistant"
                st.markdown(f'<div class="{bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        user_input = st.chat_input("Ask StudyBuddy anything about studying abroad...")

        if user_input:
            # add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(f'<div class="chat-bubble-user">{user_input}</div>', unsafe_allow_html=True)

            # get reply
            with st.chat_message("assistant"):
                with st.spinner("Thinking like a senior counselor..."):
                    reply = call_gemini_chat(st.session_state.chat_history)
                st.markdown(f'<div class="chat-bubble-assistant">{reply}</div>', unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        st.markdown('</div>', unsafe_allow_html=True)


