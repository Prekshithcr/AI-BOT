# ===============================================================
# StudyBuddy – Full NiceGUI Application
# ===============================================================
# Features:
# - Student intake form
# - Pre-interview scoring
# - Gemini suggestions
# - SQLite DB
# - Dashboard
# - Admin page (CSV export, PDF, counselor assignment)
# - Counselor login
# - Chatbot with history
# - Ready for Render deployment
# ===============================================================

from nicegui import ui, app
import os
import sqlite3
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from io import BytesIO
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
import google.generativeai as genai

# ----------------------------------------------------------
# Load .env
# ----------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

ADMIN_PASSWORD = os.getenv("STUDYBUDDY_ADMIN_PW", "adminpass")
COUNSELOR_PASSWORD = os.getenv("COUNSELOR_PASSWORD", "counselorpass")

EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER or "")

COUNSELORS = ["Counselor A", "Counselor B", "Counselor C"]

CALENDLY_EMBED_LINK = os.getenv(
    "CALENDLY_EMBED_LINK",
    "https://calendly.com/your-organization/30min"
)

DB_PATH = os.getenv("DB_PATH", "students.db")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------------------------------------
# DB Initialization
# ----------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
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
    """)
    conn.commit()
    return conn

conn = init_db()

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------
def calculate_pre_interview_score(answers: Dict) -> int:
    score = 0

    # IELTS score
    try:
        ielts = float(answers.get("ielts_score") or 0)
    except:
        ielts = 0

    if ielts >= 7.0:
        score += 30
    elif ielts >= 6.0:
        score += 20
    elif ielts > 0:
        score += 10

    # Work experience
    try:
        work = float(answers.get("work_experience_years") or 0)
    except:
        work = 0

    if work >= 3:
        score += 25
    elif work >= 1:
        score += 15

    # Motivation length
    mot = (answers.get("motivation") or "").strip()
    if len(mot) > 400:
        score += 25
    elif len(mot) > 200:
        score += 15
    elif len(mot) > 50:
        score += 5

    # Budget
    try:
        bud = float(answers.get("budget_estimate") or 0)
    except:
        bud = 0

    if bud >= 20000:
        score += 10

    return min(score, 100)


def classify_score(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def send_email_notification(to_email: str, subject: str, body: str):
    if not EMAIL_ENABLED:
        return "Email disabled."

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
        return f"Email error: {e}"


def generate_pdf_report(row: Dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "StudyBuddy Student Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)

    for k, v in row.items():
        line = f"{k}: {v}"
        pdf.multi_cell(0, 8, line)

    buff = BytesIO()
    pdf.output(buff)
    pdf_bytes = buff.getvalue()
    buff.close()

    return pdf_bytes

# ----------------------------------------------------------
# Gemini Functions
# ----------------------------------------------------------
def ask_gemini_suggestions(payload: Dict, pre_answers: Dict, score: int) -> str:

    prompt = f"""
You are StudyBuddy, a concise education consultant.
Student details:
Name: {payload['full_name']}
Country: {payload['country_of_origin']}
Program: {payload['program_interest']}
Preferred cities: {payload['preferred_cities']}
Qualification: {payload['current_qualification']}
Intake: {payload['target_intake']}
Budget: {payload['budget_estimate']}
Motivation: {pre_answers['motivation']}
IELTS: {pre_answers['ielts_score']}
Work Experience: {pre_answers['work_experience_years']}
Score: {score} ({classify_score(score)})

Task:
1. Suggest exactly 3 study-abroad cities with one-line reasons.
2. Give one short next step.
"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except:
        return """(Gemini error — fallback)
1. Toronto — Good for tech and business.
2. Melbourne — Strong universities.
3. Berlin — Affordable and practical.

Next step: Schedule a counseling call.
"""


def ask_gemini_chat(history):
    messages = [
        {"role": "user", "parts": [
            "You are StudyBuddy, a helpful study abroad consultant. Be friendly and concise."
        ]}
    ]

    for h in history:
        if h["role"] == "user":
            messages.append({"role": "user", "parts": [h["content"]]})
        else:
            messages.append({"role": "model", "parts": [h["content"]]})

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(messages)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# ----------------------------------------------------------
# Chat storage
# ----------------------------------------------------------
def get_chat():
    return app.storage.user.setdefault("chat", [])

def clear_chat():
    app.storage.user["chat"] = []

# ----------------------------------------------------------
# UI Styling
# ----------------------------------------------------------
def set_bg():
    ui.query("body").style("""
        background: radial-gradient(circle at top left, #1e293b 0, #020617 40%, #000 100%);
        color: #e5e7eb;
    """)

# ----------------------------------------------------------
# Pages
# ----------------------------------------------------------

# ----------------- HOME PAGE / INTAKE ---------------------
@ui.page("/")
def intake_page():
    set_bg()

    with ui.card().classes("w-full max-w-5xl mx-auto mt-8 bg-slate-900 text-slate-100 p-6 shadow-2xl"):
        ui.label("StudyBuddy — Quick Profile Intake").classes("text-2xl font-bold mb-2")
        ui.label("Fill in details and get instant AI-based guidance.").classes("text-slate-300 mb-4")

        # Form fields
        name = ui.input("Full name").classes("w-full")
        email = ui.input("Email").classes("w-full")
        phone = ui.input("Phone (optional)").classes("w-full")
        country = ui.input("Country of origin").classes("w-full")
        pref_city = ui.input("Preferred city / Optional").classes("w-full")
        program = ui.select(
            ["Masters", "Bachelors", "PhD", "Language", "Other"],
            label="Program Interest"
        ).classes("w-full")
        qual = ui.input("Current qualification").classes("w-full")
        intake = ui.input("Target intake (e.g., 2026-09)").classes("w-full")
        budget = ui.input("Budget (USD approx)").classes("w-full")
        contact = ui.select(["Email", "Phone / WhatsApp"], label="Preferred Contact").classes("w-full")
        motivation = ui.textarea("Why do you want to study abroad?").classes("w-full")
        ielts = ui.input("IELTS/TOEFL score").classes("w-full")
        work = ui.input("Work experience (years)").classes("w-full")
        consent = ui.checkbox("I consent to sharing my info")

        result_box = ui.column()

        async def submit():
            if not name.value or not email.value or not consent.value:
                ui.notify("Name, Email and Consent required.", type="warning")
                return

            pre = {
                "motivation": motivation.value,
                "ielts_score": ielts.value,
                "work_experience_years": work.value,
                "budget_estimate": budget.value,
            }

            score = calculate_pre_interview_score(pre)
            suggestion = ask_gemini_suggestions(
                {
                    "full_name": name.value,
                    "email": email.value,
                    "phone": phone.value,
                    "country_of_origin": country.value,
                    "preferred_cities": pref_city.value,
                    "program_interest": program.value,
                    "current_qualification": qual.value,
                    "target_intake": intake.value,
                    "budget_estimate": budget.value,
                    "preferred_contact_method": contact.value,
                    "consent": int(consent.value),
                },
                pre,
                score
            )

            sid = str(datetime.datetime.utcnow().timestamp()).replace(".", "")
            now = datetime.datetime.utcnow().isoformat()

            cur = conn.cursor()
            cur.execute("""
                INSERT INTO students (
                    id, full_name, email, phone, country_of_origin, preferred_cities,
                    program_interest, current_qualification, target_intake, budget_estimate,
                    preferred_contact_method, consent, pre_interview_answers,
                    pre_interview_score, counselor, status, created_at, suggestion_text
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sid, name.value, email.value, phone.value, country.value, pref_city.value,
                program.value, qual.value, intake.value, budget.value, contact.value,
                int(consent.value), str(pre), score, None, "New", now, suggestion
            ))
            conn.commit()

            result_box.clear()
            with result_box:
                ui.label(f"Your Score: {score} ({classify_score(score)})").classes("text-lg font-semibold")
                ui.markdown(suggestion)
                ui.link("Schedule mock interview", CALENDLY_EMBED_LINK, new_tab=True).classes("mt-2 text-blue-400")

        ui.button("Submit", on_click=submit).classes("mt-4 bg-blue-600 text-white")

    ui.link("Dashboard", "/dashboard").classes("mt-6 mr-4")
    ui.link("Admin", "/admin")
    ui.link("Counselor", "/counselor").classes("ml-4")
    ui.link("Chatbot", "/chat").classes("ml-4")

# ---------------- DASHBOARD --------------------
@ui.page("/dashboard")
def dashboard_page():
    set_bg()
    with ui.card().classes("w-full max-w-6xl mx-auto mt-8 bg-slate-900 text-slate-100 p-6"):
        ui.label("Dashboard — Analytics").classes("text-2xl font-bold mb-4")

        df = pd.read_sql_query("SELECT * FROM students", conn)
        if df.empty:
            ui.label("No data yet.")
            return

        with ui.row().classes("gap-4"):
            with ui.card().classes("p-4 bg-slate-800"):
                ui.label("Total Students")
                ui.label(str(len(df))).classes("text-xl font-bold")
            with ui.card().classes("p-4 bg-slate-800"):
                ui.label("Avg Score")
                ui.label(f"{df['pre_interview_score'].mean():.1f}").classes("text-xl font-bold")
            with ui.card().classes("p-4 bg-slate-800"):
                ui.label("High Scorers (>=75)")
                ui.label(str((df['pre_interview_score'] >= 75).sum())).classes("text-xl font-bold")

        ui.table.from_pandas(
            df[["full_name", "email", "country_of_origin", "program_interest", "pre_interview_score", "created_at"]]
        ).classes("w-full mt-4")

        ui.link("Back", "/")

# ---------------- ADMIN PAGE --------------------
@ui.page("/admin")
def admin_page():
    set_bg()

    with ui.card().classes("w-full max-w-6xl mx-auto mt-8 bg-slate-900 text-slate-100 p-6"):
        ui.label("Admin Panel").classes("text-2xl font-bold mb-4")

        pw = ui.input("Admin Password", password=True)
        output = ui.column().classes("mt-4")

        def login():
            output.clear()
            if pw.value != ADMIN_PASSWORD:
                ui.notify("Invalid password", type="warning")
                return

            df = pd.read_sql_query("SELECT * FROM students", conn)
            if df.empty:
                ui.label("No students yet.", parent=output)
                return

            csv_data = df.to_csv(index=False)
            ui.download(text=csv_data, filename="students.csv", label="Download CSV")

            for _, row in df.iterrows():
                with ui.expansion(f"{row['full_name']} — {row['program_interest']} — {row['created_at']}", parent=output):
                    ui.label(f"Email: {row['email']}")
                    ui.label(f"Phone: {row['phone']}")
                    ui.label(f"Country: {row['country_of_origin']}")
                    ui.label(f"Cities: {row['preferred_cities']}")
                    ui.label(f"Program: {row['program_interest']}")
                    ui.label(f"Score: {row['pre_interview_score']} ({classify_score(row['pre_interview_score'])})")
                    ui.label("Suggestion:")
                    ui.markdown(row["suggestion_text"])

                    counselor = ui.select([""] + COUNSELORS, value=row["counselor"] or "", label="Assign Counselor")
                    status = ui.select(["New", "In Progress", "Closed"], value=row["status"] or "New", label="Status")

                    def save(row_id=row["id"], c=counselor, s=status):
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE students SET counselor=?, status=? WHERE id=?",
                            (c.value or None, s.value, row_id)
                        )
                        conn.commit()
                        ui.notify("Updated!", type="positive")

                    ui.button("Save", on_click=save).classes("mt-2")

                    def dow_pdf(r=row.to_dict()):
                        pdf_bytes = generate_pdf_report(r)
                        ui.download(pdf_bytes, filename=f"{r['full_name']}.pdf")

                    ui.button("Download PDF", on_click=dow_pdf).classes("mt-2")

        ui.button("Login", on_click=login)

# ---------------- COUNSELOR PAGE --------------------
@ui.page("/counselor")
def counselor_page():
    set_bg()

    with ui.card().classes("w-full max-w-5xl mx-auto mt-8 bg-slate-900 p-6 text-slate-100"):
        ui.label("Counselor Login").classes("text-2xl font-bold mb-4")

        name = ui.select(COUNSELORS, label="Your Name")
        pw = ui.input("Password", password=True)

        area = ui.column().classes("mt-4")

        def login():
            area.clear()
            if pw.value != COUNSELOR_PASSWORD:
                ui.notify("Wrong password", type="warning")
                return

            df = pd.read_sql_query(
                "SELECT * FROM students WHERE counselor = ? ORDER BY created_at DESC",
                conn, params=(name.value,)
            )

            if df.empty:
                ui.label("No assigned students.", parent=area)
                return

            for _, row in df.iterrows():
                with ui.expansion(f"{row['full_name']} — Score {row['pre_interview_score']}", parent=area):
                    ui.label(f"Email: {row['email']}")
                    ui.label(f"Phone: {row['phone']}")
                    ui.label(f"Country: {row['country_of_origin']}")
                    ui.markdown(row["suggestion_text"])

        ui.button("Login", on_click=login).classes("mt-2")

# ---------------- CHATBOT --------------------
@ui.page("/chat")
def chat_page():
    set_bg()

    with ui.card().classes("w-full max-w-4xl mx-auto mt-8 bg-slate-900 p-6 text-slate-100"):
        ui.label("StudyBuddy Chatbot").classes("text-2xl font-bold mb-4")

        chat_box = ui.column().classes("h-96 overflow-auto bg-slate-800 p-4 rounded-lg shadow-inner")

        def refresh():
            chat_box.clear()
            for msg in get_chat():
                if msg["role"] == "user":
                    ui.label(msg["content"]).classes(
                        "bg-blue-600 text-white p-2 rounded-xl self-end max-w-sm"
                    ).props("style='margin-left:auto'")
                else:
                    ui.label(msg["content"]).classes(
                        "bg-slate-700 text-white p-2 rounded-xl max-w-sm"
                    )

        refresh()

        user_input = ui.input("Message...").classes("w-full mt-3")
        btn = ui.button("Send")

        async def send_msg():
            text = user_input.value.strip()
            if not text:
                return
            get_chat().append({"role": "user", "content": text})
            user_input.value = ""
            refresh()

            reply = ask_gemini_chat(get_chat())
            get_chat().append({"role": "assistant", "content": reply})
            refresh()

        btn.on("click", send_msg)
        user_input.on("keydown.enter", send_msg)

# ----------------------------------------------------------
# Run (Required for Render)
# ----------------------------------------------------------
if __name__ in ("__main__", "__mp_main__"):
    ui.run(
        title="StudyBuddy (NiceGUI)",
        reload=False,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
    )

