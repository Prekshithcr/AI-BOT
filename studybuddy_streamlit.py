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

# ---------------- Config & Env ----------------
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

ADMIN_PASSWORD = os.getenv('STUDYBUDDY_ADMIN_PW', 'adminpass')
COUNSELOR_PASSWORD = os.getenv('COUNSELOR_PASSWORD', 'counselorpass')

CALENDLY_EMBED_LINK = os.getenv(
    'CALENDLY_EMBED_LINK',
    'https://calendly.com/your-organization/30min'
)

EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
EMAIL_FROM = os.getenv('EMAIL_FROM', SMTP_USER or '')

COUNSELORS: List[str] = ['Counselor A', 'Counselor B', 'Counselor C']

DB_PATH = os.getenv('DB_PATH', 'students.db')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------- DB Setup ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        '''
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
        '''
    )
    conn.commit()
    return conn

conn = init_db()

# ---------------- Utils ----------------
def calculate_pre_interview_score(answers: Dict) -> int:
    score = 0
    # IELTS
    try:
        ielts = float(answers.get('ielts_score') or 0)
    except ValueError:
        ielts = 0
    if ielts >= 7.0:
        score += 30
    elif ielts >= 6.0:
        score += 20
    elif ielts > 0:
        score += 10
    # Work exp
    try:
        work_years = float(answers.get('work_experience_years') or 0)
    except ValueError:
        work_years = 0
    if work_years >= 3:
        score += 25
    elif work_years >= 1:
        score += 15
    # Motivation text length
    motivation = (answers.get('motivation') or '').strip()
    if len(motivation) > 400:
        score += 25
    elif len(motivation) > 200:
        score += 15
    elif len(motivation) > 50:
        score += 5
    # Budget
    try:
        budget = float(answers.get('budget_estimate') or 0)
    except ValueError:
        budget = 0
    if budget >= 20000:
        score += 10

    return min(score, 100)

def classify_score(score: int) -> str:
    if score >= 75:
        return 'High'
    elif score >= 50:
        return 'Medium'
    return 'Low'

def send_email_notification(to_email: str, subject: str, body: str):
    if not EMAIL_ENABLED:
        return 'Email disabled; skipping.'
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD and EMAIL_FROM):
        return 'SMTP not fully configured; skipping.'

    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        return 'Email sent.'
    except Exception as e:
        return f'Error sending email: {e}'

def generate_pdf_report(row: Dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'StudyBuddy Student Report', ln=True, align='C')

    pdf.set_font('Arial', '', 12)
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
        '',
        'AI Suggestions:',
        row.get('suggestion_text') or '',
    ]
    for line in lines:
        pdf.multi_cell(0, 8, line)

    buffer = BytesIO()
    pdf.output(buffer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------------- Gemini helpers ----------------
def ask_gemini_suggestions(payload: Dict, pre_answers: Dict, score: int) -> str:
    if not GEMINI_API_KEY:
        return 'GEMINI_API_KEY not set. Add it to your .env file.'

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
    except Exception:
        canned = (
            "City suggestions (fallback):\n"
            "1. Toronto — strong CS and business programs.\n"
            "2. Melbourne — high-quality universities.\n"
            "3. Berlin — affordable and tech-focused.\n\n"
            "Next step: Schedule a mock interview using the Calendly link."
        )
        return f"(Gemini error — using fallback)\n\n{canned}"

def ask_gemini_chat(history: List[Dict[str, str]]) -> str:
    if not GEMINI_API_KEY:
        return '⚠️ GEMINI_API_KEY not set. Add it to your .env file.'

    system_prompt = (
        'You are StudyBuddy, a professional, friendly and concise study-abroad advisor. '
        'You help students with university selection, country comparison, exam planning, '
        'SOP/LOI tips, visa awareness, and realistic guidance about budgets and timelines. '
        'Avoid overpromising. Be specific, practical, and structured.'
    )
    model = genai.GenerativeModel(GEMINI_MODEL)
    gemini_history = [{'role': 'user', 'parts': [system_prompt]}]
    for msg in history:
        if msg['role'] == 'user':
            gemini_history.append({'role': 'user', 'parts': [msg['content']]})
        else:
            gemini_history.append({'role': 'model', 'parts': [msg['content']]})
    try:
        response = model.generate_content(gemini_history)
        return response.text.strip()
    except Exception as e:
        return f'⚠️ Error from Gemini: {e}'

# ---------------- Per-user storage helpers ----------------
def get_chat_history() -> List[Dict[str, str]]:
    return app.storage.user.setdefault('chat_history', [])

def clear_chat_history():
    app.storage.user['chat_history'] = []

# ---------------- Shared styling ----------------
def set_bg():
    ui.query('body').style(
        'background: radial-gradient(circle at top left, #1e293b 0, #020617 40%, #000 100%); '
        'color: #e5e7eb;'
    )

# ---------------- Pages ----------------
@ui.page('/')
def intake_page():
    set_bg()
    with ui.card().classes('w-full max-w-5xl mx-auto mt-8 bg-slate-900 text-slate-100 shadow-2xl'):
        ui.label('StudyBuddy – Quick Intake').classes('text-2xl font-semibold mb-1')
        ui.label('Let us pre-screen the profile and suggest cities in one step.').classes('text-sm text-slate-300 mb-4')

        with ui.row().classes('w-full gap-4'):
            with ui.column().classes('w-1/2'):
                name_in = ui.input('Full name').classes('w-full')
                email_in = ui.input('Email').classes('w-full')
                phone_in = ui.input('Phone (optional)').classes('w-full')
                country_in = ui.input('Country of origin').classes('w-full')
                pref_city_in = ui.input('Preferred city or leave blank').classes('w-full')
                program_in = ui.select(['Masters', 'Bachelors', 'PhD', 'Language', 'Other'], label='Program interest').classes('w-full')
                qual_in = ui.input('Current qualification (e.g. B.Tech, GPA/Percentage)').classes('w-full')
            with ui.column().classes('w-1/2'):
                intake_in = ui.input('Target intake (e.g., 2026-09)').classes('w-full')
                budget_in = ui.input('Budget estimate (approx, in USD)').classes('w-full')
                contact_in = ui.select(['Email', 'Phone / WhatsApp'], label='Preferred contact').classes('w-full')
                motivation_in = ui.textarea('Why do you want to study abroad?').classes('w-full')
                ielts_in = ui.input('IELTS/TOEFL score (if any)').classes('w-full')
                work_in = ui.input('Work experience (years, can be 0)').classes('w-full')
                consent_in = ui.checkbox('I consent to sharing my info with the consultancy')

        result_area = ui.column().classes('mt-4 w-full')

        async def on_submit():
            full_name = name_in.value
            email = email_in.value
            consent = consent_in.value
            if not (full_name and email and consent):
                ui.notify('Please provide name, email and consent.', type='warning')
                return

            payload = {
                'full_name': full_name,
                'email': email,
                'phone': phone_in.value,
                'country_of_origin': country_in.value,
                'preferred_cities': pref_city_in.value,
                'program_interest': program_in.value,
                'current_qualification': qual_in.value,
                'target_intake': intake_in.value,
                'budget_estimate': budget_in.value,
                'preferred_contact_method': contact_in.value,
                'consent': int(bool(consent)),
            }
            pre_answers = {
                'motivation': motivation_in.value,
                'ielts_score': ielts_in.value,
                'work_experience_years': work_in.value,
                'budget_estimate': budget_in.value,
            }
            score = calculate_pre_interview_score(pre_answers)
            ui.notify(f'Generating suggestions... Score {score}', type='info')

            suggestion = ask_gemini_suggestions(payload, pre_answers, score)
            sid = str(datetime.datetime.utcnow().timestamp()).replace('.', '')
            now = datetime.datetime.utcnow().isoformat()

            cur = conn.cursor()
            cur.execute(
                '''
                INSERT INTO students (
                    id, full_name, email, phone, country_of_origin, preferred_cities,
                    program_interest, current_qualification, target_intake, budget_estimate,
                    preferred_contact_method, consent, pre_interview_answers, pre_interview_score,
                    counselor, status, created_at, suggestion_text
                )
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''',
                (
                    sid,
                    full_name,
                    email,
                    phone_in.value,
                    country_in.value,
                    pref_city_in.value,
                    program_in.value,
                    qual_in.value,
                    intake_in.value,
                    budget_in.value,
                    contact_in.value,
                    int(bool(consent)),
                    str(pre_answers),
                    score,
                    None,
                    'New',
                    now,
                    suggestion,
                )
            )
            conn.commit()

            status_msg = send_email_notification(
                email,
                'Your StudyBuddy suggestions',
                f'Hi {full_name},\n\nHere are your suggestions:\n\n{suggestion}\n\nThank you,\nStudyBuddy'
            )
            if EMAIL_ENABLED:
                ui.notify(status_msg, type='info')

            result_area.clear()
            with result_area:
                ui.label(f'Pre-interview score: {score} ({classify_score(score)})').classes('text-lg font-semibold')
                ui.markdown(suggestion).classes('mt-2')
                ui.link('Schedule a mock interview', CALENDLY_EMBED_LINK, new_tab=True).classes('mt-2 text-blue-300')

        ui.button('Submit', on_click=on_submit).classes('mt-4')
        
    with ui.footer().classes('justify-center text-slate-400 bg-transparent gap-4'):
        ui.link('Intake', '/')
        ui.link('Dashboard', '/dashboard')
        ui.link('Admin', '/admin')
        ui.link('Counselor', '/counselor')
        ui.link('Chatbot', '/chat')

@ui.page('/dashboard')
def dashboard_page():
    set_bg()
    with ui.card().classes('w-full max-w-6xl mx-auto mt-8 bg-slate-900 text-slate-100'):
        ui.label('Dashboard — Analytics').classes('text-2xl font-semibold mb-2')
        df = pd.read_sql_query('SELECT * FROM students', conn)
        if df.empty:
            ui.label('No data yet.').classes('text-slate-300')
            return
        with ui.row().classes('gap-4 mb-4'):
            with ui.card().classes('p-4 bg-slate-800'):
                ui.label('Total students').classes('text-sm text-slate-300')
                ui.label(str(len(df))).classes('text-xl font-semibold')
            with ui.card().classes('p-4 bg-slate-800'):
                ui.label('Avg score').classes('text-sm text-slate-300')
                ui.label(f"{df['pre_interview_score'].fillna(0).mean():.1f}").classes('text-xl font-semibold')
            high = int((df['pre_interview_score'] >= 75).sum())
            with ui.card().classes('p-4 bg-slate-800'):
                ui.label('High-score (>=75)').classes('text-sm text-slate-300')
                ui.label(str(high)).classes('text-xl font-semibold')

        ui.label('Latest submissions').classes('mt-2 mb-1')
        cols = ['full_name', 'email', 'country_of_origin', 'program_interest', 'pre_interview_score', 'created_at']
        ui.table.from_pandas(df[cols]).classes('w-full')

        ui.link('Back to Intake', '/').classes('mt-4 text-blue-300')

@ui.page('/admin')
def admin_page():
    set_bg()
    with ui.card().classes('w-full max-w-6xl mx-auto mt-8 bg-slate-900 text-slate-100'):
        ui.label('Admin — View & Manage Submissions').classes('text-2xl font-semibold mb-2')
        pw = ui.input('Enter admin password', password=True, password_toggle_button=True)
        container = ui.column().classes('mt-4 w-full')

        def check_pw():
            container.clear()
            if pw.value != ADMIN_PASSWORD:
                ui.notify('Incorrect password', type='warning')
                return
            ui.notify('Admin access granted', type='positive')
            with container:
                df = pd.read_sql_query('SELECT * FROM students ORDER BY created_at DESC', conn)
                if df.empty:
                    ui.label('No submissions yet.')
                else:
                    csv_data = df.to_csv(index=False)
                    ui.download(text=csv_data, filename='students_export.csv', label='Download all as CSV')

                    for _, row in df.iterrows():
                        with ui.expansion(f"{row['full_name']} — {row['program_interest']} — {row['created_at']}").classes('mt-2'):
                            ui.label(f"Email: {row['email']}")
                            ui.label(f"Phone: {row['phone']}")
                            ui.label(f"Country: {row['country_of_origin']}")
                            ui.label(f"Preferred cities: {row['preferred_cities']}")
                            ui.label(f"Program: {row['program_interest']}")
                            ui.label(f"Qualification: {row['current_qualification']}")
                            ui.label(f"Score: {row['pre_interview_score']} ({classify_score(row['pre_interview_score'] or 0)})")
                            ui.label(f"Counselor: {row['counselor']}")
                            ui.label(f"Status: {row['status']}")
                            ui.label('AI Suggestion:')
                            ui.markdown(row['suggestion_text'] or '')

                            counselor_select = ui.select([''] + COUNSELORS, value=row['counselor'] or '', label='Assign counselor')
                            status_select = ui.select(['New', 'In Progress', 'Closed'], value=row['status'] or 'New', label='Status')

                            def save(row_id=row['id'], c_select=counselor_select, s_select=status_select):
                                cur = conn.cursor()
                                cur.execute(
                                    'UPDATE students SET counselor = ?, status = ? WHERE id = ?',
                                    (c_select.value or None, s_select.value, row_id),
                                )
                                conn.commit()
                                ui.notify('Updated', type='positive')

                            ui.button('Save changes', on_click=save).classes('mt-2')

                            def make_pdf(row_dict=row.to_dict()):
                                pdf_bytes = generate_pdf_report(row_dict)
                                ui.download(
                                    pdf_bytes,
                                    filename=f"studybuddy_{row_dict['full_name']}_{row_dict['id']}.pdf",
                                    label='Download PDF'
                                )

                            ui.button('Generate PDF report', on_click=make_pdf).classes('mt-2')

        ui.button('Login', on_click=check_pw).classes('mt-3')
        ui.link('Back to Intake', '/').classes('mt-4 text-blue-300')

@ui.page('/counselor')
def counselor_page():
    set_bg()
    with ui.card().classes('w-full max-w-5xl mx-auto mt-8 bg-slate-900 text-slate-100'):
        ui.label('Counselor — My Students').classes('text-2xl font-semibold mb-2')
        counselor_select = ui.select(COUNSELORS, label='Choose your name')
        pw = ui.input('Counselor password', password=True, password_toggle_button=True)
        list_area = ui.column().classes('mt-4 w-full')

        def login_counselor():
            list_area.clear()
            if pw.value != COUNSELOR_PASSWORD:
                ui.notify('Incorrect counselor password', type='warning')
                return
            name = counselor_select.value
            ui.notify(f'Welcome, {name}', type='positive')
            df = pd.read_sql_query(
                'SELECT * FROM students WHERE counselor = ? ORDER BY created_at DESC',
                conn,
                params=(name,),
            )
            if df.empty:
                ui.label('No students assigned to you yet.').classes('text-slate-300')
            else:
                for _, row in df.iterrows():
                    with ui.expansion(f"{row['full_name']} — Score: {row['pre_interview_score']} — {row['created_at']}").classes('mt-2'):
                        ui.label(f"Email: {row['email']}")
                        ui.label(f"Phone: {row['phone']}")
                        ui.label(f"Country: {row['country_of_origin']}")
                        ui.label(f"Preferred cities: {row['preferred_cities']}")
                        ui.label(f"Program: {row['program_interest']}")
                        ui.label(f"Qualification: {row['current_qualification']}")
                        ui.label('AI Suggestion:')
                        ui.markdown(row['suggestion_text'] or '')

        ui.button('Login', on_click=login_counselor).classes('mt-3')
        ui.link('Back to Intake', '/').classes('mt-4 text-blue-300')

@ui.page('/chat')
def chat_page():
    set_bg()
    with ui.card().classes('w-full max-w-4xl mx-auto mt-8 bg-slate-900 text-slate-100'):
        ui.label('StudyBuddy Chatbot (Gemini)').classes('text-2xl font-semibold mb-2')
        with ui.row().classes('w-full justify-between items-center mb-2'):
            ui.label('Ask anything about countries, universities, SOPs, timelines, and visas.').classes('text-sm text-slate-300')

            def reset_chat():
                clear_chat_history()
                render_messages()

            ui.button('New conversation', on_click=reset_chat).props('outline')

        messages = ui.column().classes('w-full gap-2 p-2 border border-slate-700 rounded-lg')
        messages.style('max-height: 60vh; overflow-y: auto;')

        def render_messages():
            messages.clear()
            history = get_chat_history()
            for msg in history:
                if msg['role'] == 'user':
                    with messages:
                        with ui.row().classes('justify-end'):
                            ui.label(msg['content']).classes(
                                'bg-blue-600 text-white px-3 py-2 rounded-xl max-w-xl text-sm'
                            )
                else:
                    with messages:
                        with ui.row().classes('justify-start'):
                            ui.label(msg['content']).classes(
                                'bg-slate-800 text-slate-100 px-3 py-2 rounded-xl max-w-xl text-sm'
                            )

        render_messages()

        with ui.row().classes('mt-2 w-full items-center gap-2'):
            user_input = ui.input('Type your message...').props('clearable').classes('flex-1')
            send_button = ui.button('Send')

        async def on_send():
            text = (user_input.value or '').strip()
            if not text:
                return
            history = get_chat_history()
            history.append({'role': 'user', 'content': text})
            user_input.value = ''
            render_messages()
            reply = ask_gemini_chat(history)
            history.append({'role': 'assistant', 'content': reply})
            render_messages()

        send_button.on('click', on_send)
        user_input.on('keydown.enter', on_send)

        ui.link('Back to Intake', '/').classes('mt-4 text-blue-300')

ui.run(title='StudyBuddy (NiceGUI)')

