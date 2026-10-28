
import os
import json
import time
import uuid
from io import BytesIO
from datetime import date, timedelta
from xml.sax.saxutils import escape

import streamlit as st
from dotenv import load_dotenv
from google import genai

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

# -------------------- CONFIGURATION --------------------

load_dotenv()

DATA_FILE = "taskpilot_data.json"
MODEL_NAME = "gemini-3.6-flash"

st.set_page_config(
    page_title="TaskPilot",
    page_icon="🎯",
    layout="wide",
)

# -------------------- DATA STORAGE --------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"plans": [], "history": []}

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            return {"plans": [], "history": []}

        data.setdefault("plans", [])
        data.setdefault("history", [])
        return data

    except (json.JSONDecodeError, OSError):
        return {"plans": [], "history": []}


def save_data():
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "plans": st.session_state.plans,
                "history": st.session_state.history,
            },
            file,
            indent=4,
            ensure_ascii=False,
        )


if "data_loaded" not in st.session_state:
    saved_data = load_data()
    st.session_state.plans = saved_data["plans"]
    st.session_state.history = saved_data["history"]
    st.session_state.data_loaded = True

if "plans" not in st.session_state:
    st.session_state.plans = []

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- GEMINI AI --------------------

def generate_ai_response(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Check your .env file."
        )

    client = genai.Client(api_key=api_key)
    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            return response.text

        except Exception as error:
            last_error = error
            error_message = str(error).lower()

            retryable = (
                "503" in error_message
                or "429" in error_message
                or "unavailable" in error_message
                or "resource_exhausted" in error_message
                or "temporarily" in error_message
            )

            if retryable and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue

            raise

    raise last_error


def parse_json_response(response_text):
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("AI did not return valid JSON.")

    return json.loads(cleaned[start:end + 1])


# -------------------- PLAN GENERATION --------------------

def create_plan(goal, days, hours_per_day):
    prompt = f"""
You are TaskPilot, a practical study and productivity planning assistant.

Create a realistic {days}-day plan for this goal:
{goal}

Available study time per day: {hours_per_day} hours.
Start date: {date.today().isoformat()}.

Return ONLY valid JSON in this format:
{{
  "title": "Short plan title",
  "goal": "The user's goal",
  "days": [
    {{
      "day": 1,
      "tasks": [
        {{
          "title": "Specific task",
          "description": "Brief explanation of what to do",
          "duration": 60
        }}
      ]
    }}
  ],
  "review_suggestion": "A short suggestion for reviewing progress"
}}

Rules:
- Include exactly {days} days.
- Create 2 to 5 practical tasks per day.
- Duration is in minutes.
- Keep daily duration close to the available time.
- Use clear, achievable tasks suitable for a beginner.
- Avoid vague tasks.
- Do not include Markdown outside the JSON.
"""

    result = parse_json_response(generate_ai_response(prompt))

    daily_plans = result.get("days", [])

    if not daily_plans:
        raise ValueError("The AI did not generate any daily tasks.")

    tasks_by_day = []

    for day_index, day_data in enumerate(daily_plans[:days]):
        tasks = []

        for task_data in day_data.get("tasks", []):
            tasks.append({
                "title": str(task_data.get("title", "Untitled task")),
                "description": str(task_data.get("description", "")),
                "duration": max(0, int(task_data.get("duration", 30))),
                "completed": False,
            })

        tasks_by_day.append({
            "day": day_index + 1,
            "tasks": tasks,
        })

    while len(tasks_by_day) < days:
        tasks_by_day.append({
            "day": len(tasks_by_day) + 1,
            "tasks": [],
        })

    return {
        "id": str(uuid.uuid4()),
        "title": str(result.get("title", "My TaskPilot Plan")),
        "goal": goal,
        "start_date": date.today().isoformat(),
        "days_count": days,
        "hours_per_day": hours_per_day,
        "days": tasks_by_day,
        "review_suggestion": str(
            result.get(
                "review_suggestion",
                "Review completed tasks and adjust your next steps.",
            )
        ),
    }


# -------------------- WORKFLOW AUTOMATION --------------------

def reschedule_plan(plan):
    unfinished = []

    for day_data in plan.get("days", []):
        for task in day_data.get("tasks", []):
            if not task.get("completed", False):
                unfinished.append({
                    "title": task.get("title", ""),
                    "description": task.get("description", ""),
                    "duration": task.get("duration", 30),
                })

    if not unfinished:
        return None

    prompt = f"""
You are TaskPilot, a productivity workflow assistant.

Goal: {plan.get('goal', '')}
New plan start date: {date.today().isoformat()}

Unfinished tasks:
{json.dumps(unfinished, ensure_ascii=False)}

Create a realistic 7-day schedule using these unfinished tasks.
You may divide large tasks into smaller steps.
Do not add unrelated tasks.

Return ONLY valid JSON in this format:
{{
  "title": "Rescheduled plan title",
  "days": [
    {{
      "day": 1,
      "tasks": [
        {{
          "title": "Task title",
          "description": "What to do",
          "duration": 45
        }}
      ]
    }}
  ],
  "review_suggestion": "A short review suggestion"
}}

Include exactly 7 days and use minutes for duration.
"""

    result = parse_json_response(generate_ai_response(prompt))
    new_days = []

    for day_index, day_data in enumerate(result.get("days", [])[:7]):
        tasks = []

        for task_data in day_data.get("tasks", []):
            tasks.append({
                "title": str(task_data.get("title", "Untitled task")),
                "description": str(task_data.get("description", "")),
                "duration": max(0, int(task_data.get("duration", 30))),
                "completed": False,
            })

        new_days.append({
            "day": day_index + 1,
            "tasks": tasks,
        })

    while len(new_days) < 7:
        new_days.append({
            "day": len(new_days) + 1,
            "tasks": [],
        })

    old_total, old_completed, _ = get_task_counts(plan)

    st.session_state.history.append({
        "title": plan.get("title", "Previous plan"),
        "goal": plan.get("goal", ""),
        "archived_on": date.today().isoformat(),
        "completed": old_completed,
        "total": old_total,
    })

    return {
        "id": str(uuid.uuid4()),
        "title": str(result.get("title", "Rescheduled TaskPilot Plan")),
        "goal": plan.get("goal", ""),
        "start_date": date.today().isoformat(),
        "days_count": 7,
        "hours_per_day": plan.get("hours_per_day", 2),
        "days": new_days,
        "review_suggestion": str(
            result.get(
                "review_suggestion",
                "Review your progress at the end of each day.",
            )
        ),
    }


# -------------------- PROGRESS HELPERS --------------------

def get_task_counts(plan):
    tasks = [
        task
        for day_data in plan.get("days", [])
        for task in day_data.get("tasks", [])
    ]

    total = len(tasks)
    completed = sum(
        1 for task in tasks if task.get("completed", False)
    )

    return total, completed, total - completed


def get_task_date(plan, day_index):
    start = date.fromisoformat(plan["start_date"])
    return start + timedelta(days=day_index)


# -------------------- PDF EXPORT --------------------

def create_pdf(plan):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=plan.get("title", "TaskPilot Plan"),
        author="TaskPilot",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TaskPilotTitle",
        parent=styles["Title"],
        fontSize=21,
        leading=26,
        textColor=colors.HexColor("#244A73"),
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "TaskPilotHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#244A73"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "TaskPilotBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceAfter=4,
        wordWrap="CJK",
    )

    small_style = ParagraphStyle(
        "TaskPilotSmall",
        parent=body_style,
        fontSize=8,
        leading=10,
    )

    total, completed, pending = get_task_counts(plan)
    progress = round(completed / total * 100) if total else 0

    story = [
        Paragraph(escape(str(plan.get("title", "TaskPilot Plan"))), title_style),
        Paragraph(
            f"<b>Goal:</b> {escape(str(plan.get('goal', '')))}",
            body_style,
        ),
        Paragraph(
            f"<b>Start date:</b> {escape(str(plan.get('start_date', '')))}",
            body_style,
        ),
        Paragraph(
            f"<b>Duration:</b> {plan.get('days_count', 0)} days",
            body_style,
        ),
        Spacer(1, 5 * mm),
        Paragraph("Progress Summary", heading_style),
    ]

    summary_table = Table(
        [
            ["Total Tasks", "Completed", "Pending", "Progress"],
            [str(total), str(completed), str(pending), f"{progress}%"],
        ],
        colWidths=[40 * mm, 40 * mm, 40 * mm, 40 * mm],
        repeatRows=1,
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244A73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DDE5")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F6FA")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.extend([summary_table, Spacer(1, 5 * mm)])

    for day_index, day_data in enumerate(plan.get("days", [])):
        day_date = get_task_date(plan, day_index)
        day_number = day_data.get("day", day_index + 1)

        day_heading = Paragraph(
            f"Day {day_number} — {day_date.isoformat()}",
            heading_style,
        )

        if not day_data.get("tasks"):
            story.append(KeepTogether([
                day_heading,
                Paragraph("No tasks scheduled.", body_style),
            ]))
            continue

        table_data = [[
            Paragraph("<b>Status</b>", small_style),
            Paragraph("<b>Task and description</b>", small_style),
            Paragraph("<b>Minutes</b>", small_style),
        ]]

        for task in day_data.get("tasks", []):
            status = (
                "Completed"
                if task.get("completed", False)
                else "Pending"
            )

            task_title = escape(str(task.get("title", "Untitled task")))
            task_description = escape(str(task.get("description", "")))

            task_content = f"<b>{task_title}</b>"

            if task_description:
                task_content += f"<br/>{task_description}"

            table_data.append([
                Paragraph(status, small_style),
                Paragraph(task_content, small_style),
                Paragraph(str(task.get("duration", 0)), small_style),
            ])

        task_table = Table(
            table_data,
            colWidths=[27 * mm, 112 * mm, 20 * mm],
            repeatRows=1,
            hAlign="LEFT",
        )

        task_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EDF5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#244A73")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DDE5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.extend([
            KeepTogether([day_heading, task_table]),
            Spacer(1, 3 * mm),
        ])

    story.extend([
        Spacer(1, 3 * mm),
        Paragraph("Review Suggestion", heading_style),
        Paragraph(
            escape(str(plan.get("review_suggestion", ""))),
            body_style,
        ),
        Spacer(1, 5 * mm),
        Paragraph("Generated by TaskPilot", small_style),
    ])

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -------------------- APP HEADER --------------------

st.title("🎯 TaskPilot")
st.caption("Your AI-powered study planner and workflow assistant")

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("Create a new plan")

    with st.form("create_plan_form"):
        goal = st.text_area(
            "What is your goal?",
            placeholder="Example: Prepare for my Python exam",
        )

        days = st.slider(
            "Plan duration (days)",
            min_value=1,
            max_value=30,
            value=7,
        )

        hours_per_day = st.slider(
            "Available hours per day",
            min_value=1,
            max_value=12,
            value=2,
        )

        generate_clicked = st.form_submit_button(
            "✨ Generate AI Plan",
            use_container_width=True,
        )

    if generate_clicked:
        if not goal.strip():
            st.warning("Please enter your goal first.")
        else:
            try:
                with st.spinner("TaskPilot is creating your plan..."):
                    new_plan = create_plan(
                        goal.strip(),
                        days,
                        hours_per_day,
                    )

                st.session_state.plans.append(new_plan)
                st.session_state.active_plan_id = new_plan["id"]
                save_data()
                st.success("Your plan has been created!")

            except Exception as error:
                st.error(f"Could not create your plan: {error}")

# -------------------- SELECT ACTIVE PLAN --------------------

plans = st.session_state.plans

if not plans:
    st.info(
        "Welcome to TaskPilot! Enter a goal in the sidebar and "
        "click **Generate AI Plan** to get started."
    )
    st.stop()

plan_ids = [plan["id"] for plan in plans]

if (
    "active_plan_id" not in st.session_state
    or st.session_state.active_plan_id not in plan_ids
):
    st.session_state.active_plan_id = plans[-1]["id"]

active_plan = next(
    plan for plan in plans
    if plan["id"] == st.session_state.active_plan_id
)

if len(plans) > 1:
    plan_options = {
        f"{plan.get('title', 'Untitled Plan')} — {plan.get('start_date', '')}":
        plan["id"]
        for plan in plans
    }

    labels = list(plan_options.keys())

    selected_label = next(
        (
            label for label, plan_id in plan_options.items()
            if plan_id == active_plan["id"]
        ),
        labels[-1],
    )

    selected_label = st.selectbox(
        "Choose a plan",
        options=labels,
        index=labels.index(selected_label),
    )

    st.session_state.active_plan_id = plan_options[selected_label]

    active_plan = next(
        plan for plan in st.session_state.plans
        if plan["id"] == st.session_state.active_plan_id
    )

# -------------------- TABS --------------------

dashboard_tab, plan_tab, automation_tab, export_tab = st.tabs([
    "📊 Dashboard",
    "📝 My Plan",
    "⚙️ Workflow Automation",
    "📥 Export PDF",
])

# -------------------- DASHBOARD --------------------

with dashboard_tab:
    st.subheader("Your Daily Dashboard")
    st.write(f"**Current goal:** {active_plan.get('goal', '')}")

    total, completed, pending = get_task_counts(active_plan)
    progress = round(completed / total * 100) if total else 0

    today = date.today()
    overdue = 0
    today_tasks = []

    for day_index, day_data in enumerate(active_plan.get("days", [])):
        scheduled_date = get_task_date(active_plan, day_index)

        for task in day_data.get("tasks", []):
            if not task.get("completed", False) and scheduled_date < today:
                overdue += 1

            if scheduled_date == today:
                today_tasks.append(task)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tasks", total)
    col2.metric("Completed", completed)
    col3.metric("Pending", pending)
    col4.metric("Overdue", overdue)

    st.write("### Overall Progress")
    st.progress(progress / 100)
    st.write(f"**{progress}% complete**")

    st.divider()
    st.write("### Today's Tasks")

    if today_tasks:
        for task in today_tasks:
            status = "✅" if task.get("completed", False) else "⬜"
            st.write(
                f"{status} **{task.get('title', 'Untitled task')}** "
                f"— {task.get('duration', 0)} minutes"
            )
    else:
        st.info("No tasks are scheduled for today.")

    st.divider()
    st.write("### Plan Overview")

    for day_index, day_data in enumerate(active_plan.get("days", [])):
        day_date = get_task_date(active_plan, day_index)
        day_total = len(day_data.get("tasks", []))
        day_completed = sum(
            1 for task in day_data.get("tasks", [])
            if task.get("completed", False)
        )

        st.write(
            f"**Day {day_data.get('day', day_index + 1)} "
            f"({day_date.isoformat()}):** "
            f"{day_completed}/{day_total} tasks completed"
        )

# -------------------- MY PLAN --------------------

with plan_tab:
    st.subheader(active_plan.get("title", "My TaskPilot Plan"))
    st.write(f"**Goal:** {active_plan.get('goal', '')}")

    total, completed, pending = get_task_counts(active_plan)
    progress = round(completed / total * 100) if total else 0

    st.progress(progress / 100)
    st.write(f"{completed} of {total} tasks completed ({progress}%)")

    for day_index, day_data in enumerate(active_plan.get("days", [])):
        day_date = get_task_date(active_plan, day_index)

        with st.expander(
            f"Day {day_data.get('day', day_index + 1)} — {day_date.isoformat()}",
            expanded=(day_date == date.today()),
        ):
            if not day_data.get("tasks"):
                st.write("No tasks scheduled for this day.")
                continue

            for task_index, task in enumerate(day_data.get("tasks", [])):
                task_key = (
                    f"task_{active_plan['id']}_{day_index}_{task_index}"
                )

                if task_key not in st.session_state:
                    st.session_state[task_key] = task.get(
                        "completed", False
                    )

                st.checkbox(
                    f"{task.get('title', 'Untitled task')} "
                    f"({task.get('duration', 0)} min)",
                    key=task_key,
                )

                task["completed"] = st.session_state[task_key]

                if task.get("description"):
                    st.caption(task["description"])

    save_data()

    st.caption("Your task completion status is saved locally.")

# -------------------- WORKFLOW AUTOMATION --------------------

with automation_tab:
    st.subheader("Smart Workflow Automation")

    st.write(
        "TaskPilot can create a new 7-day schedule from the unfinished "
        "tasks in your current plan."
    )

    total, completed, pending = get_task_counts(active_plan)
    st.metric("Unfinished tasks to review", pending)

    if pending:
        if st.button(
            "🔄 Reschedule Unfinished Tasks",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "TaskPilot is reorganizing your unfinished tasks..."
                ):
                    new_plan = reschedule_plan(active_plan)

                if new_plan:
                    st.session_state.plans.append(new_plan)
                    st.session_state.active_plan_id = new_plan["id"]
                    save_data()
                    st.success("Your new 7-day plan is ready!")
                    st.rerun()

            except Exception as error:
                st.error(f"Could not reschedule tasks: {error}")
    else:
        st.success("All tasks in this plan are completed!")

    st.divider()
    st.write("### Previous Plan History")

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            old_total = item.get("total", 0)
            old_completed = item.get("completed", 0)
            old_progress = (
                round(old_completed / old_total * 100)
                if old_total else 0
            )

            st.write(
                f"**{item.get('title', 'Previous plan')}** — "
                f"{old_completed}/{old_total} tasks completed "
                f"({old_progress}%)"
            )
            st.caption(
                f"Archived on {item.get('archived_on', '')} | "
                f"Goal: {item.get('goal', '')}"
            )
    else:
        st.info("Your previous plan history will appear here.")

# -------------------- PDF EXPORT ONLY --------------------

with export_tab:
    st.subheader("Download Your Plan as PDF")

    total, completed, pending = get_task_counts(active_plan)
    progress = round(completed / total * 100) if total else 0

    st.write(f"**Plan:** {active_plan.get('title', '')}")
    st.write(f"**Goal:** {active_plan.get('goal', '')}")
    st.write(f"**Progress:** {completed}/{total} tasks completed ({progress}%)")

    try:
        pdf_data = create_pdf(active_plan)

        st.download_button(
            label="📄 Download Plan as PDF",
            data=pdf_data,
            file_name="taskpilot_plan.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as error:
        st.error(f"Could not create the PDF: {error}")

    st.caption(
        "Your PDF includes daily tasks, descriptions, durations, "
        "completion status, progress, and the review suggestion."
    )

# -------------------- FOOTER --------------------

st.divider()
st.caption(
    "TaskPilot | AI planning • Progress tracking • Workflow automation"
)
