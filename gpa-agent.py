import os
import re
from io import BytesIO
from typing import TypedDict, Optional, List, Dict, Any, Union

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ==========================================
# 1. HELPER / CONVERTER FUNCTIONS
# ==========================================

def normalize_column_name(col: str) -> str:
    """Normalize column names for consistent downstream processing."""
    return (
        str(col).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def excel_to_json(file_bytes: bytes, sheet_name: Optional[Union[str, int]] = 0) -> Dict[str, Any]:
    """Convert an Excel sheet to a JSON-safe dictionary."""
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, dtype=object)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df.where(pd.notna(df), None)
    rows: List[Dict[str, Any]] = df.to_dict(orient="records")

    return {
        "sheet": sheet_name,
        "columns": list(df.columns),
        "row_count": len(rows),
        "rows": rows
    }


def excel_to_markdown_table(file_bytes: bytes, sheet_name: Optional[Union[str, int]] = 0, max_rows: int = 50) -> str:
    """Convert sheet into markdown table preview for LLM prompt context."""
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, dtype=object)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df.where(pd.notna(df), None)
    return df.head(max_rows).to_markdown(index=False)


def extract_ousl_credits(course_code: str) -> float:
    """
    Extracts credit rating from an OUSL course code (e.g., 'ADU3218' -> 2.0 credits).
    In OUSL 7-character codes, the 2nd digit represents the credit value.
    """
    course_code = str(course_code).strip().upper()
    digits = re.findall(r'\d', course_code)
    
    if len(digits) >= 2:
        return float(digits[1])
    
    return 3.0  # Fallback default credit value


# ==========================================
# 2. STATE DEFINITION & PYDANTIC SCHEMAS
# ==========================================

class CourseItem(TypedDict):
    course_code: str
    course_name: str
    credits: float
    grade: str


class UniversityGPAState(TypedDict):
    # Inputs
    file_bytes: bytes                  # Raw Excel file
    custom_rules_prompt: str           # OUSL university grading rules
    degree_type: str                   # "90_credits" or "120_credits"
    
    # Processed Raw Data
    markdown_table: str
    json_data: Dict[str, Any]
    
    # HITL Payload
    extracted_courses: List[CourseItem] # Sent to frontend for user confirmation/edits
    user_confirmed: bool
    
    # Python Exact Math Results
    total_completed_credits: float
    target_credits: int
    remaining_credits: float
    calculated_gpa: float
    
    # Output
    final_analysis_report: str          # Motivational response & future credit advice
    error: Optional[str]


# Schema used for strict LLM extraction
class ExtractedCourseRaw(BaseModel):
    course_code: str = Field(description="Course code, e.g., ADU3218")
    course_name: str = Field(description="Title of the course")
    grade: str = Field(description="Letter grade obtained, e.g., A, B+, C")

class CourseListRawSchema(BaseModel):
    courses: List[ExtractedCourseRaw]

structured_extractor = llm.with_structured_output(CourseListRawSchema)


# ==========================================
# 3. LANGGRAPH NODES
# ==========================================

def convert_excel_node(state: UniversityGPAState) -> Dict[str, Any]:
    """Reads Excel bytes and produces structured Markdown & JSON."""
    try:
        file_bytes = state["file_bytes"]
        md_table = excel_to_markdown_table(file_bytes)
        json_payload = excel_to_json(file_bytes)
        return {"markdown_table": md_table, "json_data": json_payload, "error": None}
    except Exception as e:
        return {"error": f"Failed to parse Excel file: {str(e)}"}


def extract_courses_node(state: UniversityGPAState) -> Dict[str, Any]:
    """Extracts passed subjects and calculates credit count based on OUSL code structure."""
    if state.get("error"):
        return {"extracted_courses": []}

    system_prompt = f"""
    You are an exact data extraction assistant for OUSL Sri Lanka result sheets.
    
    ### RULES:
    1. Select ONLY subjects where Progress Status is 'Pass'.
    2. Exclude subjects with Progress Status 'NOT Eligible', 'RX', or 'Pending'.
    3. Exclude any course code where the 3rd letter is 'E' (e.g., CYE3200, CSE3214, LEE3410).
    4. Capture course_code, course_name, and grade accurately.
    
    Custom Rules Prompt:
    {state.get('custom_rules_prompt', '')}
    """

    human_prompt = f"Student Result Sheet:\n{state['markdown_table']}"
    
    response: CourseListRawSchema = structured_extractor.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    extracted_courses = []
    for c in response.courses:
        credit_val = extract_ousl_credits(c.course_code)
        extracted_courses.append({
            "course_code": c.course_code,
            "course_name": c.course_name,
            "credits": credit_val,
            "grade": c.grade
        })

    return {"extracted_courses": extracted_courses}


def human_review_node(state: UniversityGPAState) -> Dict[str, Any]:
    """
    Pauses graph execution to send extracted courses and degree choice to the frontend UI.
    Resumes when the user posts back confirmed/edited items.
    """
    user_response = interrupt({
        "message": "Please review and confirm your extracted courses and target degree.",
        "extracted_courses": state["extracted_courses"],
        "degree_type": state.get("degree_type", "90_credits")
    })
    
    return {
        "extracted_courses": user_response["confirmed_courses"],
        "degree_type": user_response["degree_type"],
        "user_confirmed": True
    }


def calculate_gpa_node(state: UniversityGPAState) -> Dict[str, Any]:
    """Deterministic GPA math engine in pure Python."""
    gpv_map = {
        "A+": 4.00, "A": 4.00, "A-": 3.70,
        "B+": 3.30, "B": 3.00, "B-": 2.70,
        "C+": 2.30, "C": 2.00, "C-": 1.70,
        "D+": 1.30, "D": 1.00, "E": 0.00, "F": 0.00
    }

    target_credits = 120 if state.get("degree_type") == "120_credits" else 90
    total_weighted_points = 0.0
    total_completed_credits = 0.0

    for course in state["extracted_courses"]:
        cr = float(course.get("credits", 0))
        grade = str(course.get("grade", "")).strip().upper()
        gpv = gpv_map.get(grade, 0.0)
        
        total_weighted_points += (cr * gpv)
        total_completed_credits += cr

    gpa = total_weighted_points / total_completed_credits if total_completed_credits > 0 else 0.0
    remaining_credits = max(0.0, float(target_credits) - total_completed_credits)

    return {
        "calculated_gpa": round(gpa, 2),
        "total_completed_credits": total_completed_credits,
        "target_credits": target_credits,
        "remaining_credits": remaining_credits
    }


def generate_ai_analysis_node(state: UniversityGPAState) -> Dict[str, Any]:
    """Generates an inspiring, motivational academic analysis report."""
    is_completed = state["remaining_credits"] <= 0
    status_str = "COMPLETED" if is_completed else "IN PROGRESS"
    
    system_instruction = f"""
    You are an encouraging academic advisor at Open University of Sri Lanka (OUSL).
    Analyze the student's academic standing and write an inspiring, highly motivational report.
    
    ### ACADEMIC PROFILE:
    - Degree Track: {state['target_credits']} Credits Degree
    - Current GPA: {state['calculated_gpa']} / 4.00
    - Completed Credits: {state['total_completed_credits']} / {state['target_credits']}
    - Remaining Credits Needed: {state['remaining_credits']}
    - Graduation Status: {status_str}
    
    ### INSTRUCTIONS:
    1. Start with an inspiring greeting acknowledging their effort.
    2. Provide an honest analysis of their current GPA ({state['calculated_gpa']}).
    3. If status is 'IN PROGRESS':
       - Explicitly mention that they need {state['remaining_credits']} more credits to complete their {state['target_credits']}-credit degree target.
       - Give tactical advice on target grades needed in remaining modules to protect or improve their GPA.
    4. If status is 'COMPLETED':
       - Congratulate them on reaching their degree target!
    """

    response = llm.invoke([SystemMessage(content=system_instruction)])
    return {"final_analysis_report": response.content}


# ==========================================
# 4. GRAPH BUILD & COMPILATION
# ==========================================

workflow = StateGraph(UniversityGPAState)

# Add Nodes
workflow.add_node("convert_excel", convert_excel_node)
workflow.add_node("extract_courses", extract_courses_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("calculate_gpa", calculate_gpa_node)
workflow.add_node("generate_ai_analysis", generate_ai_analysis_node)

# Add Edges
workflow.add_edge(START, "convert_excel")
workflow.add_edge("convert_excel", "extract_courses")
workflow.add_edge("extract_courses", "human_review")
workflow.add_edge("human_review", "calculate_gpa")
workflow.add_edge("calculate_gpa", "generate_ai_analysis")
workflow.add_edge("generate_ai_analysis", END)

# Checkpointer required for state interrupts
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ==========================================
# 5. EXECUTION EXAMPLE (SIMULATED FLOW)
# ==========================================

if __name__ == "__main__":
    # Create sample Excel data in-memory matching your image sample
    sample_data = {
        "Course Code": ["ADU3218", "CPU3101", "CYE3200", "ADU3219"],
        "Course Name": ["Basic Statistics", "Programming in C", "Continuing Ed Sub", "Applied Maths"],
        "Last Offered Year": [2024, 2024, 2024, 2024],
        "Progress Status": ["Pass", "Pass", "Pass", "NOT Eligible"],
        "Grade": ["A", "B+", "A", "F"],
        "Attempts": [1, 1, 1, 2],
        "Eligibility Completed Years": [2, 2, 2, 2],
        "Eligibility Left": ["-", "-", "-", "-"]
    }
    
    excel_buffer = BytesIO()
    pd.DataFrame(sample_data).to_excel(excel_buffer, index=False)
    excel_bytes_data = excel_buffer.getvalue()

    # Unique session configuration
    thread_config = {"configurable": {"thread_id": "ousl_student_session_1"}}

    print("--- PHASE 1: STARTING GRAPH & PARSING EXCEL ---")
    initial_input = {
        "file_bytes": excel_bytes_data,
        "degree_type": "90_credits",
        "custom_rules_prompt": "OUSL Sri Lanka GPA scale"
    }

    # Run until interrupted at human_review
    for event in app.stream(initial_input, config=thread_config):
        print(f"Executed step: {list(event.keys())}")

    # Retrieve current paused state to send data to Frontend UI
    current_state = app.get_state(thread_config)
    
    print("\n--- FRONTEND UI PAYLOAD RECEIVED ---")
    print("Extracted Courses for Table UI:")
    print(current_state.values["extracted_courses"])

    # --- PHASE 2: SIMULATING FRONTEND USER ACTION ---
    print("\n--- PHASE 2: USER CONFIRMS DATA & SELECTS 120-CREDIT DEGREE ---")
    
    user_confirmed_payload = {
        "confirmed_courses": current_state.values["extracted_courses"], # Or user edited list
        "degree_type": "120_credits"                                   # User changed degree track to 120
    }

    # Resume graph execution passing user choices
    for event in app.stream(Command(resume=user_confirmed_payload), config=thread_config):
        print(f"Executed step: {list(event.keys())}")

    # Fetch final graph output
    final_state = app.get_state(thread_config).values

    print("\n==========================================")
    print("           CALCULATED RESULTS             ")
    print("==========================================")
    print(f"Target Degree Track : {final_state['target_credits']} Credits")
    print(f"Completed Credits   : {final_state['total_completed_credits']}")
    print(f"Remaining Credits   : {final_state['remaining_credits']}")
    print(f"Calculated GPA      : {final_state['calculated_gpa']} / 4.00")
    print("\n==========================================")
    print("         AI ADVISORY & MOTIVATION         ")
    print("==========================================")
    print(final_state["final_analysis_report"])
