import os
import re
import logging
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

from api_pool import DynamicChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpa_agent")

# Initialize LLM
llm = DynamicChatGroq(model="llama-3.3-70b-versatile", temperature=0)


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


def load_excel_dataframe(file_bytes: bytes, sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
    """
    Robustly loads Excel files. Handles .xlsx, legacy .xls, 
    and HTML tables disguised as .xls files from web portals.
    """
    # 1. Try standard openpyxl (.xlsx)
    try:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, dtype=object)
        logger.info("[Excel Loader] Successfully read via openpyxl (.xlsx)")
        return df
    except Exception as e:
        logger.warning(f"[Excel Loader] openpyxl failed: {e}")

    # 2. Try xlrd for legacy binary .xls
    try:
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, engine="xlrd", dtype=object)
        logger.info("[Excel Loader] Successfully read via xlrd (.xls)")
        return df
    except Exception as e:
        logger.warning(f"[Excel Loader] xlrd failed: {e}")

    # 3. Try reading as HTML table (common for university portal .xls exports)
    try:
        dfs = pd.read_html(BytesIO(file_bytes))
        if dfs:
            logger.info("[Excel Loader] Successfully read as HTML table export")
            return dfs[0]
    except Exception as e:
        logger.warning(f"[Excel Loader] read_html failed: {e}")

    raise ValueError("Could not parse file. Ensure it is a valid Excel file (.xlsx, .xls) or result export.")


def excel_to_json(file_bytes: bytes, sheet_name: Optional[Union[str, int]] = 0) -> Dict[str, Any]:
    """Convert an Excel sheet to a JSON-safe dictionary."""
    df = load_excel_dataframe(file_bytes, sheet_name=sheet_name)
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
    df = load_excel_dataframe(file_bytes, sheet_name=sheet_name)
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
    file_bytes: bytes                  
    custom_rules_prompt: str           
    degree_type: str                   
    
    markdown_table: str
    json_data: Dict[str, Any]
    
    extracted_courses: List[CourseItem] 
    user_confirmed: bool
    
    total_completed_credits: float
    target_credits: int
    remaining_credits: float
    calculated_gpa: float
    
    final_analysis_report: str          
    error: Optional[str]


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
    logger.info("--- NODE: convert_excel_node STARTED ---")
    try:
        file_bytes = state["file_bytes"]
        md_table = excel_to_markdown_table(file_bytes)
        json_payload = excel_to_json(file_bytes)
        
        logger.info(f"[convert_excel_node] Successfully parsed file. Row count: {json_payload.get('row_count')}")
        logger.info(f"[convert_excel_node] Columns found: {json_payload.get('columns')}")
        
        return {"markdown_table": md_table, "json_data": json_payload, "error": None}
    except Exception as e:
        logger.error(f"[convert_excel_node] FAILED: {str(e)}", exc_info=True)
        return {"error": f"Failed to parse result sheet: {str(e)}"}


def extract_courses_node(state: UniversityGPAState) -> Dict[str, Any]:
    """Extracts passed subjects and calculates credit count based on OUSL code structure."""
    logger.info("--- NODE: extract_courses_node STARTED ---")
    
    if state.get("error"):
        logger.error(f"[extract_courses_node] Skipped due to previous error: {state.get('error')}")
        return {"extracted_courses": []}

    system_prompt = f"""
    You are an exact data extraction assistant for OUSL Sri Lanka result sheets.
    
    ### RULES:
    1. Select ONLY subjects where Progress Status is 'Pass'.
    2. Exclude subjects with Progress Status 'NOT Eligible', 'RX', or 'Pending'.
    3. Exclude any course code where the 3rd letter is 'E' (e.g., CYE3200, CSE3214, LTE3406, FDE3021 ).
    4. Capture course_code, course_name, and grade accurately.
    
    Custom Rules Prompt:
    {state.get('custom_rules_prompt', '')}
    """

    human_prompt = f"Student Result Sheet:\n{state['markdown_table']}"
    logger.info(f"[extract_courses_node] Input Markdown snippet:\n{state['markdown_table'][:300]}...")

    try:
        response: CourseListRawSchema = structured_extractor.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        logger.info(f"[extract_courses_node] LLM Raw Extraction Count: {len(response.courses)}")

        extracted_courses = []
        for c in response.courses:
            credit_val = extract_ousl_credits(c.course_code)
            course_obj = {
                "course_code": c.course_code,
                "course_name": c.course_name,
                "credits": credit_val,
                "grade": c.grade
            }
            extracted_courses.append(course_obj)
            logger.info(f"  -> Extracted: {course_obj}")

        return {"extracted_courses": extracted_courses}
    except Exception as e:
        logger.error(f"[extract_courses_node] LLM Extraction Failed: {str(e)}", exc_info=True)
        return {"extracted_courses": [], "error": f"Extraction failed: {str(e)}"}


def human_review_node(state: UniversityGPAState) -> Dict[str, Any]:
    logger.info("--- NODE: human_review_node INTERRUPT TRIGGERED ---")
    user_response = interrupt({
        "message": "Please review and confirm your extracted courses and target degree.",
        "extracted_courses": state["extracted_courses"],
        "degree_type": state.get("degree_type", "90_credits")
    })
    
    logger.info(f"--- NODE: human_review_node RESUMED with payload: {user_response} ---")
    return {
        "extracted_courses": user_response["confirmed_courses"],
        "degree_type": user_response["degree_type"],
        "user_confirmed": True
    }


def calculate_gpa_node(state: UniversityGPAState) -> Dict[str, Any]:
    logger.info("--- NODE: calculate_gpa_node STARTED ---")
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

    logger.info(f"[calculate_gpa_node] GPA Calculated: {round(gpa, 2)} | Completed Credits: {total_completed_credits}")

    return {
        "calculated_gpa": round(gpa, 2),
        "total_completed_credits": total_completed_credits,
        "target_credits": target_credits,
        "remaining_credits": remaining_credits
    }


def generate_ai_analysis_node(state: UniversityGPAState) -> Dict[str, Any]:
    logger.info("--- NODE: generate_ai_analysis_node STARTED ---")
    is_completed = state["remaining_credits"] <= 0
    status_str = "COMPLETED" if is_completed else "IN PROGRESS"
    
    system_instruction = f"""
    You are an AI academic assistant tool designed to help OUSL students calculate their GPA and plan their course credits.
    Provide a friendly, supportive, and clear analysis of the user's progress.
    
    ### IMPORTANT PERSONA RULES:
    1. Do NOT pretend to be official university staff, faculty, or an administrator.
    2. Speak clearly as an AI planning tool assisting the student.
    3. Keep the tone warm, friendly, encouraging, and clear.
    
    ### ACADEMIC PROFILE:
    - Target Degree Track: {state['target_credits']} Credits
    - Calculated GPA: {state['calculated_gpa']} / 4.00
    - Completed Credits: {state['total_completed_credits']} / {state['target_credits']}
    - Remaining Credits Needed: {state['remaining_credits']}
    - Degree Status: {status_str}
    
    ### RESPONSE STRUCTURE:
    1. A friendly summary of their current GPA ({state['calculated_gpa']}).
    2. If degree is 'IN PROGRESS':
       - Mention that they need {state['remaining_credits']} more credits to reach their {state['target_credits']}-credit target.
       - Provide helpful target grade recommendations for upcoming modules to maintain or improve their GPA.
    3. If degree is 'COMPLETED':
       - Celebrate their achievement in reaching their full credit goal!
    """

    response = llm.invoke([SystemMessage(content=system_instruction)])
    return {"final_analysis_report": response.content}


# ==========================================
# 4. GRAPH BUILD & COMPILATION
# ==========================================

workflow = StateGraph(UniversityGPAState)

workflow.add_node("convert_excel", convert_excel_node)
workflow.add_node("extract_courses", extract_courses_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("calculate_gpa", calculate_gpa_node)
workflow.add_node("generate_ai_analysis", generate_ai_analysis_node)

workflow.add_edge(START, "convert_excel")
workflow.add_edge("convert_excel", "extract_courses")
workflow.add_edge("extract_courses", "human_review")
workflow.add_edge("human_review", "calculate_gpa")
workflow.add_edge("calculate_gpa", "generate_ai_analysis")
workflow.add_edge("generate_ai_analysis", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
