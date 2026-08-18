import uuid
import logging
from typing import Optional, List
from fastapi import FastAPI, Form, APIRouter, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.messages import HumanMessage

from timetable_generater import get_timetable
from assistant_model import rag_chain
from se_agent_model import se_agent
from gpa_agent import app as gpa_agent_graph
from langgraph.types import Command

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fastapi_app")

origins = [
    "https://slmgx.live",
    "https://www.slmgx.live",
    "https://slmgx.edu.lk",
    "https://www.slmgx.edu.lk",
    "http://localhost:3000", # local testing
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatQuery(BaseModel):
    message: str


class SEAgentQuery(BaseModel):
    message: str
    thread_id: str = "default_session" # ID for the 24h MongoDB memory


class CourseItemSchema(BaseModel):
    course_code: str
    course_name: str
    credits: float
    grade: str


class GPAConfirmQuery(BaseModel):
    thread_id: str
    degree_type: str
    confirmed_courses: List[CourseItemSchema]


@app.post("/chat")
async def chat(query: ChatQuery):
    try:
        # Debugging
        print(f"--- Incoming Message: {query.message} ---")
        response = rag_chain.invoke(query.message)
        
        print(f"--- AI Response: {response} ---")
        return {"answer": response}
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}") 
        return {"error": str(e)}


@app.post("/timetable-chat")
async def timetable_chat(
    message: str = Form(...), 
    user_id: str = Form(...),
    file: Optional[UploadFile] = File(None) 
):
    try:
        file_bytes = None
        content_type = None
        
        if file:
            file_bytes = await file.read()
            content_type = file.content_type
            
        return get_timetable(user_id, message, file_bytes, content_type)
    except Exception as e:
        print(f"Error in timetable-chat: {e}")
        return {"error": str(e)}


@app.post("/se_agent")
async def se_agent_chat(query: SEAgentQuery):

    agent = se_agent()
    try:
        config = {"configurable": {"thread_id": query.thread_id}}
        input_state = {"messages": [HumanMessage(content=query.message)]}
        result = agent.invoke(input_state, config)
        final_answer = result["messages"][-1].content
        
        return {
            "answer": final_answer,
            "thread_id": query.thread_id
        }
        
    except Exception as e:
        print(f"SE AGENT ERROR: {str(e)}")
        return {"error": str(e)}


@app.post("/gpa/upload")
async def upload_gpa_sheet(
    file: UploadFile = File(...),
    degree_type: str = Form("90_credits"),
    custom_rules: Optional[str] = Form("OUSL Sri Lanka GPA Scale")
):
    """
    Phase 1: Parses uploaded Excel/HTML result sheet, extracts passed courses, and pauses execution.
    """
    try:
        logger.info(f"[/gpa/upload] File received: {file.filename}, Degree: {degree_type}")
        file_bytes = await file.read()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_input = {
            "file_bytes": file_bytes,
            "degree_type": degree_type,
            "custom_rules_prompt": custom_rules
        }

        # Stream graph until interrupt
        for event in gpa_agent_graph.stream(initial_input, config=config):
            logger.info(f"[/gpa/upload Stream Step] Executed: {list(event.keys())}")

        current_state = gpa_agent_graph.get_state(config)
        
        if not current_state.values:
            logger.error("[/gpa/upload] State values empty after stream execution.")
            raise HTTPException(status_code=500, detail="GPA workflow failed to initialize.")

        extracted_courses = current_state.values.get("extracted_courses", [])
        state_error = current_state.values.get("error")

        logger.info(f"[/gpa/upload] Returning {len(extracted_courses)} extracted courses to client.")
        if state_error:
            logger.error(f"[/gpa/upload] State contains error message: {state_error}")

        return {
            "status": "AWAITING_HUMAN_CONFIRMATION",
            "thread_id": thread_id,
            "degree_type": degree_type,
            "extracted_courses": extracted_courses,
            "error": state_error
        }

    except Exception as e:
        logger.error(f"[/gpa/upload Exception]: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.post("/gpa/confirm")
async def confirm_gpa_calculation(query: GPAConfirmQuery):
    """
    Phase 2: Receives confirmed course list and degree choice. Resumes graph.
    """
    try:
        logger.info(f"[/gpa/confirm] Resuming thread_id: {query.thread_id}")
        config = {"configurable": {"thread_id": query.thread_id}}
        
        current_state = gpa_agent_graph.get_state(config)
        if not current_state.values:
            raise HTTPException(status_code=404, detail="Session expired or invalid thread_id.")

        user_confirmed_payload = {
            "confirmed_courses": [c.model_dump() for c in query.confirmed_courses],
            "degree_type": query.degree_type
        }

        for event in gpa_agent_graph.stream(Command(resume=user_confirmed_payload), config=config):
            logger.info(f"[/gpa/confirm Stream Step] Executed: {list(event.keys())}")

        final_state = gpa_agent_graph.get_state(config).values

        return {
            "status": "COMPLETED",
            "thread_id": query.thread_id,
            "target_credits": final_state.get("target_credits"),
            "total_completed_credits": final_state.get("total_completed_credits"),
            "remaining_credits": final_state.get("remaining_credits"),
            "calculated_gpa": final_state.get("calculated_gpa"),
            "analysis_report": final_state.get("final_analysis_report")
        }

    except Exception as e:
        logger.error(f"[/gpa/confirm Exception]: {str(e)}", exc_info=True)
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    # Use standard 8000 port
    uvicorn.run(app, host="0.0.0.0", port=8000)
