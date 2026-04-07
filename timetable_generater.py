import re
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import base64
from database import get_timetable_db
from prompt import TIMETABLE_SYSTEM_PROMPT
from utils import get_llm

llm = get_llm()
sessions = get_timetable_db()

def _build_history_messages(history: list) -> list:
    """Convert stored history dicts back into LangChain message objects."""
    messages = []
    for entry in history:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "ai":
            messages.append(AIMessage(content=entry["content"]))
    return messages

def get_timetable(user_id: str, message: str, file_bytes: bytes = None, content_type: str = None):
    session = sessions.find_one({"user_id": user_id})
    if not session:
        session = {
            "user_id": user_id,
            "turn_count": 0,
            "has_timetable": False,
            "is_blocked": False,
            "history": []        
        }
        sessions.insert_one(session)

    if session["is_blocked"]:
        return {"type": "ERROR", "content": "Session terminated: No timetable provided within limits."}

    new_count = session["turn_count"] + 1

    content_list = [{"type": "text", "text": message}]
    
    # We check for images, specific keywords, or common date patterns (e.g., "April 15" or "15/04")
    has_date = bool(re.search(r'(\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))|(\d{1,2}/\d{1,2})', message.lower()))
    keywords = ["monday", "subject", "schedule", "exam date", "timetable", "physics", "maths", "it"]
    
    if file_bytes:
        encoded_image = base64.b64encode(file_bytes).decode("utf-8")
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{encoded_image}"}
        })
        session["has_timetable"] = True
    elif any(word in message.lower() for word in keywords) or has_date:
        session["has_timetable"] = True

    if session["has_timetable"]:
        sessions.update_one({"user_id": user_id}, {"$set": {"has_timetable": True}})

    # Block if no timetable info provided by turn 4
    if not session["has_timetable"] and new_count > 3:
        sessions.update_one({"user_id": user_id}, {"$set": {"is_blocked": True}})
        return {"type": "ERROR", "content": "I can't proceed without your exam dates. Support terminated."}

    history_messages = _build_history_messages(session.get("history", []))

    # If we have the timetable, we explicitly tell the AI to stop asking and start generating.
    status_instruction = "DATA RECEIVED: Stop asking questions. Generate the JSON timetable now." if session["has_timetable"] else "DATA MISSING: Continue asking for the timetable."

    invoke_messages = (
        [SystemMessage(content=f"{TIMETABLE_SYSTEM_PROMPT}\n{status_instruction}\nCURRENT TURN: {new_count} of 8.")]
        + history_messages
        + [HumanMessage(content=content_list)]
    )

    try:
        response = llm.invoke(invoke_messages)
        res_text = response.content
    except Exception as e:
        return {"type": "ERROR", "content": f"AI Error: {str(e)}"}

    # Check for JSON in response
    if '"weeks"' in res_text and '"month"' in res_text:
        json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if json_match:
            try:
                sessions.delete_one({"user_id": user_id})  
                return {"type": "DATA", "content": json.loads(json_match.group(0))}
            except json.JSONDecodeError:
                pass
                
    new_history_entries = [
        {"role": "user", "content": message},
        {"role": "ai", "content": res_text},
    ]

    sessions.update_one(
        {"user_id": user_id},
        {
            "$set":  {"turn_count": new_count},
            "$push": {"history": {"$each": new_history_entries}},
        }
    )

    return {"type": "TEXT", "content": res_text, "turn": new_count}
