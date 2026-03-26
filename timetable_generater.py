import re
import json
from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_llm, get_sessions_collection

llm = get_llm()
sessions = get_sessions_collection()



TIMETABLE_SYSTEM_PROMPT = """
You are the "Exam Timetable Architect" for slmgx.live and your nick name is Lizzy 🩵. 
Your goal is to help the user create a personal study schedule for exams.

STEPS:
1. You can ask 6 questions. If the user refuses to give enough details, try a maximum of 8 questions. 
2. Ask for: Exam start date, Subjects, and preferred study hours in each question.
3. First question: Ask for the exam timetable in a formatted way or ask to send a screenshot/file.
4. If you get the timetable, ask the next appropriate questions.
5. If the user fails to provide the timetable within 3 questions, the system will trigger an error (handled by server).
6. Ask about favorite and weak subjects to make it personal.
7. Once you have enough info, generate a complete month-long timetable.
8. If turn 8 is reached without all info, give a sample realistic timetable and send a funny message about the user being difficult.

OUTPUT RULE:
- Plain text for chatting.
- If providing the timetable, you MUST reply ONLY with a single JSON object.
- Use colorful hex codes (e.g., #5941a9, #10b981).

JSON STRUCTURE:
{
  "month": "Month Year",
  "exam_target": "Exam Name",
  "weeks": [
    {
      "days": [
        {
          "date": "Day Date",
          "sessions": [{"time": "Range", "subject": "Name", "color": "Hex", "note": "Topic"}]
        }
      ]
    }
  ]
}
"""



def get_timetable(user_id: str, message: str, file_bytes: bytes = None, content_type: str = None):
    session = sessions.find_one({"user_id": user_id})
    if not session:
        session = {
            "user_id": user_id,
            "turn_count": 0,
            "has_timetable": False,
            "is_blocked": False
        }
        sessions.insert_one(session)

    if session["is_blocked"]:
        return {"type": "ERROR", "content": "Session terminated: No timetable provided within limits."}

    new_count = session["turn_count"] + 1
    
    content_list = [{"type": "text", "text": message}]
    if file_bytes:
        encoded_image = base64.b64encode(file_bytes).decode("utf-8")
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{encoded_image}"}
        })
        # Mark as having timetable since they provided a document
        sessions.update_one({"user_id": user_id}, {"$set": {"has_timetable": True}})
        session["has_timetable"] = True

    # Check text for keywords if no file was sent
    elif any(word in message.lower() for word in ["monday", "subject", "schedule", "exam date", "timetable"]):
        sessions.update_one({"user_id": user_id}, {"$set": {"has_timetable": True}})
        session["has_timetable"] = True

    #  Block if no info provided by turn 4
    if not session["has_timetable"] and new_count > 3:
        sessions.update_one({"user_id": user_id}, {"$set": {"is_blocked": True}})
        return {"type": "ERROR", "content": "I can't proceed without your exam dates. Support terminated."}

    # Invoke the AI
    try:
        response = llm.invoke([
            SystemMessage(content=f"{TIMETABLE_SYSTEM_PROMPT}\nCURRENT TURN: {new_count} of 8."),
            HumanMessage(content=content_list)
        ])
        res_text = response.content
    except Exception as e:
        return {"type": "ERROR", "content": f"AI Error: {str(e)}"}
    
    #  Handle JSON vs TEXT responses
    if "{" in res_text and "}" in res_text:
        json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
        if json_match:
            try:
                # Session is finished, clean up DB memory
                sessions.delete_one({"user_id": user_id})
                return {"type": "DATA", "content": json.loads(json_match.group(0))}
            except json.JSONDecodeError:
                pass 

    sessions.update_one({"user_id": user_id}, {"$set": {"turn_count": new_count}})
    
    return {"type": "TEXT", "content": res_text, "turn": new_count}
