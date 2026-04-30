from fastapi import FastAPI, Form, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from langchain.messages import HumanMessage

from timetable_generater import get_timetable
from assistant_model import rag_chain
from se_agent_model import se_agent

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



if __name__ == "__main__":
    import uvicorn
    # Use standard 8000 port
    uvicorn.run(app, host="0.0.0.0", port=8000)
