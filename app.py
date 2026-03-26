from fastapi import FastAPI, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from timetable_generater import get_timetable
from assistant_model import rag_chain

origins = [
    "https://slmgx.live",
    "https://www.slmgx.live",
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
async def timetable_chat(message: str = Form(...), user_id: str = Form(...)):
    try:
        return get_timetable(user_id, message)
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    # Use standard 8000 port
    uvicorn.run(app, host="0.0.0.0", port=8000)
