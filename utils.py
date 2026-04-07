import re
from database import get_vectorstore_for_web_assistant
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def get_retriever():
    vector_store= get_vectorstore_for_web_assistant()
    return (vector_store.as_retriever(search_kwargs={"k": 5}))


def extract_clean_json(ai_string):
    """Extracts JSON from markdown code blocks or raw strings."""
    try:
        clean = re.sub(r'```json|```', '', ai_string).strip()
        return clean
    except Exception:
        return ai_string
