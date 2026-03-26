import os
import re
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# vector db
client = MongoClient(os.getenv("MONGODB_URI"))
db_name = "website_assistant"
collection_name = "code_vectors"
collection = client[db_name][collection_name]

# timetable memory
db = client["website_assistant"]
sessions_col = db["timetable_sessions"]

vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index" 
)

def get_retriver():
    return (vector_store.as_retriever(search_kwargs={"k": 5}))

def get_sessions_collection():
    return sessions_col


def extract_clean_json(ai_string):
    """Extracts JSON from markdown code blocks or raw strings."""
    try:
        clean = re.sub(r'```json|```', '', ai_string).strip()
        return clean
    except Exception:
        return ai_string
