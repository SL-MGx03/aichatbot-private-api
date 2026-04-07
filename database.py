import os
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))

def get_mongodb_checkpointer():
    """
    Sets up a MongoDB checkpointer with a built-in 24-hour 
    auto-delete using the native library parameter.
    """
    db_name = "se_agent_prod"
    checkpoint_coll = "checkpoints"
    
    saver = MongoDBSaver(
        client, 
        db_name=db_name,
        checkpoint_collection_name=checkpoint_coll,
        ttl=86400  # 24 hours in seconds
    )
    
    print("✅ MongoDB Memory Initialized (Native 24h TTL Active)")
    return saver

def get_collection_for_se_agent():
    db_name = "se_agents"
    collection_name = "se_agents"
    collection = client[db_name][collection_name]
    return collection


def get_vectorstore_for_se_agent():
    embeddings =GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=3072)
    vector_store = MongoDBAtlasVectorSearch(
        collection= get_collection_for_se_agent(),
        embedding=embeddings,
        index_name= "vector_index"
        )
    return vector_store



def get_vectorstore_for_web_assistant():

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", output_dimensionality=768)
    db_name = "website_assistant"
    collection_name = "code_vectors"
    collection = client[db_name][collection_name]


    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index" 
    )

    return vector_store


def get_timetable_db():
    
    db = client["website_assistant"]
    sessions_col = db["timetable_sessions"]
    return sessions_col
