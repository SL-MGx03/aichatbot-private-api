from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from prompt import ASISTANT_PROMPT
from utils import get_retriever, get_llm


llm =get_llm()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", ASISTANT_PROMPT),
    ("human", "{question}"),
])


def format_docs(docs):
    formatted = []
    for doc in docs:
        path = doc.metadata.get("path", "unknown file")
        content = f"--- FILE: {path} ---\n{doc.page_content}"
        formatted.append(content)
    return "\n\n".join(formatted)

retriever = get_retriever()

# --- THE RAG CHAIN ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
