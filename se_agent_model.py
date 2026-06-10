import os
import base64
import json
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.messages import SystemMessage, ToolMessage

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import RetryPolicy

from prompt import SE_PROMPT
from database import get_mongodb_checkpointer, get_vectorstore_for_se_agent

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def software_knowledgebase(query: str):
    """Search the Software Engineering PDF database for theories and concepts."""

    docs = get_vectorstore_for_se_agent().similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])


@tool
def get_uml_viewer_link(mermaid_code: str) -> str:
    """Generates a robust, clickable link to view a Mermaid UML diagram."""

    state = {
        "code": mermaid_code,
        "mermaid": {"theme": "default"},
        "updateEditor": True,
        "autoSync": True,
        "updateDiagram": True
    }

    json_state = json.dumps(state)
    data = json_state.encode('utf-8')
    base64_encoded = base64.b64encode(data).decode('utf-8')
    
    long_url= f"https://mermaid.live/edit#base64:{base64_encoded}"

    try:
        api_url = f"http://tinyurl.com/api-create.php?url={long_url}"
        short_url = requests.get(api_url).text
        return short_url
    except:
        return long_url


def chatbot(state: State):
    input_messages = [SystemMessage(content=SE_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(input_messages)
    return {"messages": [response]}


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    
    return END


def se_agent():
    agent= StateGraph(MessagesState)

    agent.add_node("chatbot", chatbot, retry=RetryPolicy(max_attempts=3))
    agent.add_node("tool_node", tool_node)

    agent.add_edge(START, "chatbot")
    agent.add_edge("tool_node", "chatbot")
    agent.add_conditional_edges(
        "chatbot",
        should_continue,
        ["tool_node",END]
    )
    checkpointer = get_mongodb_checkpointer()
    agent = agent.compile(checkpointer=checkpointer)
    return agent


def show_agent(agent):
    try:
        png_bytes = agent.get_graph(xray=True).draw_mermaid_png()
        with open("agent_graph.png", "wb") as f:
            f.write(png_bytes)
            
        print("Success! Open 'agent_graph.png' in your folder to see the flow.")
    except Exception as e:
        print(f"Could not generate graph: {e}")

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

tools = [software_knowledgebase, get_uml_viewer_link]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = model.bind_tools(tools)
