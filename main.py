from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt import ToolNode
from typing import Literal, Optional, Annotated, Sequence
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END, StateGraph, START
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import requests
import os
import time
from openai import OpenAI
import operator
from typing_extensions import TypedDict




from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage

load_dotenv()

# Initiate messages state/list
state = MessagesState()

# DEFINE TOOLS

@tool
def general_routing(input: str):
    """Use this tool for:
    - General knowledge questions
    - Simple explanations
    - Basic facts and concepts
    - Non-time-sensitive information
    Do NOT use for: current events, coding, or generating media
    """
    # CHATGPT
    print("used General routing tool")
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

    completion = client.chat.completions.create(
        model="grok-vision-beta",
        messages=[
            {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."},
            {"role": "user", "content": f"{input}"},
        ],
    )
    print(".")
    print(".")
    print(".")
    print("--------------------------------")
    print(completion.choices[0].message.content)
    return "General routing tool"


@tool
def browser_search(input: str):
    """Use this tool when:
    - Real-time information is needed
    - Current events or news are requested
    - Specific facts need verification
    - User explicitly asks to search the web
    Do NOT use for: general knowledge or simple questions
    """
    print("used function broswer")
    print(TavilySearchResults(max_results=2).invoke(input))
    return TavilySearchResults(max_results=2).invoke(input)

@tool
def code_work(input: str):
    """Use this tool when:
    - Writing or debugging code
    - Explaining programming concepts
    - Providing code examples
    - Answering technical programming questions
    Do NOT use for: non-programming tasks
    """
    print("USED FUNCTION: CODE WORK -- ChatGPT-omini")
    return "code work tool"


@tool
def image_generation(input: str):
    """Use this tool when:
    - User requests to create or generate an image
    - Visual representation is needed
    - Art or design creation is requested
    Keywords: create, generate, draw, design, picture, image, make
    Do NOT use for: searching existing images or non-image tasks
    """
    print("used FLUX image generation")
    url = "https://api.bfl.ml/v1/flux-pro-1.1"
    image_url = "https://api.bfl.ml/v1/get_result?id="
    # prompt_given = MessagesState["messages"][-2].content
    # print("PROMPT GIVEN----------:", prompt_given)
    payload = {
        "prompt": f"{input}",
        "width": 1024,
        "height": 768,
        "prompt_upsampling": True,
        "seed": 42,
        "safety_tolerance": 2,
        "output_format": "jpeg"
    }

    BFL_API_KEY = os.getenv('BFL_API_KEY')
    headers = {
        "Content-Type": "application/json",
        "X-Key": f"{BFL_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)
    image_id = response.json()["id"]
    image_url = image_url + image_id
    print("IMAGE URL:", image_url)

    return image_url

tools = [general_routing, browser_search, code_work, image_generation]
tool_node = ToolNode(tools)

model_with_tools = ChatAnthropic(model="claude-3-haiku-20240307", temperature=1, api_key=os.getenv("ANTHROPIC_API_KEY")).bind_tools(tools)


# result = tool_node.invoke({"messages": [model_with_tools.invoke("Make an image of a high dosage looking pre-workout pill with a red and black label that says 'X-AI' on it")]})
print(".")
print(".")
print(".")
print("--------------------------------")
# print(result)
print("--------------------------------")


# Creating ReAct agent - takes a query as an input, then repeadtely calls tools until it has enough information to answer the query.

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]
    # The system message should be the first message in the list
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [
            SystemMessage(content="""You are a helpful AI assistant with access to various tools:
    - Use general_routing for basic knowledge and explanations
    - Use browser_search for current events and real-time info
    - Use code_work for programming tasks
    - Use image_generation for creating new images
    
    Choose tools carefully based on the user's needs.""")
        ] + messages
    
    response = model_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# ------------------------------
@tool
def router_tool():
    """Route to the appropriate tool based on the query type.
    ALWAYS use this tool first to determine which specialized tool to use next.
    """
    print("Using router tool to determine next action")
    
    # Use Claude to determine the best tool
    router_model = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    last_message = state["messages"][-1].content
    # # messages = state["messages"]
    # last_message = messages[-1]
    # print("LAST MESSAGE: ", last_message)
    prompt = f"""
    Your job is to analyze this query and tell me which tool you think is most appropriate to use next based on the following query: {last_message}.

    Tool names:
    "general_routing" - Used for general knowledge, simple explanations, basic facts
    "browser_search" - Used for current events, real-time info, web searches
    "image_generation" - Used for creating/generating images
    "code_work" - For anything related to writing code, debugging code, etc.

    Only return the name of the tool and NOTHING ELSE.

    FOR EXAMPLE, if the query is about the latest updates of Conor McGregor, you should return "browser_search".

    As another example, if the query is about generating any kind of images, you should return "image_generation".
    
    """
    
    response = router_model.invoke(prompt)
    tool_name = response.content.strip().lower()
    print(f"Router selected: {tool_name}")
    return {"messages": [AIMessage(content=tool_name)]}



# GRAPH DEFINITION
workflow = StateGraph(MessagesState)

# NODES
workflow.add_node("router", router_tool)

# EDGES
workflow.add_edge(START, "router")
workflow.add_edge("router", END)

app = workflow.compile()

# For testing locally (won't affect LangStudio)
if __name__ == "__main__":
    first_message = HumanMessage(content="Generate an image of a mysterious woman with full dark eyes.")
    state["messages"] = [first_message]
    result = app.invoke(state)
    print(result)
    
# left off at trying to get the image generation tool to work with the state messages.