from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import requests
import os
import time
from openai import OpenAI


load_dotenv()

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


result = tool_node.invoke({"messages": [model_with_tools.invoke("Make an image of a high dosage looking pre-workout pill with a red and black label that says 'X-AI' on it")]})
print(".")
print(".")
print(".")
print("--------------------------------")
print(result)
print("--------------------------------")



from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.runnables.config import RunnableConfig
import operator
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.graph import END, StateGraph, START

model = ChatAnthropic(model_name="claude-2.1")
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def _call_model(state):
    state["messages"]
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()


openai_model = ChatOpenAI()

models = {
    "anthropic": "claude-3-haiku-20240307",
    "openai": openai_model,
}


def _call_model(state: AgentState, config: RunnableConfig):
    # Access the config through the configurable key
    model_name = config["configurable"].get("model", "anthropic")
    model = models[model_name]
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(AgentState)
builder.add_node("model", _call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()