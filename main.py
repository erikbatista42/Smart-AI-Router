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

load_dotenv()

# DEFINE TOOLS
@tool
def general_routing(input: str):
    """Use Tavily seat"""
    return "General routing tool"

@tool
def browser_search(input: str):
    """Use Tavily to search the browser for real-time web results"""
    print("used function broswer")
    return TavilySearchResults(max_results=2).invoke(input)

@tool
def code_work(input: str):
    """Use claude to write code from any related code context"""
    print("used function code work")
    return "code work tool"

@tool
def image_generation(input: str):
    """Use flux to generate images from any related image text generation context"""
    print("used flux image generation")
    return "flux image generation tool"


message_with_multiple_tool_calls = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "general_routing",
            "args": {"input":"hi there"},
            "id": "tool_call_id_1",
            "type": "tool_call",
        },
        {
            "name": "browser_search",
            "args": {"input": "search the web for the latest news about conor mcgregor"},
            "id": "tool_call_id_2",
            "type": "tool_call",
        },
    ],
)

tools = [browser_search, general_routing, code_work, image_generation]
tool_node = ToolNode(tools)
# result = tool_node.invoke({"messages": [message_with_multiple_tool_calls]})
model_with_tools = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0).bind_tools(tools)

# result = model_with_tools.invoke("what's the weather in sf?")

result = tool_node.invoke({"messages": [model_with_tools.invoke("Create an image of a cat.")]})
print(".")
print(".")
print(".")
print("--------------------------------")
print(result)
print("--------------------------------")


# EXECUTE BOTH OF THE TOOLS THAT WE HAVE DEFINED. IT EXECUTES IN ORDER.