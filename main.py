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


def get_image_result(image_id, api_key):
    headers = {
        'X-Key': api_key
    }
    
    url = f"https://api.bfl.ml/v1/get_result?id={image_id}"
    
    # The image might take some time to generate, so we need to poll until it's ready
    while True:
        response = requests.get(url, headers=headers)
        result = response.json()
        
        if result['status'] != "Task not found":
            return result
            
        time.sleep(1)  # Wait a second before checking again

# Example usage:
# First get the image ID from your generation request
# Then:


@tool
def image_generation(input: str):
    """Use flux to generate images from any related image text generation context"""
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