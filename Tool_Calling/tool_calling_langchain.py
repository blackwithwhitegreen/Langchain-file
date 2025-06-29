from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
import requests


# tool Creation
@tool
def division(a: int, b:int) -> int:
    """Given 2 numbers a and b for Divsion"""
    return a/b

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

# llm tool binding/ llm connect with tool
llm_with_tools = llm.bind_tools([division])

# Tool Calling
query = HumanMessage("can you divide 4 by 2")
messages = [query]
result = llm_with_tools.invoke(messages)

messages.append(result)

# tool Execution
tool_result = division.invoke(result.tool_calls[0])
messages.append(tool_result)

# again llm imvoke or sending to again llm 
print(llm_with_tools.invoke(messages))


# print(final_result.content)

