from langchain_community.tools import BaseTool
from langchain.tools import tool

# Custom tools 
@tool
def add(a:int, b:int) -> int:
    """Add two numbers"""
    return a+b
@tool
def multiply(a:int, b: int) -> int:
    """Multiply two numbers"""
    return a*b

# this is the main class
# this class help for usebility of multiple tools.
class MathToolkit:
    def get_tools(self):
        return [add,multiply]
    

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "->",tool.description)
