from langchain_community.tools import tool

def division(a,b):
    """Divison of a / b"""
    return a/b


# Definin type hint 
# that means as shown a is int type and b is int type and ruturn int.
def division(a:int, b:int) -> int:
    """Division of a / b"""
    return a/b

# Add tool decorator
# @tool decorator makes it special function and llm can communicate to it.
@tool
def division(a:int, b:int) -> int:
    """Division of a /b"""
    return a/b


result = division({"a":2, "b":3})

print(result)
print(division.name)
print(division.description)
print(division.args)