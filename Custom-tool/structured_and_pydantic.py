from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class DivisionInput(BaseModel):
    a : int = Field(required=True, description="The first number to add")
    b : int = Field(required=True, description="The second number to add")


def division_func(a: int, b:int) -> int:
    return a/b

division_tool = StructuredTool.from_function(
    func=division_func,
    name = "division",
    description="Division two number",
    args_schema=DivisionInput
)

result = division_tool.invoke({"a":3,"b":3})

print(result)
print(division_tool.name)
print(division_tool.description)
print(division_tool.args)

