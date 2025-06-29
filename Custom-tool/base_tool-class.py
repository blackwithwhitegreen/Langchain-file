from langchain.tools import BaseTool
from typing import Type 
from pydantic import BaseModel, Field


# arg schema using pydantic
class DivisionInput(BaseModel):
    a : int = Field(required = True, description="The first number to add")
    b : int = Field(required=True, description="The second number to add")


class DivisionTool(BaseTool):
    name: str = "divison",
    description: str = "Divison Two numbers",

    args_schema: Type[BaseModel] = DivisionInput

    def _run(self, a: int, b: int) -> int:
        return a/b
    

division_tool = DivisionTool()

result = division_tool.invoke({"a": 9, "b": 3})

print(result)
print(division_tool.description)
print(division_tool.args)
print(division_tool.name)