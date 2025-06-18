from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv(load_dotenv())

model_gen = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

class person(BaseModel):

    name: str = Field(description='Name of a person')
    age: int = Field(gt=18, description='Age of the Person.')
    city: str = Field(description='Name of the city')


parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template='Generate the name, age and city {place} of a frictional person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)

chain = template | model_gen | parser

result = chain.invoke({'place':'indian'})

print(result)