from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

load_dotenv(load_dotenv())

model_gen = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a frictional Person \n{format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model_gen | parser

result = chain.invoke({})

print(result)
print(type(result))

# Promblem with jsonparse is we can't say what's is the structuring of out json, LLm wil decide it by itself.   
