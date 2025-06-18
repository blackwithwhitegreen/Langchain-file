from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv(load_dotenv())

model_gen = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)



# we make schema with help fo ResponceSchema
schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')

]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model_gen | parser

result = chain.invoke({'topic':'black hole'})

print(result)


