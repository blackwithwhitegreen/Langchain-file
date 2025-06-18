from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 0.5
)

template = PromptTemplate(
    template= "Generate 5 intersting facts about {topic}",
    input_variables=['topic']

)

parser = StrOutputParser()

chain = template | model | parser
result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()# this line is for vizualizaion 