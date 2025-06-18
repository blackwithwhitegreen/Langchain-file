from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv,find_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_gen = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 0.5
)


# 1st prompt -> detailed report
template1 = PromptTemplate(
    template  = "Write a detailed report on {topic}",
    input_variable = ['topic']
)

# 2nd Prompt -> Summary
template2 = PromptTemplate(
    template = "Write a 5 line summarize on the following text. /n{text}",
    input_varibale = ['text']
)

# Firslty create output-parser
parser = StrOutputParser()

# now forming a Chain
# Chian is a pipline which helps to execute

# After forming template1 we send it out model and model now generate so manuy thing like content, metadata, garbage values etc. for removing that garbage and we get the detailed report by using of parser. Now we send it to template2 and then to model, and now my model will generate 5 line summary as per task, Now again we get the garbage value and all so we again use Parser 

chain = template1 | model_gen | parser | template2 | model_gen | parser

result = chain.invoke({'topic':'black hole'})

print(result)


