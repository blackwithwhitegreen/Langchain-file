# Actually this code is not working with HuggingFace because of Free Api key, insted of we can use the ChatOpneAI form langchain.openai, here i use GoogleGenai model. 

# from langchain_community.llms import HuggingFaceEndpoint
# from langchain_community.chat_models import ChatHuggingFace
# from dotenv import load_dotenv, find_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv,find_dotenv
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

prompt1 = template1.invoke({'topic':'black hole'})
result = model_gen.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result1 = model_gen.invoke(prompt2)

print(result1.content)
