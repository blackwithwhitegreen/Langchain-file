from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']

)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke {text}',
    input_variables=['text']

)

chain = RunnableSequence(prompt1,model,prompt2, model ,parser)

result = chain.invoke({'topic':'ai'})

print(result)
