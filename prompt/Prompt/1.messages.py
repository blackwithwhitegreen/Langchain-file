from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

# loading model requiring Api Key.
model = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 1.5
)

messages  = [
    SystemMessage(content="You are a helpful Assistant"),
    HumanMessage(content="Tell me about Langchain")
]

result = model.invoke(messages)

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)


