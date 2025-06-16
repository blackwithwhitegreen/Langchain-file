from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv,find_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage


load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model = "models/gemini-1.5-flash-latest",
    temperature = 1.5
)

chat_history = [
    SystemMessage(content="You are helpful AI assistant")
] # Creating a Chat history for context 

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("\n AI:\n",result.content)

print(chat_history)