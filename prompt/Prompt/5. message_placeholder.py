from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []


# load chat History
with open('chat-ai.txt')as f:
    chat_history.extend(f.readlines())

# print(chat_history)

# Create Prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print(prompt)
