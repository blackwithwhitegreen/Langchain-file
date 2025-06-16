from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# This is the way to create Dymanic Way of creaed meassages.
chat_template = ChatPromptTemplate([ #ChatPromptTemplate.from_messages ----> This also can be used.
    ('system','You are a helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}')

]) # This is the different way to create a messages

prompt = chat_template.invoke({'domain':'cricket','topic':'Free-hit'})

print(prompt) 