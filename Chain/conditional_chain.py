from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive','Negitive'] = Field(discription = "Give the Feedback sentiment either Positive or Negitive.")


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template= "find the sentiment of the following feedback text into positive and negitive \n{feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template= "Write a response for Positive feedback \n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template= "Write a response for Negitive feedback \n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    #(condition, chain)
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negitive', prompt3 | model | parser),
    #default chain
    RunnableLambda(lambda x: "could not find Sentiment.") # Changing into runnable
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'This is a smartphone is one of the best feature and totally impressive'})
print(result)

chain.get_graph().print_ascii()