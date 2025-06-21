from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableBranch, RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=1
)

prompt1 = PromptTemplate(
    template='Write a Detail report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chian = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chian)

result = final_chain.invoke({'topic':'Ai vs Humans'})

print(result)

print(final_chain.get_graph().draw_ascii())


