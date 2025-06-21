from langchain.schema.runnable import RunnableLambda,RunnableParallel,RunnablePassthrough, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
def word_count(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=1
)

prompt = PromptTemplate(
    template='Write a joke - {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)


# **************************** it is a cleaner way by generating a function and use it.**************************
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
})


# ************this Function also can be used ********************
# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count' : RunnableLambda(lambda x: len(x.split()))
# })

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)


result = final_chain.invoke({'topic':'ai'})

final_result = """{} \n word Count - {}""".format(result['joke'],result['word_count'])

print(final_result)

print(final_chain.get_graph().draw_ascii())

