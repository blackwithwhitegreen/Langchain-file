from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv(find_dotenv())

# Step 1a Indexing

video_id = "cdiD-9MMpb0"
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join(chunk['text'] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")


# Step 1b Indexing(Text Splitter)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Step 1c and 1d - Indexing(Embedding Generation and storing in vector store)

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# Step 2 Retrieval
retriever = vector_store.as_retriever(
    search_type='similarity', search_kwargs={"k": 4})


# Step 3 Augementaion
prompt = PromptTemplate(
    template="""
        You are a helpful assistant,
        Answer Only from the provide transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
""",
    input_variables=['context', 'question']
)

# question = "is the topic of Transformer discussed in this video? if yes then what was discussed"
# retrieved_docs = retriever.invoke(question)

# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# final_prompt = prompt.invoke({"context": context_text, "question": question})


# Chain Creating

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# Step 4 Generation
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest", temperature=0.5)


# parallel_chain.invoke('What is Bots')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke('can you summarize the video')
print(result)
