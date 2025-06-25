from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=400,
    do_sample=False,
    temperature=0.5
)

result = llm.invoke(
    "Summarize Attention all you need research paper in 150 words")
print(result)
