from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader

url = "https://www.cricbuzz.com/cricket-news/134639/going-where-no-south-african-has-been-before"
loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)