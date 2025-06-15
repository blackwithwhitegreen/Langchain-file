from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('ml.pdf')


docs = loader.load()

print(len(docs))
print(docs[15].page_content)

print(docs[15].metadata)
