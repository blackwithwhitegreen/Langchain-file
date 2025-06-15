from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "books",
    glob= "*.pdf", # txt --> "**/*.txt", csv ---> "data/*.csv", Any type of folder ----> "**/*"
    loader_cls=PyPDFLoader
)
docs = loader.lazy_load() # lazy_load runs faster as compare to load()

for document in docs:
    print(document.metadata)
