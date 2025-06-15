from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="instagram_following_list_full.csv",encoding="utf-8")

docs = loader.load()

print(len(docs))
print(docs[40])