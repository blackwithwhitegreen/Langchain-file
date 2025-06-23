# Acutully Langchain SemanticChunker is not performed very well and it's in the experimental stage, due to this is not giving the best results, instead we can use the RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Ye SemanticChunker ka object banata hai — jo ki text ko smartly todta hai (split karta hai) uske meaning ke basis par.
# Na ki sirf fixed length ya number of tokens pe — balki jab text ka meaning change hota hai tabhi naya chunk banata hai.
# Model har sentence ka vector (embedding) banata hai — fir chunker check karta hai ki sentence ke meaning kitne alag ya milte-julte hain.
# breakpoint_threshold_type="standard_deviation"
# Chunker ye dekhta hai: lagataar sentences kitne similar hain.
# Jab kisi 2 sentences ke beech similarity normal se zyada kam ho jaati hai (yaani meaning badal gaya), to waha chunk tod deta hai. ka matlab hai: average similarity se kitna deviation ho raha hai, uss hisaab se chunk banta hai.
# breakpoint_threshold_amount=1
# Ye decide karta hai kitna sensitive ho chunker.
# Agar value 1 hai, to wo break karega jab similarity 1 standard deviation se zyada gir jaaye.
# Zyada value = kam chunks
# Chhoti value = zyada detailed chunking (zyada todta hai)
text_splitter = SemanticChunker(
    embeddings=embeddings, breakpoint_threshold_type ="standard_deviation",
    breakpoint_threshold_amount = 1
)

sample = """
Object Oriented Programming is a fundamental concept in Python, empowering developers to build modular, maintainable, and scalable applications. By understanding the core OOP principles (classes, objects, inheritance, encapsulation, polymorphism, and abstraction), programmers can leverage the full potential of Python OOP capabilities to design elegant and efficient solutions to complex problems.
Lung cancer symptoms can include a persistent cough, coughing up blood, chest pain, shortness of breath, hoarseness, and fatigue.
Early signs of lung cancer can include a persistent cough, chest pain, shortness of breath, wheezing, hoarseness, and recurring respiratory infections. Weight loss, fatigue, and coughing up blood are also potential indicators. It's crucial to note that some individuals may not experience any symptoms in the early stages

Investigators have been at the site where Air India Flight AI171 crashed yesterday moments after take-off in the west Indian city of Ahmedabad.All but one of the 242 people on the London-bound flight died when it crashed into a residential area, with a local senior health official telling the BBC today at least eight people in the area it came down in were also killed.

"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)










