import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

dbpath = "./own_chroma_db"

# Load your custom documents
#############################################################################################
## ---------------------- 1. ------------------------------ ##
custom_texts = [
    "JNH is a famous talent IT engineer",
    "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
    "ChromaDB is an open-source vector database designed for storing and retrieving embeddings.",
    "Deepseek-R1-7B is a large language model optimized for chat-based applications."
]

# ## -------------------- 2. TXT FILE --------------------- ##
# # Read data from a text file
# with open("custom_data.txt", "r", encoding="utf-8") as file:
#     custom_texts = file.readlines()

# ## -------------------- 3. TXT FOLDER -------------------- ##
# data_directory = "./documents"
# custom_texts = []

# # Read all text files in the directory
# for filename in os.listdir(data_directory):
#     if filename.endswith(".txt"):
#         with open(os.path.join(data_directory, filename), "r", encoding="utf-8") as file:
#             custom_texts.append(file.read())

# ## -------------------- 4. PDF --------------------------- ##
# import fitz  # PyMuPDF

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as doc:
#         text = "\n".join([page.get_text("text") for page in doc])
#     return text

# # Load multiple PDFs
# pdf_files = ["document1.pdf", "document2.pdf"]
# custom_texts = [extract_text_from_pdf(pdf) for pdf in pdf_files]

# ## -------------------- 5. CSV --------------------------- ##
# import pandas as pd

# # Read CSV
# df = pd.read_csv("data.csv")

# # Assume you want to store text from a column named 'content'
# custom_texts = df["content"].dropna().tolist()

#############################################################################################

# Store in ChromaDB and save to disk
def strore_in_chroma_db():
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path = dbpath)

    # Load embeddings (you can change the model if needed)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(custom_texts)
    vector_db = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")

# Load from disk
def get_chroma_db_instance():
    db = Chroma(persist_directory = dbpath)
    return db;

def retrieve_docs(query, k=3):
    vector_db = get_chroma_db_instance()
    results = vector_db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])