import chromadb

# Create a persistent client
client = chromadb.PersistentClient(path="./chroma_db")

# Convert PDF to Text
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    """
    Load a PDF file and convert to text documents
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        list: List of document pages
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages

def create_chunks(documents, chunk_size=800, chunk_overlap=200):
    """
    Split documents into overlapping chunks
    
    Args:
        documents (list): List of documents to split
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# Delete old collection (optional, for a full refresh)
client.delete_collection(name="pdf_collection")


# Load and Chunk the PDF
#pdf_path = "./testing/einstein-albert.pdf"
pdf_path = "./testing/information1.pdf"
documents = load_pdf(pdf_path)
chunks = create_chunks(documents)

# Create a collection with OpenAI embeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# API Key from Env
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(os.getenv('OPENAI_API_KEY'))


embedding_function = OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"  # Latest OpenAI embedding model
)

collection = client.create_collection(
    name="pdf_collection",
    embedding_function=embedding_function
)

# Add documents to collection
documents = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
ids = [str(i) for i in range(len(chunks))]

# Add to collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# Get Statistics about the collection size
collection.count()