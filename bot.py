import os
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Embedding Function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# Initialize ChromaDB persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Recreate collection with OpenAI embedding function
try:
    client.delete_collection(name="chicken_bizna")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="chicken_bizna",
    embedding_function=openai_ef
)

# Extract and chunk PDF text
reader = PdfReader("chicken_bizna.pdf")
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Simple chunking by paragraph/lines
chunks = [chunk.strip() for chunk in full_text.split("\n\n") if len(chunk.strip()) > 30]

# Add documents to ChromaDB
ids = [f"doc_{i}" for i in range(len(chunks))]
collection.add(
    documents=chunks,
    ids=ids
)

print(f"Successfully embedded {len(chunks)} chunks using OpenAI embeddings into ChromaDB.")