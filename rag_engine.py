import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="chicken_bizna",
    embedding_function=openai_ef
)

def retrieve_context(query: str, n_results: int = 3) -> str:
    """Retrieve the top matching context snippets from ChromaDB."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results.get("documents", [[]])[0]
        return "\n---\n".join(docs) if docs else "No relevant context found."
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "No relevant context found."