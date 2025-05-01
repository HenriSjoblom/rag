import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.config import settings
import uuid

# Dummy data
dummy_documents = [
    "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "To reset your password, click the 'Forgot Password' link on the login page.",
    "Our premium subscription costs $9.99 per month and includes ad-free access.",
    "You can contact customer support via email at support@example.com or by phone.",
    "The latest software update includes performance improvements and bug fixes.",
    "Shipping usually takes 3-5 business days within the continental US.",
    "Make sure your device is connected to Wi-Fi before attempting the update.",
]

def load_data():
    print("Loading dummy data into ChromaDB...")
    print(f"Using model: {settings.EMBEDDING_MODEL_NAME}")
    print(f"Using Chroma collection: {settings.CHROMA_COLLECTION_NAME}")
    print(f"Chroma mode: {settings.CHROMA_MODE}, Path: {settings.CHROMA_LOCAL_PATH}") # Debug print path

    # --- 1. Load Embedding Model ---
    try:
        model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return

    # --- 2. Connect to ChromaDB ---
    try:
        chroma_config = {
            "chroma_db_impl": "duckdb+parquet",
            "persist_directory": settings.CHROMA_LOCAL_PATH, # Use configured path
            # Add server settings here if using CHROMA_MODE='http'/'https'
        }
        chroma_settings_dict = {k: v for k, v in chroma_config.items() if v is not None}
        client = chromadb.Client(ChromaSettings(**chroma_settings_dict))
        # Ensure persistence path exists (handled in config validator now, but good practice)
        # if settings.CHROMA_MODE == 'local' and settings.CHROMA_LOCAL_PATH:
        #      os.makedirs(settings.CHROMA_LOCAL_PATH, exist_ok=True)

        # Delete collection if it exists (for clean reload) - optional
        try:
            print(f"Attempting to delete existing collection '{settings.CHROMA_COLLECTION_NAME}'...")
            client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            print("Existing collection deleted.")
        except Exception as e:
             # Collection might not exist, which is fine
             print(f"Could not delete collection (may not exist): {e}")


        collection = client.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
        print(f"ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' ready.")
    except Exception as e:
        print(f"Error connecting to or setting up ChromaDB: {e}")
        return

    # --- 3. Embed and Add Documents ---
    print(f"Embedding {len(dummy_documents)} documents...")
    try:
        embeddings = model.encode(dummy_documents, convert_to_tensor=False).tolist() # Ensure list output
        ids = [str(uuid.uuid4()) for _ in dummy_documents] # Generate unique IDs

        print(f"Adding {len(ids)} documents to the collection...")
        collection.add(
            embeddings=embeddings,
            documents=dummy_documents,
            ids=ids
            # You can add metadatas=[{"source": "dummy_faq"}] * len(dummy_documents) here too
        )
        print("Data added successfully!")
        print(f"Collection count: {collection.count()}")

    except Exception as e:
        print(f"Error embedding or adding documents: {e}")

if __name__ == "__main__":
    load_data()