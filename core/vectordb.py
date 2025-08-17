from qdrant_client import QdrantClient
import os
import dotenv

dotenv.load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL", ""),
    api_key=os.getenv("QDRANT_API_KEY", ""),
)
