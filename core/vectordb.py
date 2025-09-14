from qdrant_client import AsyncQdrantClient
import os
import dotenv

dotenv.load_dotenv()

client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL", ""),
    api_key=os.getenv("QDRANT_API_KEY", ""),
)
