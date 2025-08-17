from core.vectordb import client
from core.embedding import embedding_model
import uuid
import traceback


class SemanticSearchCache:
    def __init__(self, collection_name: str = "cache-1"):
        self.collection_name = collection_name

    def add(self, sources: list):
        try:
            if not sources:
                return

            # filter out source where from_cache is False
            sources = [
                source for source in sources if not source.get("from_cache", False)
            ]

            if not sources:
                return

            # embed each source, with source[query] + source[snippet] as text
            # Handle cases where query might not exist in source
            texts = []
            for source in sources:
                query_text = source.get("query", "")
                snippet_text = source.get("snippet", "")
                combined_text = f"{query_text} {snippet_text}".strip()
                texts.append(combined_text)

            embeddings = list(embedding_model.embed(texts))

            # upsert to Qdrant
            points = [
                {
                    "id": f"{str(uuid.uuid4())}",
                    "vector": embedding,
                    "payload": {
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "snippet": source.get("snippet", ""),
                        "query": source.get("query", ""),
                        "from_cache": True,
                    },
                }
                for i, (embedding, source) in enumerate(zip(embeddings, sources))
            ]

            client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print("Successfully added sources to cache.")
        except Exception as e:
            traceback.print_exc()
            print(f"Error adding sources to cache: {e}")

    def get(self, query: str, k: int = 5):
        try:
            # embed the query
            query_embedding = list(embedding_model.embed([query]))[0]

            # search in Qdrant
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.5,
            )

            # format results
            sources = [
                {
                    "url": point.payload.get("url", ""),
                    "title": point.payload.get("title", ""),
                    "snippet": point.payload.get("snippet", ""),
                    "query": point.payload.get("query", ""),
                    "from_cache": True,
                }
                for point in results
            ]

            return sources
        except Exception as e:
            traceback.print_exc()
            print(f"Error retrieving from cache: {e}")
            return []


semantic_cache = SemanticSearchCache()
