from core.vectordb import client
from qdrant_client import models
from core.embedding import dense_embedding_model, sparse_embedding_model
import uuid
import traceback


class SemanticSearchCache:
    def __init__(self, collection_name: str = "cache-2"):
        self.collection_name = collection_name
        self.useCache = True
        self.collectDataToCache = True

    def set_cache_settings(
        self, useCache: bool = True, collectDataToCache: bool = True
    ):
        self.useCache = useCache
        self.collectDataToCache = collectDataToCache

    def add(self, sources: list):
        if not self.collectDataToCache:
            print("Cache is disabled, not adding sources.")
            return
        try:
            if not sources:
                return

            # filter out source where aviod_cache is False
            sources = [
                source for source in sources if not source.get("aviod_cache", True)
            ]

            if not sources:
                return

            # embed each source, with source[query] + source[snippet] as text
            # Handle cases where query might not exist in source
            texts = []
            for source in sources:
                query_text = source.get("query", "")
                snippet_text = source.get("snippet", "")
                title_text = source.get("title", "")
                if query_text:
                    combined_text = (
                        f"{query_text}".strip()
                    )  # if query_text is present, then only use it
                else:
                    combined_text = f"{snippet_text} {title_text}".strip()
                print(f"Embedding text: {combined_text}")
                texts.append(combined_text)

            dense_embeddings = list(dense_embedding_model.embed(texts))
            sparse_embeddings = list(sparse_embedding_model.embed(texts))

            # upsert to Qdrant
            points = [
                {
                    "id": f"{str(uuid.uuid4())}",
                    "vector": {
                        "bge_dense_vector": dense_embedding,
                        "bm25_sparse_vector": sparse_embedding.as_object(),
                    },
                    "payload": {
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "snippet": source.get("snippet", ""),
                        "query": source.get("query", ""),
                        "aviod_cache": True,
                    },
                }
                for i, (dense_embedding, sparse_embedding, source) in enumerate(
                    zip(dense_embeddings, sparse_embeddings, sources)
                )
            ]

            client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print("Successfully added sources to cache.")
        except Exception as e:
            traceback.print_exc()
            print(f"Error adding sources to cache: {e}")

    def get(self, query: str, k: int = 5, threshold: float = 0.8):
        if not self.useCache:
            print("Cache is disabled, not retrieving sources.")
            return []
        try:
            prefetch = [
                models.Prefetch(
                    query=next(dense_embedding_model.query_embed(query)),
                    using="bge_dense_vector",
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=(next(sparse_embedding_model.query_embed(query)).as_object()),
                    using="bm25_sparse_vector",
                    limit=k * 2,
                ),
            ]

            results = client.query_points(
                self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                with_payload=True,
                limit=k,
                score_threshold=threshold,
            )

            # format results
            sources = [
                {
                    "url": point.payload.get("url", ""),
                    "title": point.payload.get("title", ""),
                    "snippet": point.payload.get("snippet", ""),
                    "query": point.payload.get("query", ""),
                    "aviod_cache": True,
                }
                for point in results.points
            ]

            return sources
        except Exception as e:
            traceback.print_exc()
            print(f"Error retrieving from cache: {e}")
            return []


semantic_cache = SemanticSearchCache()
