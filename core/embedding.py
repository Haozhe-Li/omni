from pathlib import Path
from fastembed import TextEmbedding, SparseTextEmbedding

# Get the project root directory (parent of 'core' directory)
current_file = Path(__file__)
project_root = current_file.parent.parent
dense_model_path = project_root / "models" / "bge-small-en-v1.5"
sparse_model_path = project_root / "models" / "bm25"

dense_embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    specific_model_path=str(dense_model_path),
)

sparse_embedding_model = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    specific_model_path=str(sparse_model_path),
)
