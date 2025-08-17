from pathlib import Path
from fastembed import TextEmbedding

# Get the project root directory (parent of 'core' directory)
current_file = Path(__file__)
project_root = current_file.parent.parent
model_path = project_root / "models" / "bge-small-en-v1.5"

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    specific_model_path=str(model_path),
)
