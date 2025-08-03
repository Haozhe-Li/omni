class SourcesStore:
    def __init__(self):
        """Initialize the SourcesStore with an empty list."""
        self.sources = []

    def set_sources(self, sources: list[str]) -> None:
        """Store the sources.

        Args:
            sources (list[str]): The list of source strings to store.
        """
        sources = list(set(sources))
        self.sources = sources

    def get_sources(self) -> list[str]:
        """Retrieve the stored sources.

        Returns:
            list[str]: The list of stored source strings.
        """
        return self.sources if hasattr(self, "sources") else []

    def clear_sources(self) -> None:
        """Clear the stored sources."""
        if hasattr(self, "sources"):
            del self.sources


ss = SourcesStore()
