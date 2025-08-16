class SourcesStore:
    def __init__(self):
        """Initialize the SourcesStore with an empty list."""
        self.sources = []

    def set_sources(self, sources: list[dict]) -> None:
        """Store the sources.

        Args:
            sources (list[dict]): The list of source dictionaries to store.
        """
        # Remove duplicates based on URL to avoid duplicate sources
        seen_urls = set()
        unique_sources = []
        for source in sources:
            url = source.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        self.sources = unique_sources

    def get_sources(self) -> list[dict]:
        """Retrieve the stored sources.

        Returns:
            list[dict]: The list of stored source dictionaries.
        """
        return self.sources if hasattr(self, "sources") else []

    def clear_sources(self) -> None:
        """Clear the stored sources."""
        if hasattr(self, "sources"):
            del self.sources


ss = SourcesStore()
