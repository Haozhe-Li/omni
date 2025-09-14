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
        # make sure each sources's url, title is non empty, otherwise remove that source
        unique_sources = [
            source
            for source in unique_sources
            if source.get("url") and source.get("title")
        ]
        if not hasattr(self, "sources") or not self.sources:
            self.sources = unique_sources
        else:
            self.sources.extend(unique_sources)
            # Ensure sources are unique after extending
            seen_urls = set()
            self.sources = [
                source
                for source in self.sources
                if not (
                    source.get("url") in seen_urls or seen_urls.add(source.get("url"))
                )
            ]

    def get_sources(self) -> list[dict]:
        """Retrieve the stored sources.

        Returns:
            list[dict]: The list of stored source dictionaries.
        """
        return self.sources if hasattr(self, "sources") else []

    def clear_sources(self) -> None:
        """Clear the stored sources."""
        self.sources = []


ss = SourcesStore()
