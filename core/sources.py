class SourcesStore:
    def __init__(self):
        """Initialize the SourcesStore with an empty list."""
        self.sources = []

    def set_sources(self, sources: list[str]) -> None:
        """Store the sources."""
        # if self.sources is not None:
        if self.sources:
            self.sources.extend(sources)
        else:
            self.sources = sources

    def get_sources(self) -> list[str]:
        """Retrieve the stored sources."""
        return self.sources if hasattr(self, "sources") else []

    def clear_sources(self) -> None:
        """Clear the stored sources."""
        if hasattr(self, "sources"):
            del self.sources


ss = SourcesStore()
