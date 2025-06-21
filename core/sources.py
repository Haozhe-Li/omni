class SourcesStore:
    def set_sources(self, sources: list[str]) -> None:
        """Store the sources."""
        self.sources = sources

    def get_sources(self) -> list[str]:
        """Retrieve the stored sources."""
        return self.sources if hasattr(self, "sources") else []

    def clear_sources(self) -> None:
        """Clear the stored sources."""
        if hasattr(self, "sources"):
            del self.sources


ss = SourcesStore()
