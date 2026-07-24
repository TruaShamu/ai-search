"""
Pydantic models for the cleaned book data.
"""

from pydantic import BaseModel


class Book(BaseModel):
    work_id: str                        # OpenLibrary work key, e.g. "/works/OL123W"
    title: str
    authors: list[str]                  # resolved author names
    description: str | None = None      # parsed from JSON, truncated to 512 chars
    subjects: list[str]                 # normalized (lowercased, deduped)
    first_publish_year: int | None = None
    cover_id: int | None = None         # first cover ID for thumbnail URL
    subject_places: list[str] = []
    subject_people: list[str] = []
    subject_times: list[str] = []
    # Metadata for tiering
    tier: int = 3                       # 1=rich, 2=subjects-only, 3=title-only

    @property
    def cover_url_medium(self) -> str | None:
        if self.cover_id:
            return f"https://covers.openlibrary.org/b/id/{self.cover_id}-M.jpg"
        return None

    def embedding_text(self) -> str:
        """Build the text to embed for this book (Nomic task prefix included)."""
        author_str = ", ".join(self.authors) if self.authors else "Unknown"
        parts = [f"{self.title} by {author_str}"]

        if self.description:
            parts.append(self.description[:512])

        if self.subjects:
            parts.append(", ".join(self.subjects[:10]))

        if self.subject_people:
            parts.append("People: " + ", ".join(self.subject_people[:5]))
        if self.subject_places:
            parts.append("Places: " + ", ".join(self.subject_places[:5]))

        return "search_document: " + ". ".join(parts)

    @staticmethod
    def query_text(query: str) -> str:
        """Format a user query for embedding (Nomic task prefix)."""
        return "search_query: " + query
