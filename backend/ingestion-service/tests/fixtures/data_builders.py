from typing import List

from langchain_core.documents import Document


class DocumentFactory:
    @staticmethod
    def create_simple(content: str = "test content") -> Document:
        return Document(
            page_content=content, metadata={"source": "test.pdf", "page": 1}
        )

    @staticmethod
    def create_batch(count: int = 3) -> List[Document]:
        return [DocumentFactory.create_simple(f"Content {i}") for i in range(count)]

    @staticmethod
    def create_large(size_kb: int = 100) -> Document:
        content = "x" * (size_kb * 1024)
        return Document(
            page_content=content,
            metadata={"source": "large.pdf", "size": f"{size_kb}KB"},
        )
