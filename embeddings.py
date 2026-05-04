from typing import List
from langchain_core.embeddings import Embeddings
import chromadb.utils.embedding_functions as ef


class ChromaDefaultEmbeddings(Embeddings):
    def __init__(self):
        self._fn = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]