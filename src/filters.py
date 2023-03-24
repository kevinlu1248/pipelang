"""
Filters reduce the number of elements in a list.
"""

from pipeline import Pipeline, pipeline
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma


FilterLike = Pipeline[list[str], list[str]]
EmbeddingsLike = Pipeline[str, list[float]]
VectorStoreLike = Pipeline[list[float], list[str]]

def NoFilter() -> FilterLike:
    return lambda texts: texts

def LangchainEmbeddings(
    embeddings: Embeddings
) -> EmbeddingsLike:
    return embeddings.embed_query

def LangchainVectorStore(
    searcher: VectorStore
) -> VectorStoreLike:
    return searcher.similarity_search_by_vector

@pipeline
def EmbeddingsFilter(
    texts: list[str],
    searcher: VectorStoreLike,
    embeddings: EmbeddingsLike = LangchainEmbeddings(OpenAIEmbeddings()),
) -> list[str]:
    return searcher(embeddings(texts))

