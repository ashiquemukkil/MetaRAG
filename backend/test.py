

from backend.llama2 import SaladOllamaEmbeddings
from langchain.vectorstores import Chroma


persist_directory = f"docs/02c332bb-5496-42f8-8ee6-2f4eb03f9ea7/chroma"
embedding = SaladOllamaEmbeddings(model="llama2")

vdb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)
vdb.persist()
print(vdb.similarity_search_with_relevance_scores("what is total profit"))