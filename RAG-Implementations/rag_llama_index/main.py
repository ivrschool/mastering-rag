from src.config import load_config
from src.loader import load_documents
from src.chunker.factory import ChunkerFactory
from src.embedder.factory import EmbedderFactory
from src.vector_store.factory import VectorStoreFactory
from src.llm.factory import LLMFactory
from src.rag_pipeline import build_rag_pipeline

def main():
    config = load_config()

    documents = load_documents("data/sample_docs")
    chunker = ChunkerFactory.create_chunker(config)
    nodes = chunker.get_nodes_from_documents(documents)

    embed_model = EmbedderFactory.create_embedder(config)
    vector_store = VectorStoreFactory.create_vector_store(config)
    llm = LLMFactory.create_llm(config)

    query_engine = build_rag_pipeline(nodes, embed_model, vector_store, llm)
    response = query_engine.query("What is the purpose of this project?")
    print(response)

if __name__ == "__main__":
    main()
