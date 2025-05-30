from llama_index.core import VectorStoreIndex, Settings

def build_rag_pipeline(nodes, embed_model, vector_store, llm):
    Settings.embed_model = embed_model
    Settings.llm = llm
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    return index.as_query_engine()
