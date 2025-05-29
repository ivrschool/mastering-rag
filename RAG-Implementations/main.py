import os
import sys
import pickle
import shutil
from document_parser.utils.config_loader import load_config as load_parser_config
from llm_framework.llm.model_factory import get_model as get_llm_model
from document_parser.parsers.factory import get_parser
from document_parser.chunkers.factory import get_chunker
from vector_store_framework.utils.data_loader import load_pkl
from vector_store_framework.vectorstores.factory import get_vector_store


# Add the project root to sys.path so imports from sibling modules work


# ---- Configuration ----
TEXT_FILES = ["projects/text_files/romantic_story.txt", "projects/text_files/tech_story.txt"]
PARSED_DIR = "rag_orchestrator/parsed_output"
PARSED_FILE = "chunks_output.pkl"
QUERY = "Who is Elira Whitman?"
TOP_K = 5

# ---- Helpers ----
def parse_and_chunk_files(files):
    os.makedirs(PARSED_DIR, exist_ok=True)
    config = load_parser_config("document_parser/config/config.ini")
    parser = get_parser("txt", mode=config["GENERAL"].get("parser_mode", "simple"))
    chunker = get_chunker(config["GENERAL"].get("chunker", "simple"))
    output_path = os.path.join(PARSED_DIR, PARSED_FILE)

    records = []
    for file_path in files:
        text = parser.parse(file_path)
        chunks = chunker.chunk(text)
        base_id = os.path.splitext(os.path.basename(file_path))[0]
        for idx, chunk in enumerate(chunks):
            records.append({
                "_id": f"{base_id}_chunk{idx}",
                "chunk_text": chunk,
                "category": "txt"
            })

    with open(output_path, "wb") as f:
        pickle.dump(records, f)
    return output_path

def upsert_and_query_vector_store(parsed_pkl_path, query):
    config = load_parser_config("vector_store_framework/config/config.ini")
    store = get_vector_store()
    provider = config["VECTORSTORE"]["provider"]
    namespace = config[provider.upper()]["namespace"]

    records = load_pkl(parsed_pkl_path)
    store.upsert_documents(namespace, records)
    top_docs = store.query(namespace, query, top_k=TOP_K)
    return top_docs

def generate_final_answer(top_docs, query):
    llm = get_llm_model()
    context = "\n".join(doc["fields"]["chunk_text"] for doc in top_docs)
    prompt = f"Answer the following based on the context:\n{context}\n\nQuestion: {query}"
    return llm.generate_response(prompt)

# ---- Pipeline Execution ----
def main():
    parsed_pkl_path = parse_and_chunk_files(TEXT_FILES)
    top_docs = upsert_and_query_vector_store(parsed_pkl_path, QUERY)
    final_answer = generate_final_answer(top_docs, QUERY)
    print("\n=== Final Answer ===\n", final_answer)

if __name__ == "__main__":
    main()
