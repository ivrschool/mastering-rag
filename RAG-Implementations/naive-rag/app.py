import os
import pickle
import tempfile
import streamlit as st
from document_parser.utils.config_loader import load_config as load_parser_config
from document_parser.parsers.factory import get_parser
from document_parser.chunkers.factory import get_chunker
from vector_store_framework.vectorstores.factory import get_vector_store
from vector_store_framework.utils.data_loader import load_pkl
from llm_framework.llm.model_factory import get_model as get_llm_model

# --- Constants ---
TEMP_PARSED_DIR = tempfile.gettempdir()
PARSED_FILE = "chunks_output.pkl"

# --- Helpers ---
def parse_and_chunk_files(uploaded_files):
    config = load_parser_config("document_parser/config/config.ini")
    parser = get_parser("txt", mode=config["GENERAL"].get("parser_mode", "simple"))
    chunker = get_chunker(config["GENERAL"].get("chunker", "simple"))

    records = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        chunks = chunker.chunk(text)
        base_id = os.path.splitext(uploaded_file.name)[0]
        for idx, chunk in enumerate(chunks):
            records.append({
                "_id": f"{base_id}_chunk{idx}",
                "chunk_text": chunk,
                "category": "txt"
            })

    output_path = os.path.join(TEMP_PARSED_DIR, PARSED_FILE)
    with open(output_path, "wb") as f:
        pickle.dump(records, f)

    return output_path, records

def upsert_to_vector_store(records):
    config = load_parser_config("vector_store_framework/config/config.ini")
    store = get_vector_store()
    provider = config["VECTORSTORE"]["provider"]
    namespace = config[provider.upper()]["namespace"]
    store.upsert_documents(namespace, records)
    return store, namespace

def query_vector_store(store, namespace, query, top_k):
    return store.query(namespace, query, top_k=top_k)

def generate_final_answer(top_docs, query):
    llm = get_llm_model()
    context = "\n".join(doc["fields"]["chunk_text"] for doc in top_docs)
    prompt = f"Answer the following based on the context:\n{context}\n\nQuestion: {query}"
    return llm.generate_response(prompt)

# --- Streamlit App ---
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG-Powered Q&A Assistant")

# State variables
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
    st.session_state.records = []
    st.session_state.store = None
    st.session_state.namespace = None

# Upload and parse files once
with st.expander("Upload & Process Files", expanded=not st.session_state.documents_loaded):
    uploaded_files = st.file_uploader("Upload .txt files", type="txt", accept_multiple_files=True)
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload one or more .txt files.")
        else:
            with st.spinner("Parsing and chunking files..."):
                parsed_path, records = parse_and_chunk_files(uploaded_files)
            with st.spinner("Upserting into vector store..."):
                store, namespace = upsert_to_vector_store(records)

            # Store in session state
            st.session_state.documents_loaded = True
            st.session_state.records = records
            st.session_state.store = store
            st.session_state.namespace = namespace
            st.success("Documents successfully processed and stored!")

# Q&A Section
if st.session_state.documents_loaded:
    st.markdown("---")
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:")
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=5)

    if st.button("Get Answer"):
        with st.spinner("Querying..."):
            top_docs = query_vector_store(st.session_state.store, st.session_state.namespace, query, top_k)
            answer = generate_final_answer(top_docs, query)

        st.subheader("🔍 Retrieved Chunks:")
        for doc in top_docs:
            st.markdown(f"**ID**: {doc['_id']} — **Score**: {round(doc['_score'], 2)}")
            st.code(doc["fields"]["chunk_text"], language="text")

        st.subheader("Final Answer:")
        st.success(answer)
