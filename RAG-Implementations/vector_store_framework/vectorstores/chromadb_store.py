import os
import re
import chromadb
import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from vector_store_framework.vectorstores.base import BaseVectorStore
from vector_store_framework.vectorstores.registry import register_vector_store
from vector_store_framework.utils.config_loader import load_config, get_env
from typing import List, Dict

config = load_config()

def is_valid_repo_id(repo_id: str) -> bool:
    # Hugging Face model naming conventions
    return re.fullmatch(r"^[a-zA-Z0-9]+([\-_.]{1}[a-zA-Z0-9]+)*(/[a-zA-Z0-9]+([\-_.]{1}[a-zA-Z0-9]+)*)?$", repo_id) is not None

@register_vector_store("chromadb")
class ChromaDBStore(BaseVectorStore):
    def __init__(self):
        self.model_name = config["CHROMADB"].get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2").strip()
        self.namespace = config["CHROMADB"].get("namespace", "default")
        self.persist_path = config["CHROMADB"].get("persist_path", "vectorstores/data/chroma_db")
        self.token = get_env("HF_TOKEN")

        # Validate model name
        if not is_valid_repo_id(self.model_name):
            raise ValueError(f"[CONFIG ERROR] Invalid Hugging Face model name: '{self.model_name}'")

        # Setup ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_path)

        # Optional device preference
        device_pref = config["CHROMADB"].get("device", "auto")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_pref == "auto"
            else torch.device(device_pref)
        )
        print(f"[CHROMADB] Using device: {self.device}")

        # Embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            token=self.token
        )

        # Collection
        self.collection = self.client.get_or_create_collection(
            name=self.namespace,
            embedding_function=self.embedding_function
        )

    def upsert_documents(self, namespace: str, records: List[Dict]) -> None:
        ids = [rec["_id"] for rec in records]
        texts = [rec["chunk_text"] for rec in records]
        metadatas = [rec for rec in records]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def query(self, namespace: str, text: str, top_k: int = 5) -> List[Dict]:
        results = self.collection.query(query_texts=[text], n_results=top_k)

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "_id": results["ids"][0][i],
                "_score": results["distances"][0][i],
                "fields": results["metadatas"][0][i]
            })
        return hits

    def delete_ids(self, namespace: str, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
        print(f"[CHROMADB] Deleted {len(ids)} ID(s)")
