from pinecone import Pinecone
from vectorstores.base import BaseVectorStore
from vectorstores.registry import register_vector_store
from utils.config_loader import load_config, get_env
import time
from typing import List, Dict

config = load_config()

@register_vector_store("pinecone")
class PineconeStore(BaseVectorStore):
    def __init__(self):
        self.api_key = get_env("PINECONE_API_KEY")
        self.index_name = config["PINECONE"]["index_name"]
        self.model = config["PINECONE"]["embedding_model"]
        self.namespace = config["PINECONE"]["namespace"]
        self.region = config["PINECONE"]["region"]
        self.cloud = config["PINECONE"]["cloud"]

        self.pc = Pinecone(api_key=self.api_key)

        # ✅ Use list_indexes() to avoid duplicate index creation
        existing_indexes = self.pc.list_indexes()
        if self.index_name not in existing_indexes:
            try:
                self.pc.create_index_for_model(
                    name=self.index_name,
                    cloud=self.cloud,
                    region=self.region,
                    embed={"model": self.model, "field_map": {"text": "chunk_text"}}
                )
            except Exception as e:
                print(f"[WARN] Index creation skipped: {e}")

        self.index = self.pc.Index(self.index_name)

    def upsert_documents(self, namespace: str, records: List[Dict]) -> None:
        self.index.upsert_records(namespace, records)
        time.sleep(2)  # Slight delay to ensure records are indexed

    def query(self, namespace: str, text: str, top_k: int = 5) -> List[Dict]:
        results = self.index.search(
            namespace=namespace,
            query={"top_k": top_k, "inputs": {"text": text}}
        )
        return results["result"]["hits"]
    
    def delete_ids(self, namespace: str, ids: list[str]) -> None:
        self.index.delete(ids=ids, namespace=namespace)

