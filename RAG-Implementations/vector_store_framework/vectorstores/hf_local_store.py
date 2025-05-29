import os
import pickle
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
import torch
from vector_store_framework.vectorstores.base import BaseVectorStore
from vector_store_framework.vectorstores.registry import register_vector_store
from vector_store_framework.utils.config_loader import load_config, get_env

config = load_config()

@register_vector_store("hf_local")
class HuggingFaceLocalStore(BaseVectorStore):
    def __init__(self):
        self.model_name = config["HF_LOCAL"]["embedding_model"]
        self.namespace = config["HF_LOCAL"]["namespace"]
        self.storage_path = config["HF_LOCAL"]["storage_path"]
        # self.device = config["HF_LOCAL"].get("device", "cpu")
        # Determine device
        device_pref = config["HF_LOCAL"].get("device", "auto")
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_pref == "auto"
            else torch.device(device_pref)
        )
        print(f"[HF_LOCAL] Using device: {self.device}")


        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        # Load or initialize vector DB
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "rb") as f:
                self.vector_db = pickle.load(f)
        else:
            self.vector_db = {}

    def save_db(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.vector_db, f)

    def embed_text(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return embeddings.squeeze().tolist()


    def upsert_documents(self, namespace: str, records: List[Dict]) -> None:
        if namespace not in self.vector_db:
            self.vector_db[namespace] = []

        existing_ids = {doc["_id"] for doc in self.vector_db[namespace]}
        new_docs = []

        for record in records:
            if record["_id"] in existing_ids:
                continue  # skip existing
            vector = self.embed_text(record["chunk_text"])
            new_docs.append({
                "_id": record["_id"],
                "vector": vector,
                "fields": record
            })

        self.vector_db[namespace].extend(new_docs)
        self.save_db()

    def query(self, namespace: str, text: str, top_k: int = 5) -> List[Dict]:
        if namespace not in self.vector_db:
            return []

        query_vec = torch.tensor(self.embed_text(text))
        docs = self.vector_db[namespace]

        def cosine_score(vec1, vec2):
            v1 = torch.tensor(vec1)
            v2 = torch.tensor(vec2)
            return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()

        ranked = sorted([
            {"_id": doc["_id"], "_score": cosine_score(query_vec, doc["vector"]), "fields": doc["fields"]}
            for doc in docs
        ], key=lambda x: x["_score"], reverse=True)

        return ranked[:top_k]

    def delete_ids(self, namespace: str, ids: List[str]) -> None:
        if namespace in self.vector_db:
            original_count = len(self.vector_db[namespace])
            self.vector_db[namespace] = [
                doc for doc in self.vector_db[namespace] if doc["_id"] not in ids
            ]
            removed = original_count - len(self.vector_db[namespace])
            print(f"[HF_LOCAL] Deleted {removed} out of {len(ids)} ID(s)")
            self.save_db()
        else:
            print(f"[HF_LOCAL] Namespace '{namespace}' not found — nothing to delete")

