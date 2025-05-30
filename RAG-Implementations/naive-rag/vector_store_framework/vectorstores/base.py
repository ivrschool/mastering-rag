from abc import ABC, abstractmethod
from typing import List, Dict

class BaseVectorStore(ABC):
    @abstractmethod
    def upsert_documents(self, namespace: str, records: List[Dict]) -> None:
        pass

    @abstractmethod
    def query(self, namespace: str, text: str, top_k: int = 5) -> List[Dict]:
        pass

    @abstractmethod
    def delete_ids(self, namespace: str, ids: List[str]) -> None:
        pass

