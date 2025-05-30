import os
from vector_store_framework.utils.config_loader import load_config
from vector_store_framework.utils.data_loader import load_pkl
from vector_store_framework.vectorstores.factory import get_vector_store

def normalize_list(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]

def get_existing_ids(store, namespace: str, ids: list[str]) -> set:
    if hasattr(store, "index") and hasattr(store.index, "fetch"):
        try:
            response = store.index.fetch(ids=ids, namespace=namespace)
            return set(response.vectors.keys())
        except Exception as e:
            print(f"[ERROR] Failed to fetch existing vectors: {e}")
            return set()
    return set()  # For local store (HF_LOCAL) or unsupported DBs

def validate_records(records, source: str) -> bool:
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        print(f"[ERROR] File '{source}' does not contain a list of valid dictionaries.")
        return False
    return True

def main():
    config = load_config()
    store = get_vector_store()

    provider = config["VECTORSTORE"]["provider"]
    provider_section = config[provider.upper()]
    namespace = provider_section.get("namespace", "default")
    upserted_ids = []

    # ---------------- UPSERT ----------------
    upsert_files = normalize_list(config["UPSERT"].get("files", ""))
    for path in upsert_files:
        if not os.path.isfile(path):
            print(f"[WARN] Skipping missing file: {path}")
            continue

        records = load_pkl(path)
        if not validate_records(records, path):
            continue

        all_ids = [r["_id"] for r in records]
        existing_ids = get_existing_ids(store, namespace, all_ids)
        new_records = [r for r in records if r["_id"] not in existing_ids]

        if new_records:
            print(f"Upserting {len(new_records)} new record(s) from {path}")
            store.upsert_documents(namespace, new_records)
            upserted_ids.extend([r["_id"] for r in new_records])
        else:
            print(f"All {len(records)} record(s) from {path} already exist — skipping.")

    # ---------------- QUERY ----------------
    query_text = "Famous historical structures"
    print(f"\nQuerying for: '{query_text}'\n")
    results = store.query(namespace, query_text, top_k=5)
    for hit in results:
        print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")

    # ---------------- CONFIG-BASED DELETES ----------------
    delete_ids = normalize_list(config["DELETE"].get("ids", ""))
    if delete_ids:
        print(f"\nDeleting {len(delete_ids)} vector(s) by ID (from config)")
        if hasattr(store, "delete_ids"):
            store.delete_ids(namespace, delete_ids)
        else:
            store.index.delete(ids=delete_ids, namespace=namespace)

    delete_files = normalize_list(config["DELETE"].get("files", ""))
    for path in delete_files:
        if not os.path.isfile(path):
            print(f"[WARN] Skipping missing delete file: {path}")
            continue

        records = load_pkl(path)
        if not validate_records(records, path):
            continue

        ids = [rec["_id"] for rec in records]
        print(f"Deleting {len(ids)} vector(s) from {path}")
        if hasattr(store, "delete_ids"):
            store.delete_ids(namespace, ids)
        else:
            store.index.delete(ids=ids, namespace=namespace)

if __name__ == "__main__":
    main()
