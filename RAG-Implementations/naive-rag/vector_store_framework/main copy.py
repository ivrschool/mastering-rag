from vectorstores.factory import get_vector_store
from vectorstores.data.sample_docs import records

def get_new_records(index, namespace, all_records):
    """Return only records whose _id is not in the vector DB."""
    all_ids = [rec["_id"] for rec in all_records]
    fetch_response = index.fetch(ids=all_ids, namespace=namespace)
    existing_ids = set(fetch_response.vectors.keys())
    new_records = [rec for rec in all_records if rec["_id"] not in existing_ids]
    return new_records

def main():
    store = get_vector_store()
    namespace = "example-namespace"
    query = "Famous historical structures"

    # ✅ Check and insert only new records
    new_records = get_new_records(store.index, namespace, records)
    if new_records:
        print(f"Upserting {len(new_records)} new records...")
        store.upsert_documents(namespace, new_records)
    else:
        print("No new documents to upsert.")

    # 🔍 Query the vector DB
    print(f"\nQuerying for: '{query}'\n")
    results = store.query(namespace, query, top_k=5)

    for hit in results:
        print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['chunk_text']:<50}")

if __name__ == "__main__":
    main()
