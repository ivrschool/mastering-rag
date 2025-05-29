from vectorstores.factory import get_vector_store

def delete_vectors_by_id(ids, namespace="example-namespace"):
    """
    Delete one or more vectors from the specified namespace.
    :param ids: str or List[str] - vector ID(s) to delete
    :param namespace: str - namespace to delete from
    """
    if isinstance(ids, str):
        ids = [ids]  # normalize single ID to list

    store = get_vector_store()
    store.index.delete(ids=ids, namespace=namespace)
    print(f"Deleted {len(ids)} vector(s) from namespace '{namespace}'.")

# Example usage:
if __name__ == "__main__":
    # Delete a single ID
    delete_vectors_by_id("rec1")

    # Delete multiple IDs
    # delete_vectors_by_id(["rec2", "rec3", "rec4"])
