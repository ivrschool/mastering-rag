VECTOR_DB_REGISTRY = {}

def register_vector_store(name):
    def decorator(cls):
        VECTOR_DB_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_vector_store_class(name):
    if name.lower() not in VECTOR_DB_REGISTRY:
        raise ValueError(f"Vector store '{name}' not registered.")
    return VECTOR_DB_REGISTRY[name.lower()]
