import pickle
from typing import List, Dict

# def load_pkl(file_path: str) -> List[Dict]:
#     with open(file_path, "rb") as f:
#         return pickle.load(f)

def load_pkl(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[DEBUG] Loaded {len(data)} records from {path}, sample type: {type(data[0]) if data else 'empty'}")
    return data
