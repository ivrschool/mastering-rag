# create_pkl.py
import pickle
from sample_docs import records  # your existing .py list

with open("sample_docs.pkl", "wb") as f:
    pickle.dump(records, f)
