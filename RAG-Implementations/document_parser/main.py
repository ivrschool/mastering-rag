import os
import pickle
from utils.config_loader import load_config
from parsers.factory import get_parser
from chunkers.factory import get_chunker

def is_valid_parsed_file(path):
    try:
        with open(path, "rb") as f:
            records = pickle.load(f)
        if (
            isinstance(records, list)
            and len(records) > 0
            and all(isinstance(r, dict) for r in records)
            and all("_id" in r and "chunk_text" in r for r in records)
        ):
            print(f"[INFO] Skipping parsing — found {len(records)} valid records in {path}")
            return True
    except Exception as e:
        print(f"[WARN] Failed to validate existing parsed file: {e}")
    return False

def main():
    config = load_config()
    input_dir = config["GENERAL"]["input_dir"]
    output_dir = config["GENERAL"]["output_dir"]
    output_file = config["GENERAL"].get("output_file", "parsed_records.pkl")
    file_type = config["GENERAL"]["file_type"]
    chunker_name = config["GENERAL"]["chunker"]
    parser_mode = config["GENERAL"].get("parser_mode", "simple")

    parser = get_parser(file_type, mode=parser_mode)
    chunker = get_chunker(chunker_name)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Skip if valid parsed file already exists
    if os.path.exists(output_path) and is_valid_parsed_file(output_path):
        return

    doc_records = []

    for file_name in os.listdir(input_dir):
        if not file_name.endswith(f".{file_type}"):
            continue

        file_path = os.path.join(input_dir, file_name)
        text = parser.parse(file_path)
        chunks = chunker.chunk(text)

        base_id = os.path.splitext(file_name)[0]
        for idx, chunk in enumerate(chunks):
            doc_records.append({
                "_id": f"{base_id}_chunk{idx}",
                "chunk_text": chunk,
                "category": file_type
            })

    with open(output_path, "wb") as f:
        pickle.dump(doc_records, f)

    print(f"[INFO] Saved {len(doc_records)} chunks to {output_path}")

if __name__ == "__main__":
    main()
