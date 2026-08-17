# install depdendencies:
# pip install sqlite-vec
# pip install sentence-transformers

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from memory.vector_store import get_connection, create_store, semantic_search
from chat_client import chat


def main():
    db_path = Path(__file__).parent / "memory" / "Memory.db"
    connection = get_connection(str(db_path))

    create_store(connection)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    full_chat = chat(connection, model)

if __name__ == "__main__":
    main()
