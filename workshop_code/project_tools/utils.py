import itertools
import json
from typing import Any, Dict, List, Optional

import chromadb
import matplotlib.pyplot as plt
import ollama
from ollama import ResponseError, pull, show

from project_tools.config import COLLECTION_NAME, DATA_FILE, EMBED_MODEL

DEFAULT_COLOR_CYCLE = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def load_json_file(filepath: str = DATA_FILE) -> Any:
    """Loads JSON data from the specified file."""
    with open(filepath, "r") as f:
        return json.load(f)


def initialize_model(model_name: str) -> None:
    """
    Initializes the Ollama model. If the model is not found, it attempts to pull it.
    """
    try:
        show(model_name)
    except ResponseError:
        print(f"Model '{model_name}' not found. Pulling it now...")
        pull(model_name)
        print("Model pulled successfully.")


def get_embedding(text: str, model: str = EMBED_MODEL) -> List[float]:
    """Retrieve the embedding for a given text using Ollama."""
    response = ollama.embed(model=model, input=text)
    embedding = response.get("embedding") or response.get("embeddings")
    if not embedding:
        raise ValueError("Embedding not found in the response.")
    # Return the first element if the embedding is nested (list of lists)
    return embedding[0] if isinstance(embedding[0], list) else embedding


def initialize_vector_store() -> Any:
    """
    Initializes a Chroma vector store collection named 'docs'.
    If the collection is empty, documents from DATA_FILE are loaded. For each document,
    if a precomputed 'embedding' exists, it is used; otherwise, one is generated.
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)

    if not collection.get()["ids"]:
        print("Loading documents from data.json into the vector store...")

        try:
            data = load_json_file(DATA_FILE)
            docs_data = data.get("sample_embeddings", [])

            if docs_data:
                # Batch the additions for improved performance.
                ids, embeddings, documents = [], [], []
                for doc in docs_data:
                    doc_id = doc.get("id", "")
                    text = doc.get("text", "")
                    if not text:
                        continue  # Skip if no text is provided.
                    if "embedding" in doc:
                        emb = doc["embedding"]
                    else:
                        emb = get_embedding(text)

                    ids.append(doc_id)
                    embeddings.append(emb)
                    documents.append(text)

                if ids:
                    collection.add(ids=ids, embeddings=embeddings, documents=documents)
                    print("Documents loaded into the vector store.")
                else:
                    print("No valid documents found to load.")
            else:
                print("No documents found under 'sample_embeddings' in data.json.")
        except Exception as e:
            print("Error loading data.json:", e)
    else:
        print("Vector store already contains documents.")

    return collection


def get_dynamic_category_colors(
    docs: List[Dict[str, Any]], provided_colors: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Generate a dynamic mapping from category to color.
    If provided_colors is given (from data.json), it is used as a base and augmented.
    """
    if provided_colors is None:
        provided_colors = {}
    unique_categories = sorted(
        {doc.get("source", "unknown") for doc in docs if "embedding" in doc}
    )
    for cat in unique_categories:
        if cat not in provided_colors:
            provided_colors[cat] = next(DEFAULT_COLOR_CYCLE)
    return provided_colors
