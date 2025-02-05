#!/usr/bin/env python3
"""
Interactive 3D visualization of word and sentence embeddings.

Features:
  - Supports both word-level and sentence-level embeddings.
  - Allows dynamic input of new words/sentences with category assignment.
  - Uses PCA for dimensionality reduction.
  - Provides three plotting modes:
      1. Individual points with text annotations.
      2. Cluster shading (grouped by category with convex hulls and centroids).
      3. Both individual points and clusters.
  - Loads dynamic categories and assigns colors from data.json.
  - Prompts for the number of documents to load before computing embeddings.
  - Logs the total number of embeddings loaded from data.json.
  - Updates the plot in real-time.
"""

import logging
import queue
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Local project modules.
from project_tools.config import DATA_FILE, EMBED_MODEL
from project_tools.utils import (
    DEFAULT_COLOR_CYCLE,
    get_dynamic_category_colors,
    get_embedding,
    initialize_model,
    load_json_file,
)
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Constants
PCA_SCALE_FACTOR = 10  # Increase point separation in visualization

# Configure logging.
logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_sample_embeddings(
    data_type: str, max_docs: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Load words or sentences from JSON, computing missing embeddings if necessary.

    Steps:
      - Load the JSON data from DATA_FILE.
      - Slice the document list to max_docs (if provided) before computing embeddings.
      - Ensure the embedding model is available.
      - Compute missing embeddings for each document.
      - Log a summary with the total number of embeddings loaded.

    Args:
        data_type: Either "words" or "sentences".
        max_docs: Maximum number of documents to load (if None, load all).

    Returns:
        A tuple containing:
          - A list of document dictionaries.
          - A mapping from category names to color strings.
    """
    try:
        data = load_json_file(DATA_FILE)
    except Exception as e:
        logging.error("Error loading %s: %s", DATA_FILE, e)
        sys.exit(1)

    # Determine which key to use based on the provided data type.
    key = "sample_words" if data_type == "words" else "sample_embeddings"
    docs_data: List[Dict[str, Any]] = data.get(key, [])
    if not docs_data:
        logging.error("No %s found in %s under key '%s'.", data_type, DATA_FILE, key)
        sys.exit(1)

    # Limit the number of documents (if requested) before processing.
    if max_docs is not None and max_docs >= 0:
        docs_data = docs_data[:max_docs]

    # Load any provided category colors.
    provided_colors: Optional[Dict[str, str]] = data.get("category_colors", None)

    # Initialize the embedding model.
    initialize_model(EMBED_MODEL)

    # Compute missing embeddings.
    embedding_count = 0
    for doc in docs_data:
        if "embedding" not in doc:
            try:
                doc["embedding"] = get_embedding(doc["text"])
            except Exception as e:
                logging.warning(
                    "Error computing embedding for '%s': %s", doc.get("text", "N/A"), e
                )
                continue
        embedding_count += 1

    logging.info("Loaded %d embeddings from %s", embedding_count, DATA_FILE)
    return docs_data, get_dynamic_category_colors(docs_data, provided_colors)


def compute_pca(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize and reduce dimensions of embeddings using PCA.

    If there are fewer than 3 samples, compute as many components as possible
    and pad the result with zeros to always return an array of shape (n_samples, 3).

    Args:
        embeddings: A NumPy array of shape (n_samples, n_features).

    Returns:
        A NumPy array of shape (n_samples, 3) containing the PCA-reduced embeddings.
    """
    n_samples, n_features = embeddings.shape

    # When not enough samples to compute 3 components:
    if n_samples < 3:
        # Scale the embeddings.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(embeddings)
        n_components = n_samples  # can only compute as many components as samples
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        # Pad with zeros to get 3 dimensions.
        pad_width = 3 - n_components
        if pad_width > 0:
            zeros = np.zeros((n_samples, pad_width))
            X_reduced = np.hstack([X_reduced, zeros])
        return X_reduced * PCA_SCALE_FACTOR

    # Otherwise, there are enough samples: compute 3 components.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced * PCA_SCALE_FACTOR


def update_plot(
    docs: List[Dict[str, Any]], ax: Any, category_colors: Dict[str, str], plot_mode: str
) -> None:
    """
    Update the 3D plot of embeddings.

    There are three modes:
      - "points": Display individual points with text labels.
      - "clusters": Group points by category with convex hull shading and centroids.
      - "both": Overlay text labels on top of cluster plots.

    Args:
        docs: List of document dictionaries (each must include an "embedding" key).
        ax: The 3D matplotlib axis to update.
        category_colors: Mapping from category names to colors.
        plot_mode: Plotting mode ("points", "clusters", or "both").
    """
    # Filter documents that have an embedding.
    valid_docs = [doc for doc in docs if "embedding" in doc]
    if not valid_docs:
        logging.warning("No embeddings available to plot.")
        return

    embeddings = np.array([doc["embedding"] for doc in valid_docs])
    labels = [doc["text"] for doc in valid_docs]
    categories = [doc.get("source", "unknown") for doc in valid_docs]

    # Compute PCA reduction.
    X_reduced = compute_pca(embeddings)

    # Clear the previous plot.
    ax.cla()

    if plot_mode == "points":
        # Map each document's category to its assigned color.
        color_mapping = [category_colors.get(cat, "gray") for cat in categories]
        ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            X_reduced[:, 2],
            c=color_mapping,
            marker="o",
            alpha=0.75,
            depthshade=True,
        )
        # Annotate each point.
        for i, label in enumerate(labels):
            ax.text(
                X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], label, fontsize=9
            )

        # Create a legend for categories.
        unique_categories = sorted(set(categories))
        legend_patches = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=cat,
                markersize=8,
                markerfacecolor=category_colors.get(cat, "gray"),
            )
            for cat in unique_categories
        ]
        ax.legend(handles=legend_patches, title="Categories", loc="best")

    elif plot_mode in ("clusters", "both"):
        unique_categories = sorted(set(categories))
        for cat in unique_categories:
            indices = [i for i, c in enumerate(categories) if c == cat]
            points = X_reduced[indices]
            color = category_colors.get(cat, "gray")
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                marker="o",
                alpha=0.75,
                depthshade=True,
                label=cat,
            )
            # Compute and annotate the centroid.
            centroid = points.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                centroid[2],
                cat,
                fontsize=12,
                weight="bold",
                color="black",
            )
            # Draw convex hull if enough points are available.
            if len(points) >= 4:
                try:
                    hull = ConvexHull(points)
                    facets = [points[simplex] for simplex in hull.simplices]
                    poly = Poly3DCollection(facets, alpha=0.2, facecolor=color)
                    ax.add_collection3d(poly)
                except Exception as e:
                    logging.warning(
                        "Unable to compute convex hull for category '%s': %s", cat, e
                    )

        ax.legend(title="Categories", loc="best")
        # Overlay text labels if mode is "both".
        if plot_mode == "both":
            for i, label in enumerate(labels):
                ax.text(
                    X_reduced[i, 0], X_reduced[i, 1], X_reduced[i, 2], label, fontsize=9
                )

    ax.set_title("Word & Sentence Embeddings in 3D")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    plt.draw()
    plt.pause(0.001)


def interactive_update(
    data_type: str,
    docs: List[Dict[str, Any]],
    category_colors: Dict[str, str],
    plot_mode: str,
) -> None:
    """
    Run the interactive update loop for dynamic addition of documents.

    Users can input new words/sentences and assign categories. Input calls are handled
    in a separate thread, and an event is used to signal termination.
    """
    plt.ion()  # Enable interactive mode.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    update_plot(docs, ax, category_colors, plot_mode)

    new_docs_queue: "queue.Queue[Any]" = queue.Queue()
    stop_event = threading.Event()

    def input_thread() -> None:
        def embed_input(text: str, category: str, doc_count: int):
            try:
                new_embedding = get_embedding(text)
                new_doc = {
                    "id": f"user_{doc_count}",
                    "text": text,
                    "source": category,
                    "embedding": new_embedding,
                }
                doc_count += 1
                logging.info(
                    "Queued new document: %s (category: '%s')",
                    new_doc["id"],
                    new_doc["source"],
                )
                return new_doc, doc_count
            except Exception as e:
                logging.error("Error computing embedding: %s", e)
                return None, doc_count

        new_doc_count = 1
        while not stop_event.is_set():
            try:
                user_input = input(
                    "\nEnter new word/sentence (or 'exit' to quit): "
                ).strip()
            except EOFError:
                break
            if user_input.lower() in ("exit", "quit"):
                stop_event.set()
                new_docs_queue.put("exit")
                break
            category_input = (
                input(
                    "Enter category for this document (or press Enter for 'unknown'): "
                ).strip()
                or "unknown"
            )
            result, new_doc_count = embed_input(
                user_input, category_input, new_doc_count
            )
            if result is not None:
                new_docs_queue.put(result)

    # Start the input thread as a daemon.
    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()

    try:
        while not stop_event.is_set():
            flag = False
            while not new_docs_queue.empty():
                new_item = new_docs_queue.get()
                if new_item == "exit":
                    logging.info("Exiting interactive mode.")
                    stop_event.set()
                    break
                else:
                    flag = True
                    docs.append(new_item)
                    if new_item["source"] not in category_colors:
                        category_colors[new_item["source"]] = next(
                            DEFAULT_COLOR_CYCLE, "gray"
                        )
                    logging.info(
                        "Added document %s with category '%s'",
                        new_item["id"],
                        new_item["source"],
                    )
            if flag:
                update_plot(docs, ax, category_colors, plot_mode)
            plt.pause(0.5)
    except KeyboardInterrupt:
        logging.info("Interactive update interrupted by user.")
    finally:
        stop_event.set()
        plt.ioff()
        plt.show()


def stream_update(
    docs: List[Dict[str, Any]], category_colors: Dict[str, str], plot_mode: str
) -> None:
    """
    Stream update mode: plot each document into the 3D plot one-by-one,
    with a 1-second pause between each update.
    """
    plt.ion()  # Enable interactive mode.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    current_docs: List[Dict[str, Any]] = []

    for doc in docs:
        current_docs.append(doc)
        if doc["source"] not in category_colors:
            category_colors[doc["source"]] = next(DEFAULT_COLOR_CYCLE, "gray")
        update_plot(current_docs, ax, category_colors, plot_mode)
        plt.pause(1)  # Pause for 1 second between updates

    plt.ioff()
    plt.show()


def main() -> None:
    """
    Main execution function:
      - Prompts the user for the data type and maximum document count.
      - Loads sample embeddings.
      - Prompts for the plotting mode.
      - Then, prompts for an update mode:
          (1) Interactive mode (dynamic input via a separate thread)
          (2) Stream update mode (plots each doc second-by-second)
    """
    data_type = input("Choose 'words' or 'sentences': ").strip().lower()
    max_docs: Optional[int] = None
    max_docs_input = input(
        "Enter maximum number of documents to load (or press Enter for all): "
    ).strip()
    if max_docs_input:
        try:
            max_docs = int(max_docs_input)
        except ValueError:
            logging.warning("Invalid number entered; loading all documents.")

    # Plot mode selection.
    print("\nSelect plot mode:")
    print("1: Individual points")
    print("2: Cluster shading (grouped by category with convex hulls)")
    print("3: Both individual points and clusters")
    plot_mode_input = input("Enter choice (1/2/3): ").strip()
    if plot_mode_input == "1":
        plot_mode = "points"
    elif plot_mode_input == "2":
        plot_mode = "clusters"
    elif plot_mode_input == "3":
        plot_mode = "both"
    else:
        logging.warning("Invalid selection, defaulting to individual points mode.")
        plot_mode = "points"

    docs, category_colors = load_sample_embeddings(data_type, max_docs)
    categories = {doc.get("source", "unknown") for doc in docs}
    logging.info(
        "Loaded %d documents with categories: %s", len(docs), sorted(categories)
    )

    # Choose update mode.
    print("\nSelect update mode:")
    print("1: Interactive update (enter new docs dynamically)")
    print("2: Stream update (plot each doc second-by-second)")
    update_mode_input = input("Enter choice (1/2): ").strip()

    if update_mode_input == "2":
        stream_update(docs, category_colors, plot_mode)
    else:
        interactive_update(data_type, docs, category_colors, plot_mode)


if __name__ == "__main__":
    main()
