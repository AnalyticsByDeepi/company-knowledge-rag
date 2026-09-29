import numpy as np


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    embedding_model,
    faiss_index,
    chunks,
    top_k=5,
    max_distance=1.20
):
    """
    Retrieve relevant company policy documents using FAISS.

    Parameters:
        query: User's question
        embedding_model: SentenceTransformer embedding model
        faiss_index: FAISS vector index
        chunks: Stored document chunks and metadata
        top_k: Number of results to retrieve
        max_distance: Maximum FAISS distance allowed

    Returns:
        List of relevant retrieved documents
    """

    # --------------------------------------------------------
    # Create embedding for user's query
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # Make sure FAISS receives float32 values
    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    # --------------------------------------------------------
    # Store relevant results
    # --------------------------------------------------------

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        # Ignore invalid FAISS index
        if index == -1:
            continue

        # Ignore results that are too far from the query
        if float(distance) > max_distance:
            continue

        results.append({
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            "distance": float(distance)
        })

    return results


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

def display_retrieval_results(results):
    """
    Display retrieved documents in the terminal.
    Useful for testing retrieval separately from Streamlit.
    """

    print(
        f"\nRetrieved results: {len(results)}"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Document: "
            f"{metadata.get('document_name')}"
        )

        print(
            f"Page: "
            f"{metadata.get('page_number')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section')}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            f"Text: "
            f"{result['text'][:500]}..."
        )