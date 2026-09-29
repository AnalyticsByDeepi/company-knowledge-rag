from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("✅ Embedding model loaded successfully!")

    return model


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(chunks, model):

    print("\n" + "=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)

    # Extract text from chunks
    texts = [chunk["text"] for chunk in chunks]

    print(f"Number of chunks: {len(texts)}")

    # Generate embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("✅ Embeddings generated successfully!")

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    return embeddings


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Embedding module loaded successfully.")

    print(f"Model: {MODEL_NAME}")

    print("Expected embedding dimension: 384")