from pathlib import Path
import faiss
import numpy as np
import pickle


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_STORE_DIR = BASE_DIR / "vector_store"

INDEX_FILE = VECTOR_STORE_DIR / "company_policies.index"
METADATA_FILE = VECTOR_STORE_DIR / "company_policies.pkl"


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):

    print("=" * 60)
    print("CREATING FAISS INDEX")
    print("=" * 60)

    # Convert embeddings to float32
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Get embedding dimension
    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")
    print(f"Number of vectors: {len(embeddings)}")

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings)

    print("✅ FAISS index created!")
    print(f"Total vectors stored: {index.ntotal}")

    return index


# ============================================================
# SAVE FAISS INDEX
# ============================================================

def save_faiss_index(index, chunks):

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save chunks + metadata
    with open(
        METADATA_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("\n" + "=" * 60)
    print("FAISS VECTOR STORE SAVED")
    print("=" * 60)

    print(f"Index file    : {INDEX_FILE}")
    print(f"Metadata file : {METADATA_FILE}")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_faiss_index():

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "rb"
    ) as file:

        chunks = pickle.load(file)

    print("✅ FAISS index loaded!")
    print(f"Vectors: {index.ntotal}")
    print(f"Chunks : {len(chunks)}")

    return index, chunks