from typing import List, Dict
import re


# ============================================================
# STRATEGY 1 — FIXED-SIZE CHUNKING
# ============================================================

def fixed_size_chunking(
    documents: List[Dict],
    chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> List[Dict]:

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        start = 0
        text_length = len(text)

        while start < text_length:

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunk_metadata = metadata.copy()

                chunk_metadata["chunking_strategy"] = "fixed_size"

                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })

            start += chunk_size - chunk_overlap

    return chunks


# ============================================================
# STRATEGY 2 — SECTION-AWARE CHUNKING
# ============================================================

def section_aware_chunking(
    documents: List[Dict],
    max_chunk_size: int = 1500
) -> List[Dict]:

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        # Split using numbered headings
        sections = re.split(
            r"(?=\n?\d+\.\s+[A-Za-z])",
            text
        )

        for section in sections:

            section = section.strip()

            if not section:
                continue

            # If section is small enough, keep it together
            if len(section) <= max_chunk_size:

                chunk_metadata = metadata.copy()

                chunk_metadata["chunking_strategy"] = "section_aware"

                chunks.append({
                    "text": section,
                    "metadata": chunk_metadata
                })

            else:

                # Split large sections into smaller chunks
                start = 0

                while start < len(section):

                    end = start + max_chunk_size

                    chunk_text = section[start:end].strip()

                    if chunk_text:

                        chunk_metadata = metadata.copy()

                        chunk_metadata["chunking_strategy"] = "section_aware"

                        chunks.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        })

                    start += max_chunk_size

    return chunks


# ============================================================
# CHUNKING COMPARISON
# ============================================================

def compare_chunking_strategies(documents):

    fixed_chunks = fixed_size_chunking(documents)

    section_chunks = section_aware_chunking(documents)

    print("\n" + "=" * 60)
    print("CHUNKING COMPARISON")
    print("=" * 60)

    print(f"\nOriginal pages/documents : {len(documents)}")
    print(f"Fixed-size chunks        : {len(fixed_chunks)}")
    print(f"Section-aware chunks     : {len(section_chunks)}")

    return fixed_chunks, section_chunks