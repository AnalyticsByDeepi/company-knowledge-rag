from pathlib import Path
import pymupdf
import re


# ============================================================
# FIND PROJECT DOCUMENTS FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section(text):

    lines = text.splitlines()

    for line in lines:
        line = line.strip()

        if re.match(r"^\d+\.\s+[A-Za-z]", line):
            return line

    return "General"


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf(file_path):

    documents = []

    pdf = pymupdf.open(file_path)

    document_name = file_path.stem.replace("_", " ").title()

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text("text").strip()

        if not text:
            continue

        section = detect_section(text)

        document = {
            "text": text,

            "metadata": {
                "document_name": document_name,
                "document_type": "pdf",
                "page_number": page_number,
                "section": section,
                "source_file": file_path.name
            }
        }

        documents.append(document)

    pdf.close()

    return documents


# ============================================================
# LOAD ALL PDF DOCUMENTS
# ============================================================

def load_all_documents():

    all_documents = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    print("Documents folder:")
    print(DOCUMENTS_DIR)

    print(f"\nFound {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        pages = load_pdf(pdf_file)

        all_documents.extend(pages)

        print(f"  → {len(pages)} pages loaded")

    return all_documents


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    documents = load_all_documents()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print(f"Total pages loaded: {len(documents)}")

    if documents:

        print("\nExample metadata:")
        print("-" * 60)
        print(documents[0]["metadata"])

        print("\nText preview:")
        print(documents[0]["text"][:500])