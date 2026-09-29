import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_DIR / ".env")


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from embeddings import load_embedding_model
from vector_store import load_faiss_index
from retriever import retrieve_documents
from generator import create_groq_client, generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="🏢",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏢 Enterprise Company Knowledge Assistant")

st.write(
    "Ask questions about company policies, employee benefits, "
    "travel, leave, IT, medical insurance, and workplace conduct."
)


# ============================================================
# LOAD RESOURCES
# ============================================================

@st.cache_resource
def load_resources():

    embedding_model = load_embedding_model()

    faiss_index, chunks = load_faiss_index()

    groq_client = create_groq_client()

    return (
        embedding_model,
        faiss_index,
        chunks,
        groq_client
    )


try:

    (
        embedding_model,
        faiss_index,
        chunks,
        groq_client
    ) = load_resources()

except Exception as e:

    st.error(f"Failed to load the RAG system: {e}")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.write(
        f"Documents indexed: "
        f"{len(set(
            chunk['metadata']['document_name']
            for chunk in chunks
        ))}"
    )

    st.write(
        f"Chunks indexed: {len(chunks)}"
    )

    st.divider()

    st.write("**Available policies:**")

    documents = sorted(
        set(
            chunk["metadata"]["document_name"]
            for chunk in chunks
        )
    )

    for document in documents:

        st.write(f"• {document}")


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a company policy question:",
    placeholder="Example: How many days of annual leave can employees take?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🔍 Ask Assistant"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching company policies..."
        ):

            results = retrieve_documents(
                query=question,
                embedding_model=embedding_model,
                faiss_index=faiss_index,
                chunks=chunks,
                top_k=5
            )

        if not results:

            st.warning(
                "I couldn't find relevant information "
                "in the company policies."
            )

        else:

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question=question,
                    retrieved_documents=results,
                    client=groq_client
                )

            # ==================================================
            # ANSWER
            # ==================================================

            st.subheader("💬 Answer")

            st.write(answer)

            # ==================================================
            # RETRIEVED SOURCES
            # ==================================================

            st.subheader("📚 Retrieved Sources")

            for i, result in enumerate(
                results,
                start=1
            ):

                metadata = result["metadata"]

                with st.expander(
                    f"Source {i}: "
                    f"{metadata.get('document_name')}"
                ):

                    st.write(
                        f"**Page:** "
                        f"{metadata.get('page_number')}"
                    )

                    st.write(
                        f"**Section:** "
                        f"{metadata.get('section')}"
                    )

                    st.write(
                        f"**Source file:** "
                        f"{metadata.get('source_file')}"
                    )

                    st.write(
                        "**Retrieved text:**"
                    )

                    st.write(
                        result["text"]
                    )