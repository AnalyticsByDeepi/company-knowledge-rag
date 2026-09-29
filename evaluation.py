import pandas as pd


# ============================================================
# EVALUATION DATASET
# ============================================================

EVALUATION_QUESTIONS = [
    {
        "question": "How much can an employee claim for accommodation during business travel?",
        "expected_keywords": ["5,000", "3,500"],
        "expected_document": "Travel Policy"
    },
    {
        "question": "How much can an employee claim for food during business travel?",
        "expected_keywords": ["1,000"],
        "expected_document": "Travel Policy"
    },
    {
        "question": "What is the local transportation reimbursement limit?",
        "expected_keywords": ["800"],
        "expected_document": "Travel Policy"
    },
    {
        "question": "What is the medical insurance sum assured?",
        "expected_keywords": ["3,00,000"],
        "expected_document": "Medical Policy"
    },
    {
        "question": "What is the OPD reimbursement limit?",
        "expected_keywords": ["10,000"],
        "expected_document": "Medical Policy"
    },
    {
        "question": "What is the annual leave entitlement for employees?",
        "expected_keywords": ["annual"],
        "expected_document": "Leave Policy"
    },
    {
        "question": "What should an employee do when taking leave?",
        "expected_keywords": ["leave"],
        "expected_document": "Leave Policy"
    },
    {
        "question": "What should an employee do if their company laptop is lost?",
        "expected_keywords": ["report", "immediately"],
        "expected_document": "IT Policy"
    },
    {
        "question": "How should a suspected IT security incident be reported?",
        "expected_keywords": ["IT", "Security"],
        "expected_document": "IT Policy"
    },
    {
        "question": "What are the expectations for employee conduct?",
        "expected_keywords": ["conduct"],
        "expected_document": "Code Of Conduct"
    },
    {
        "question": "What should employees do with confidential company information?",
        "expected_keywords": ["confidential"],
        "expected_document": "Code Of Conduct"
    },
    {
        "question": "What information is covered by the employee handbook?",
        "expected_keywords": ["employee"],
        "expected_document": "Employee Handbook"
    },
    {
        "question": "What are the general responsibilities of employees?",
        "expected_keywords": ["employee"],
        "expected_document": "Employee Handbook"
    },
    {
        "question": "What are the rules for appropriate use of company IT resources?",
        "expected_keywords": ["IT"],
        "expected_document": "IT Policy"
    },
    {
        "question": "What approval is required for expenses exceeding travel limits?",
        "expected_keywords": ["approval"],
        "expected_document": "Travel Policy"
    }
]


# ============================================================
# KEYWORD EVALUATION
# ============================================================

def evaluate_answer(answer, expected_keywords):

    answer_lower = answer.lower()

    matched_keywords = []

    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:
            matched_keywords.append(keyword)

    score = (
        len(matched_keywords) /
        len(expected_keywords)
    )

    return score, matched_keywords


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation(
    embedding_model,
    faiss_index,
    chunks,
    groq_client
):

    evaluation_results = []

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    for number, item in enumerate(
        EVALUATION_QUESTIONS,
        start=1
    ):

        question = item["question"]

        print(f"\nQuestion {number}: {question}")

        # Retrieve
        from retriever import retrieve_documents

        results = retrieve_documents(
            query=question,
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            chunks=chunks,
            top_k=5
        )

        # Generate
        from generator import generate_answer

        answer = generate_answer(
            question=question,
            retrieved_documents=results,
            client=groq_client
        )

        # Evaluate
        score, matched = evaluate_answer(
            answer,
            item["expected_keywords"]
        )

        retrieved_documents = [
            result["metadata"].get("document_name")
            for result in results
        ]

        expected_document = item["expected_document"]

        document_retrieved = any(
            expected_document.lower()
            in document.lower()
            for document in retrieved_documents
        )

        evaluation_results.append({
            "question": question,
            "expected_document": expected_document,
            "document_retrieved": document_retrieved,
            "keyword_score": score,
            "matched_keywords": ", ".join(matched),
            "answer": answer
        })

        print(
            f"Keyword score: {score:.2f}"
        )

        print(
            f"Expected document retrieved: "
            f"{document_retrieved}"
        )

    return pd.DataFrame(evaluation_results)


# ============================================================
# EVALUATION SUMMARY
# ============================================================

def print_evaluation_summary(results_df):

    keyword_accuracy = (
        results_df["keyword_score"].mean()
        * 100
    )

    retrieval_accuracy = (
        results_df["document_retrieved"].mean()
        * 100
    )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total questions       : {len(results_df)}"
    )

    print(
        f"Retrieval accuracy    : "
        f"{retrieval_accuracy:.2f}%"
    )

    print(
        f"Keyword answer score  : "
        f"{keyword_accuracy:.2f}%"
    )

    return {
        "retrieval_accuracy": retrieval_accuracy,
        "keyword_accuracy": keyword_accuracy
    }