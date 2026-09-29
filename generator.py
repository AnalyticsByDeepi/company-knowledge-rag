import os
from groq import Groq


# ============================================================
# GROQ CONFIGURATION
# ============================================================

DEFAULT_MODEL = "qwen/qwen3.8-27b"


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

def create_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file."
        )

    client = Groq(
        api_key=api_key
    )

    return client


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_documents,
    client,
    model=DEFAULT_MODEL
):

    from prompts import build_prompt

    prompt = build_prompt(
        question,
        retrieved_documents
    )

    response = client.chat.completions.create(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.0
    )

    answer = response.choices[0].message.content

    return answer