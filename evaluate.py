# evaluate.py
# Measures accuracy of the RAG system against a small, known test set.
# Run with:  python3 evaluate.py
#
# This checks two separate things:
#   1. Retrieval accuracy — did the vector database find the correct source note?
#   2. Answer accuracy    — does the generated answer contain the expected fact?

from rag_system import SimpleRAG

# One test case per course topic (1-8).
# expected_source_keyword: a phrase that should appear in the CORRECT source note
# expected_answer_keyword: a phrase that should appear in a CORRECT generated answer
TEST_SET = [
    {
        "question": "What is the difference between discriminative and generative models?",
        "expected_source_keyword": "Discriminative models",
        "expected_answer_keyword": "distribution",
    },
    {
        "question": "How many parameters does GPT-4 have?",
        "expected_source_keyword": "1.8 trillion",
        "expected_answer_keyword": "1.8 trillion",
    },
    {
        "question": "What three matrices does self-attention use?",
        "expected_source_keyword": "Query",
        "expected_answer_keyword": "Query",
    },
    {
        "question": "What metric is used to compare embeddings?",
        "expected_source_keyword": "Cosine similarity",
        "expected_answer_keyword": "cosine",
    },
    {
        "question": "What is Chain-of-Thought prompting?",
        "expected_source_keyword": "Chain-of-Thought",
        "expected_answer_keyword": "step by step",
    },
    {
        "question": "What is QLoRA?",
        "expected_source_keyword": "QLoRA",
        "expected_answer_keyword": "4-bit",
    },
    {
        "question": "What are the two main phases of the RAG pipeline?",
        "expected_source_keyword": "indexing",
        "expected_answer_keyword": "indexing",
    },
    {
        "question": "What framework is used to deploy the RAG system as an API?",
        "expected_source_keyword": "FastAPI",
        "expected_answer_keyword": "FastAPI",
    },
]


def evaluate():
    rag = SimpleRAG()
    retrieval_hits = 0
    answer_hits = 0
    results = []

    for case in TEST_SET:
        result = rag.generate_answer(case["question"], top_k=3)
        sources = result["sources"]
        answer = result["answer"]

        retrieved_correctly = any(
            case["expected_source_keyword"].lower() in s.lower() for s in sources
        )
        answered_correctly = (
            case["expected_answer_keyword"].lower() in answer.lower()
        )

        if retrieved_correctly:
            retrieval_hits += 1
        if answered_correctly:
            answer_hits += 1

        results.append(
            {
                "question": case["question"],
                "retrieved_correctly": retrieved_correctly,
                "answered_correctly": answered_correctly,
                "answer": answer,
            }
        )

    total = len(TEST_SET)
    print("=" * 60)
    print(f"Retrieval accuracy: {retrieval_hits}/{total} ({retrieval_hits/total*100:.0f}%)")
    print(f"Answer accuracy:    {answer_hits}/{total} ({answer_hits/total*100:.0f}%)")
    print("=" * 60)
    print()

    for r in results:
        r_mark = "PASS" if r["retrieved_correctly"] else "FAIL"
        a_mark = "PASS" if r["answered_correctly"] else "FAIL"
        print(f"[retrieval: {r_mark} | answer: {a_mark}] {r['question']}")
        print(f"    -> {r['answer'][:120]}...")
        print()


if __name__ == "__main__":
    evaluate()
