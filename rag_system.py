# rag_system.py
# Core RAG system — handles indexing, retrieval, and generation.

import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


class SimpleRAG:
    def __init__(self, collection_name: str = "knowledge_base"):
        # Connect to a local ChromaDB database (stored in ./rag_db folder)
        self.chroma_client = chromadb.PersistentClient(path="./rag_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        # Connect to Groq (the LLM that generates answers)
        self.llm_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def add_documents(self, documents: list[str], ids: list[str] = None):
        """PHASE 1: Indexing — add documents to the vector store."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids)
        print(f"Indexed {len(documents)} documents.")

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        """PHASE 2a: Retrieve relevant chunks from the vector store."""
        results = self.collection.query(query_texts=[question], n_results=top_k)
        return results["documents"][0]

    def generate_answer(self, question: str, top_k: int = 3) -> dict:
        """PHASE 2b + 2c: Augment the prompt with context, then generate."""
        # Step 1 — Retrieve
        retrieved_chunks = self.retrieve(question, top_k)
        context = "\n\n".join(
            [f"[Source {i+1}]: {c}" for i, c in enumerate(retrieved_chunks)]
        )

        # Step 2 — Augment (build the prompt)
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided context. If the context "
                    "doesn't have enough information, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        # Step 3 — Generate
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.1,  # low temp = factual, not creative
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": retrieved_chunks,
        }


# =====================================================================
# KNOWLEDGE BASE — GenAI Course Notes (Topics 1-8)
# =====================================================================

GENAI_COURSE_NOTES = [
    # --- Topic 1: Introduction to Generative AI ---
    "Generative AI refers to artificial intelligence systems that can create new content "
    "such as text, images, audio, and code, rather than simply analyzing or classifying "
    "existing data. Examples include ChatGPT, DALL-E, and Midjourney.",

    "Discriminative models learn the boundary between classes (e.g., spam vs not spam), "
    "while generative models learn the underlying distribution of the data so they can "
    "produce entirely new samples that resemble the training data.",

    "Foundation models are large AI models trained on broad, diverse datasets that can be "
    "adapted to many downstream tasks. GPT-4, LLaMA, and Claude are examples of "
    "foundation models for text. They represent a shift from task-specific to general-purpose AI.",

    "Key milestones in generative AI include the introduction of GANs in 2014 by Ian Goodfellow, "
    "the Transformer architecture in 2017 by Vaswani et al., GPT-1 in 2018, BERT in 2018, "
    "GPT-3 in 2020, ChatGPT in November 2022, and GPT-4 in March 2023.",

    # --- Topic 2: Large Language Models (LLMs) ---
    "Large Language Models (LLMs) are neural networks with billions of parameters trained on "
    "massive text corpora. They learn statistical patterns in language and can generate coherent, "
    "contextually relevant text. GPT-4 has an estimated 1.8 trillion parameters.",

    "LLMs are trained in two phases: pre-training, where the model learns general language "
    "patterns from vast unlabeled text using self-supervised learning, and fine-tuning, where "
    "the model is specialized on a narrower task or dataset with labeled examples.",

    "Tokenization is the process of converting raw text into numerical tokens that an LLM can "
    "process. Common methods include Byte-Pair Encoding (BPE), WordPiece, and SentencePiece. "
    "A single word may be split into multiple tokens depending on frequency.",

    "The context window of an LLM is the maximum number of tokens it can process in a single "
    "input-output cycle. GPT-3 had a 4K token context window, GPT-4 Turbo expanded to 128K, "
    "and newer models continue to push these limits.",

    "Inference is the process of running a trained model to generate predictions or outputs. "
    "LLM inference is computationally expensive because the model must process the full context "
    "and generate tokens one at a time in an autoregressive manner.",

    # --- Topic 3: Transformer Architecture ---
    "The Transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' "
    "by Vaswani et al., replaced recurrence with self-attention mechanisms, allowing the model "
    "to process all positions in a sequence in parallel rather than sequentially.",

    "Self-attention computes a weighted sum of all positions in a sequence for each position, "
    "allowing the model to capture dependencies regardless of distance. It uses three learned "
    "matrices — Query (Q), Key (K), and Value (V) — to compute attention scores.",

    "Multi-head attention runs several self-attention operations in parallel, each with different "
    "learned projections. This allows the model to attend to information from different "
    "representation subspaces at different positions simultaneously.",

    "Positional encoding is added to input embeddings to give the Transformer information about "
    "token order, since self-attention is permutation-invariant. The original Transformer used "
    "sinusoidal functions; modern models often use learned or rotary positional embeddings (RoPE).",

    "Encoder-only Transformers like BERT process input bidirectionally and are suited for "
    "classification, NER, and understanding tasks. Decoder-only Transformers like GPT process "
    "input left-to-right and are suited for text generation. Encoder-decoder models like T5 "
    "handle sequence-to-sequence tasks like translation and summarization.",

    # --- Topic 4: Embeddings and Vector Representations ---
    "Word embeddings are dense vector representations of words where semantically similar words "
    "are close together in vector space. Word2Vec, GloVe, and FastText are classic embedding "
    "methods. Modern LLMs produce contextual embeddings where the same word gets different "
    "vectors depending on surrounding context.",

    "Sentence embeddings represent an entire sentence or paragraph as a single fixed-length "
    "vector. Models like Sentence-BERT (SBERT) and OpenAI's text-embedding-ada-002 are "
    "specifically trained to produce high-quality sentence-level embeddings useful for "
    "semantic search and similarity comparison.",

    "Cosine similarity measures the angle between two vectors and is the standard metric for "
    "comparing embeddings. A cosine similarity of 1.0 means identical direction, 0 means "
    "orthogonal (unrelated), and -1 means opposite. It is preferred over Euclidean distance "
    "for high-dimensional text embeddings.",

    "Vector databases like ChromaDB, Pinecone, Weaviate, and FAISS are purpose-built to store "
    "and search high-dimensional embedding vectors efficiently. They use approximate nearest "
    "neighbor (ANN) algorithms like HNSW to find similar vectors without brute-force comparison.",

    # --- Topic 5: Prompt Engineering ---
    "Prompt engineering is the practice of designing input prompts to get desired outputs from "
    "LLMs. Effective prompts are specific, provide context, include examples, and specify the "
    "desired output format. It is often the fastest way to improve LLM outputs without any code.",

    "Zero-shot prompting gives the model a task with no examples. Few-shot prompting includes "
    "a few input-output examples in the prompt to guide the model. Few-shot generally produces "
    "better results for complex or ambiguous tasks, at the cost of using more tokens.",

    "Chain-of-Thought (CoT) prompting asks the model to show its reasoning step by step before "
    "giving a final answer. Adding 'Let's think step by step' can improve accuracy on math, "
    "logic, and multi-step reasoning tasks by 10-40% on common benchmarks.",

    "System prompts set the behavior, persona, or constraints for an LLM at the start of a "
    "conversation. For example, 'You are a helpful medical assistant. Always cite sources.' "
    "They persist across the conversation and shape every response the model produces.",

    "Temperature controls randomness in LLM outputs. Temperature 0 gives the most deterministic "
    "response (always picks the highest-probability token), while temperature 1.0 or higher "
    "produces more creative and varied responses. For factual Q&A, use low temperature (0.1-0.3).",

    # --- Topic 6: Fine-Tuning ---
    "Fine-tuning adapts a pre-trained model to a specific task or domain by continuing training "
    "on a smaller, task-specific dataset. Full fine-tuning updates all model parameters, which "
    "is expensive for large models. It can dramatically improve performance on specialized tasks.",

    "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes the "
    "original model weights and injects small trainable matrices (adapters) into each layer. "
    "This reduces the number of trainable parameters by 90-99%, making fine-tuning feasible "
    "on consumer GPUs.",

    "QLoRA combines LoRA with 4-bit quantization, further reducing memory requirements. It "
    "allows fine-tuning a 65-billion parameter model on a single 48GB GPU. The base model is "
    "loaded in 4-bit precision while the LoRA adapters are trained in higher precision.",

    "RLHF (Reinforcement Learning from Human Feedback) is a fine-tuning technique where human "
    "raters rank model outputs, a reward model is trained on those rankings, and the LLM is "
    "optimized using PPO to maximize the reward. ChatGPT used RLHF to improve helpfulness "
    "and safety.",

    "Instruction tuning fine-tunes a base model on a dataset of (instruction, response) pairs "
    "to make it follow instructions better. Models like Alpaca, Vicuna, and LLaMA-2-Chat are "
    "instruction-tuned versions of their base models.",

    # --- Topic 7: Retrieval-Augmented Generation (RAG) ---
    "RAG (Retrieval-Augmented Generation) combines information retrieval with LLM generation. "
    "Instead of relying only on what the model memorized during training, RAG retrieves relevant "
    "documents from an external knowledge base and includes them in the prompt so the model "
    "can generate answers grounded in up-to-date, specific information.",

    "The RAG pipeline has two main phases: indexing (done once) and querying (done per question). "
    "Indexing involves chunking documents, generating embeddings, and storing them in a vector "
    "database. Querying involves embedding the question, retrieving similar chunks, and passing "
    "them as context to the LLM.",

    "Document chunking splits large documents into smaller pieces before embedding. Common "
    "strategies include fixed-size chunks (e.g., 500 tokens with 50-token overlap), sentence-based "
    "splitting, and recursive character splitting. Chunk size affects retrieval precision: too "
    "large misses specifics, too small loses context.",

    "In RAG, the retrieval step uses semantic search — the user's question is embedded into a "
    "vector, and the vector database returns the most similar document chunks. This is more "
    "powerful than keyword search because it matches meaning, not just exact words.",

    "RAG reduces hallucination because the LLM is instructed to answer based only on retrieved "
    "context, not its parametric memory. It also solves the knowledge cutoff problem since the "
    "external knowledge base can be updated without retraining the model.",

    "Advanced RAG techniques include hybrid search (combining semantic and keyword search), "
    "re-ranking retrieved results for relevance, query expansion to improve recall, and "
    "multi-step retrieval where the model asks follow-up queries to gather more context.",

    # --- Topic 8: Evaluation and Deployment ---
    "LLM evaluation metrics include BLEU and ROUGE for comparing generated text to references, "
    "perplexity for measuring model confidence, and human evaluation for quality, helpfulness, "
    "and safety. For RAG systems, faithfulness (does the answer match the sources?) and "
    "relevance (are the retrieved documents useful?) are key metrics.",

    "Hallucination in LLMs occurs when the model generates plausible-sounding but factually "
    "incorrect information. Mitigation strategies include RAG, constrained decoding, lower "
    "temperature settings, and prompt instructions like 'only answer based on the given context'.",

    "FastAPI is a modern Python web framework used to wrap ML models as REST APIs. It provides "
    "automatic interactive documentation at /docs (Swagger UI), request validation using Pydantic "
    "models, and async support. It is the standard way to deploy LLM-based systems as services.",

    "Model deployment considerations include latency (time to generate a response), throughput "
    "(requests per second), cost (API calls or GPU hours), and scalability. Using hosted APIs "
    "like Groq or OpenAI offloads infrastructure, while self-hosting gives more control but "
    "requires GPU management.",
]


if __name__ == "__main__":
    print("Creating RAG system and indexing GenAI course notes...")
    rag = SimpleRAG()

    # Index all notes with unique IDs
    ids = [f"note_{i}" for i in range(len(GENAI_COURSE_NOTES))]
    rag.add_documents(GENAI_COURSE_NOTES, ids=ids)

    print(f"\nDone! Indexed {len(GENAI_COURSE_NOTES)} notes.")
    print("\nTesting with a sample question...")

    result = rag.generate_answer("What is the difference between BERT and GPT?")
    print(f"\nQuestion: What is the difference between BERT and GPT?")
    print(f"Answer: {result['answer']}")
    print(f"\nSources used:")
    for src in result["sources"]:
        print(f"  - {src[:100]}...")
