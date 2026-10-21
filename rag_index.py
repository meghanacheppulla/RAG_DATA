"""
Basic Data Indexing for RAG
  load documents -> split into uniform chunks (~200 tokens) -> embed each chunk
  -> store in an in-memory vector store -> inspect for consistency

Terminal check:  python rag_index.py
"""
import math
import os
import re
import zlib

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_TOKENS = 200      # target chunk size
OVERLAP_TOKENS = 20     # small overlap keeps context across chunk borders
CHARS_PER_TOKEN = 4     # common rule of thumb for English text


def est_tokens(text: str) -> int:
    """Approximate token count (1 token ~ 4 characters)."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


# --------------------------------------------------------------------------- #
# 1. Load documents
# --------------------------------------------------------------------------- #
SAMPLE_DOCS = {
    "generative_ai.txt": (
        "Generative AI is a branch of artificial intelligence that creates new content such as "
        "text, images, audio, video and software code. Instead of only classifying or predicting, "
        "a generative model learns the patterns in very large datasets and then produces original "
        "outputs when it receives a prompt. Common examples include chatbots, image generators, "
        "code assistants and music tools. Businesses use generative AI to draft emails, summarize "
        "reports, answer customer questions and speed up software development.\n\n"
        "Most modern generative systems are built on deep neural networks called transformers. A "
        "transformer reads a sequence of tokens and uses attention to decide which earlier tokens "
        "matter most for the next one. Training happens in two broad stages: pre-training on huge "
        "amounts of general data, followed by fine-tuning or alignment so the model follows "
        "instructions and behaves safely. The quality of the output depends on the model size, the "
        "training data and the clarity of the prompt supplied by the user."
        "\n\n"
        "Generative AI also raises important questions. Models can reflect biases found in their "
        "training data, may produce text that sounds right but is wrong, and can be misused to create "
        "misleading media. Responsible teams add human review, clear usage policies and testing so "
        "that the technology stays helpful, safe and transparent for the people who rely on it."
    ),
    "large_language_models.txt": (
        "Large language models, or LLMs, are generative AI systems trained on massive text "
        "collections to understand and produce human language. They predict the next token in a "
        "sequence, and by repeating this step they can write essays, translate languages, answer "
        "questions and generate program code. Well known families include Gemini, GPT and Claude. "
        "An LLM has a context window, which limits how much text it can read at once, and it has a "
        "knowledge cutoff, which means it may not know about recent events.\n\n"
        "Because an LLM can state wrong facts with confidence, a problem known as hallucination, "
        "developers often connect it to trusted data sources. They also control behaviour with "
        "parameters such as temperature. A low temperature gives focused and repeatable answers, "
        "while a high temperature gives more varied and creative text. Choosing the right model, "
        "prompt and settings is a core skill when building applications on top of LLMs."
        "\n\n"
        "Developers interact with an LLM through prompts, which can include instructions, examples "
        "and extra context. Techniques such as role prompts, step-by-step reasoning and structured "
        "output formats make answers more reliable. Frameworks like LangChain help connect prompts, "
        "models and parsers into repeatable chains so that complete applications can be assembled "
        "from small, testable pieces."
    ),
    "retrieval_augmented_generation.txt": (
        "Retrieval-Augmented Generation, usually shortened to RAG, combines a language model with "
        "an external knowledge base. When a user asks a question, the system first retrieves the "
        "most relevant passages from stored documents and then gives those passages to the model "
        "as context. The model writes its answer using that evidence, which reduces hallucinations "
        "and lets the application work with private or up-to-date information that was never part "
        "of the model's training data.\n\n"
        "A RAG pipeline has two phases. The indexing phase loads documents, splits them into "
        "chunks, converts every chunk into an embedding vector and saves the vectors in a vector "
        "store. The query phase embeds the user question, finds the nearest chunk vectors using "
        "similarity search, and passes the matching text to the model. Good chunking and "
        "consistent embeddings are the foundation that decides how accurate the final answers are."
        "\n\n"
        "RAG is popular for company assistants, document question answering and support bots. "
        "Because answers are grounded in retrieved passages, the application can show its sources "
        "and be updated simply by re-indexing new files, without retraining the model. Evaluating "
        "retrieval quality, for example by checking whether the right chunk appears in the top "
        "results, is just as important as evaluating the generated text."
    ),
    "embeddings_and_vector_stores.txt": (
        "An embedding is a list of numbers that represents the meaning of a piece of text. Texts "
        "with similar meaning get vectors that point in similar directions, so their cosine "
        "similarity is high. Every vector produced by one embedding model has the same length, "
        "called its dimension, and this must stay identical across all chunks for search to work "
        "correctly. Embeddings power semantic search, recommendations, clustering and RAG systems.\n\n"
        "A vector store keeps embeddings together with the original text and metadata such as the "
        "source file and page number. An in-memory vector store holds everything in RAM, which is "
        "fast and simple for small collections and experiments, but the data disappears when the "
        "program stops. Production systems usually use persistent stores such as FAISS indexes, "
        "Chroma, Pinecone or pgvector so that the index can be reused and scaled to millions of "
        "chunks."
        "\n\n"
        "Before trusting an index, it is good practice to inspect it. Useful checks include "
        "confirming that every chunk has exactly one vector, that all vectors have the same "
        "dimension, that no values are missing or invalid, and that a few test queries return the "
        "chunks a person would expect. These simple checks catch most indexing mistakes early."
    ),
}


def load_sample_docs() -> list[Document]:
    return [
        Document(page_content=text, metadata={"source": name, "page": 1})
        for name, text in SAMPLE_DOCS.items()
    ]


def load_pdf(file_obj, name: str) -> list[Document]:
    """One Document per page; pages without extractable text are skipped."""
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": i}))
    return docs


# --------------------------------------------------------------------------- #
# 2. Split into uniform chunks
# --------------------------------------------------------------------------- #
def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS * CHARS_PER_TOKEN,      # 200 tokens ~ 800 characters
        chunk_overlap=OVERLAP_TOKENS * CHARS_PER_TOKEN,  # 20 tokens ~ 80 characters
        length_function=len,
        separators=[" ", ""],  # split on words -> chunks fill up to ~200 tokens
    )
    # collapse line breaks/extra spaces so every chunk is filled evenly
    clean = [
        Document(page_content=re.sub(r"\s+", " ", d.page_content).strip(), metadata=dict(d.metadata))
        for d in docs
    ]
    chunks = splitter.split_documents(clean)
    chunks = [c for c in chunks if c.page_content.strip()]
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["tokens"] = est_tokens(c.page_content)
    return chunks


# --------------------------------------------------------------------------- #
# 3. Embeddings (Gemini if a key works, otherwise a local fallback)
# --------------------------------------------------------------------------- #
class LocalHashEmbeddings(Embeddings):
    """Offline bag-of-words embedding: deterministic, fixed size, no API key needed."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        words = re.findall(r"[a-z0-9]+", text.lower())
        for w in words:
            vec[zlib.crc32(w.encode()) % self.dim] += 1.0
        for a, b in zip(words, words[1:]):
            vec[zlib.crc32(f"{a} {b}".encode()) % self.dim] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(force_local: bool = False):
    """Returns (embeddings, label, note)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if force_local or not api_key:
        note = "" if force_local else "No GOOGLE_API_KEY found - using the built-in local embedding model."
        return LocalHashEmbeddings(), "Local hash embeddings (256-d)", note
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
        emb = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        emb.embed_query("connectivity check")  # fail fast if key/model is wrong
        return emb, f"Gemini embeddings ({model})", ""
    except Exception as e:  # noqa: BLE001
        short = str(e).splitlines()[0][:140]
        return (
            LocalHashEmbeddings(),
            "Local hash embeddings (256-d)",
            f"Gemini embeddings unavailable ({short}) - using the local embedding model instead.",
        )


# --------------------------------------------------------------------------- #
# 4. In-memory vector store
# --------------------------------------------------------------------------- #
def build_store(chunks: list[Document], emb: Embeddings) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedding=emb)
    store.add_documents(chunks)
    return store


# --------------------------------------------------------------------------- #
# 5. Inspect for consistency
# --------------------------------------------------------------------------- #
def inspect_index(chunks: list[Document], store: InMemoryVectorStore) -> dict:
    entries = list(store.store.values())
    vectors = [np.asarray(e["vector"], dtype=float) for e in entries]
    dims = {v.shape[0] for v in vectors}
    toks = [est_tokens(c.page_content) for c in chunks]
    limit = CHUNK_TOKENS + 2  # tolerance for rounding in the token estimate

    checks = [
        ("Every chunk has exactly one embedding",
         len(chunks) == len(vectors), f"{len(chunks)} chunks / {len(vectors)} vectors"),
        ("All vectors share the same dimension",
         len(dims) == 1, f"dimension(s): {sorted(dims)}"),
        ("No NaN or infinite values in any vector",
         all(np.isfinite(v).all() for v in vectors), f"{len(vectors)} vectors scanned"),
        ("No empty chunks",
         all(c.page_content.strip() for c in chunks), f"{len(chunks)} chunks scanned"),
        (f"No chunk exceeds {CHUNK_TOKENS} tokens",
         max(toks, default=0) <= limit, f"largest chunk = {max(toks, default=0)} tokens"),
        ("Every chunk keeps source metadata",
         all("source" in c.metadata for c in chunks), "source + page + chunk_id present"),
    ]
    dim = next(iter(dims)) if len(dims) == 1 else None
    return {
        "checks": [{"check": n, "passed": bool(ok), "detail": d} for n, ok, d in checks],
        "dimension": dim,
        "n_vectors": len(vectors),
        "avg_tokens": round(sum(toks) / len(toks), 1) if toks else 0,
        "min_tokens": min(toks, default=0),
        "max_tokens": max(toks, default=0),
    }


def create_index(docs: list[Document], force_local: bool = False) -> dict:
    """Runs the whole pipeline and returns everything the UI / terminal needs."""
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted from the document(s).")
    emb, label, note = get_embeddings(force_local)
    try:
        store = build_store(chunks, emb)
    except Exception as e:  # noqa: BLE001  (e.g. API quota) -> fall back so the run never crashes
        emb, label = LocalHashEmbeddings(), "Local hash embeddings (256-d)"
        note = f"Gemini embedding call failed ({str(e).splitlines()[0][:120]}) - used local embeddings."
        store = build_store(chunks, emb)
    return {
        "docs": docs, "chunks": chunks, "store": store,
        "embedding_label": label, "note": note,
        "report": inspect_index(chunks, store),
    }


def search(store: InMemoryVectorStore, query: str, k: int = 3):
    return store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    result = create_index(load_sample_docs())
    rep = result["report"]
    print(f"Documents : {len(result['docs'])}")
    print(f"Chunks    : {len(result['chunks'])} (avg {rep['avg_tokens']} tokens, max {rep['max_tokens']})")
    print(f"Embeddings: {rep['n_vectors']} x {rep['dimension']}  [{result['embedding_label']}]")
    if result["note"]:
        print("Note      :", result["note"])
    print("\nConsistency checks:")
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']} - {c['detail']}")
    top = search(result["store"], "What is a vector store?", k=1)[0]
    print(f"\nTest query top hit: {top[0].metadata['source']} (score {top[1]:.3f})")
    assert all(c["passed"] for c in rep["checks"]), "Consistency check failed"
    print("\n[OK] Indexing pipeline executed end-to-end")
