# ============================================================
# rag_engine.py — Core RAG logic (retrieval + generation)
# Generator: OpenAI-compatible Responses API (OpenAI or Groq)
# ============================================================

import os

import numpy as np
from Bio import Entrez
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Config ────────────────────────────────────────────────────
ENTREZ_EMAIL     = os.getenv("ENTREZ_EMAIL", "your@email.com")
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "groq").lower()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL    = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 75
DEFAULT_K        = 4
MAX_TOKENS       = 512

# ── Singletons (loaded once at startup) ───────────────────────
vectorizer: TfidfVectorizer | None = None
faiss_index = None
chunks: list[Document]             = []
openai_client: OpenAI | None       = None


class CosineSearchIndex:
    """Small wrapper that mimics the FAISS fields the API exposes."""

    def __init__(self, matrix: sparse.csr_matrix):
        self.matrix = matrix
        self.ntotal = matrix.shape[0]
        self.d = matrix.shape[1]

    def search(self, query_matrix: sparse.csr_matrix, k: int):
        scores = (query_matrix @ self.matrix.T).toarray().astype(np.float32)
        top_indices = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        top_scores = np.take_along_axis(scores, top_indices, axis=1)
        return top_scores, top_indices


# ── Initialisation ────────────────────────────────────────────

def init_engine(query: str = "guselkumab Tremfya psoriasis clinical trial",
                max_results: int = 60):
    """
    Call once at app startup.
    Fetches PubMed abstracts → chunks → vectorises → builds search index.
    """
    global vectorizer, faiss_index, chunks

    print(f"[init] Fetching PubMed abstracts for: '{query}'")
    raw_docs = _fetch_pubmed_abstracts(query, max_results)
    if not raw_docs:
        raise RuntimeError("PubMed returned no abstracts; cannot build the RAG index.")

    print(f"[init] Chunking {len(raw_docs)} abstracts...")
    chunks = _chunk_documents(raw_docs)
    if not chunks:
        raise RuntimeError("Chunking produced no documents; cannot build the RAG index.")

    print(f"[init] Vectorising {len(chunks)} chunks...")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    chunk_matrix = vectorizer.fit_transform([c.page_content for c in chunks]).tocsr()

    print("[init] Building cosine-similarity index...")
    faiss_index = CosineSearchIndex(chunk_matrix)

    print(f"[init] ✅ Ready — {faiss_index.ntotal} vectors indexed.")


# ── Retrieval ─────────────────────────────────────────────────

def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Embed query → cosine search → return top-k chunks."""
    if vectorizer is None or faiss_index is None:
        raise RuntimeError("RAG engine is not initialised.")

    k = min(k, len(chunks))
    if k == 0:
        raise RuntimeError("RAG index is empty.")

    query_matrix = vectorizer.transform([query]).tocsr()
    scores, indices = faiss_index.search(query_matrix, k)

    return [
        {
            "score":    float(scores[0][i]),
            "content":  chunks[indices[0][i]].page_content,
            "metadata": chunks[indices[0][i]].metadata,
        }
        for i in range(k)
    ]


# ── Generation ────────────────────────────────────────────────
def generate(query: str, context_chunks: list[dict]) -> str:
    """Build prompt from retrieved context → call the configured LLM → return answer."""

    context_str = "\n\n---\n\n".join([
        f"[Source {i+1} | PMID {c['metadata'].get('pmid', '?')} | {c['metadata'].get('year', '?')}]\n"
        f"{c['content']}"
        for i, c in enumerate(context_chunks)
    ])

    system_prompt = (
        "You are a medical information assistant specialising in dermatology "
        "and immunology treatments.\n"
        "Answer the question using ONLY the provided PubMed context below.\n"
        "If the context is insufficient, say so explicitly. Do not invent facts.\n"
        "Be concise but complete. Cite source numbers (e.g. [Source 1]) inline."
    )

    user_prompt = (
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER:"
    )

    provider = _get_provider_config()

    client = OpenAI(
        api_key=provider["api_key"],
        base_url=provider["base_url"],
    )

    response = client.chat.completions.create(
        model=provider["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=MAX_TOKENS,
    )

    return response.choices[0].message.content.strip()


# ── Full RAG pipeline ─────────────────────────────────────────

def rag_query(query: str, k: int = DEFAULT_K) -> dict:
    """
    End-to-end: retrieve → generate → return structured result.

    Returns:
        {
            "answer":  str,
            "sources": [{"pmid", "title", "year", "score"}, ...]
        }
    """
    context = retrieve(query, k=k)
    answer  = generate(query, context)

    sources = [
        {
            "pmid":  c["metadata"]["pmid"],
            "title": c["metadata"]["title"],
            "year":  c["metadata"]["year"],
            "score": round(c["score"], 4),
        }
        for c in context
    ]

    return {"answer": answer, "sources": sources}


# ── Private helpers ───────────────────────────────────────────

def _fetch_pubmed_abstracts(query: str, max_results: int) -> list[dict]:
    Entrez.email = ENTREZ_EMAIL

    search_handle = Entrez.esearch(
        db="pubmed", term=query, retmax=max_results, sort="relevance"
    )
    pmids = Entrez.read(search_handle)["IdList"]
    search_handle.close()

    fetch_handle = Entrez.efetch(
        db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml"
    )
    records = Entrez.read(fetch_handle)
    fetch_handle.close()

    documents = []
    for record in records["PubmedArticle"]:
        try:
            article  = record["MedlineCitation"]["Article"]
            title    = str(article.get("ArticleTitle", ""))
            abs_list = article.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(a) for a in abs_list)
            if not abstract.strip():
                continue
            year = str(
                article.get("Journal", {})
                       .get("JournalIssue", {})
                       .get("PubDate", {})
                       .get("Year", "Unknown")
            )
            documents.append({
                "title":    title,
                "abstract": abstract,
                "year":     year,
                "pmid":     str(record["MedlineCitation"]["PMID"]),
            })
        except Exception:
            continue

    return documents


def _get_openai_client() -> OpenAI:
    global openai_client

    provider = _get_provider_config()

    if openai_client is None:
        openai_client = OpenAI(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
        )

    return openai_client


def _get_provider_config() -> dict[str, str | None]:
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "model": OPENAI_MODEL,
            "base_url": None,
        }

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return {
            "provider": "groq",
            "api_key": GROQ_API_KEY,
            "model": GROQ_MODEL,
            "base_url": GROQ_BASE_URL,
        }

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'groq'."
    )


def _chunk_documents(raw_docs: list[dict]) -> list[Document]:
    lc_docs = [
        Document(
            page_content=f"Title: {d['title']}\n\nAbstract: {d['abstract']}",
            metadata={"pmid": d["pmid"], "title": d["title"], "year": d["year"]},
        )
        for d in raw_docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(lc_docs)
