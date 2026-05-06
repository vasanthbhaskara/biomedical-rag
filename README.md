# 🧬 Pharma RAG Pipeline

Retrieval-Augmented Generation over PubMed abstracts for **guselkumab (Tremfya)** immunology literature. Built as a production-grade ML system with FastAPI serving, Docker containerisation, RAGAS evaluation, and MLflow experiment tracking.

---

## Architecture

```
PubMed (Entrez API)
       │
       ▼
  Abstract Fetch
       │
       ▼
  Recursive Chunking  ←── chunk_size / overlap (MLflow ablations)
       │
       ▼
  Retrieval Features  ←── TF-IDF n-grams
       │
       ▼
  In-memory cosine index
       │
  Query ──► Embed ──► Retrieve top-k chunks
                           │
                           ▼
               OpenAI-compatible LLM
                           │
                           ▼
                      Grounded Answer
                           │
                     RAGAS Evaluation
                     (faithfulness, relevancy, precision)
```

---

## Stack

| Layer | Tech |
|---|---|
| Data source | PubMed via Biopython Entrez |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Retrieval features | TF-IDF n-grams |
| Vector store | In-memory cosine index |
| Generator | OpenAI-compatible Responses API (Groq or OpenAI) |
| API | FastAPI + Uvicorn |
| Containerisation | Docker + Docker Compose |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |
| Experiment tracking | MLflow |
| UI | Streamlit |

---

## Quickstart

### 1. Set environment variables

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_groq_key
export GROQ_MODEL=openai/gpt-oss-20b
export ENTREZ_EMAIL=your@email.com
```

To use OpenAI instead:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_key
export OPENAI_MODEL=gpt-4o-mini
```

### 2. Run with Docker

```bash
docker compose up --build
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the mechanism of action of guselkumab?", "k": 4}'
```

### 4. Run the Streamlit UI

```bash
pip install streamlit
streamlit run ui/streamlit_app.py
```

### 5. Run RAGAS evaluation

```bash
pip install -r requirements-eval.txt
python eval/ragas_eval.py
```

### 6. Run chunking ablation experiments

```bash
mlflow ui &   # open http://localhost:5000
jupyter notebook experiments/chunking_ablation.ipynb
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/index-stats` | Retrieval index info |
| POST | `/query` | RAG query |

### `/query` request schema

```json
{
  "question": "What is the mechanism of action of guselkumab?",
  "k": 4
}
```

### `/query` response schema

```json
{
  "question": "...",
  "answer": "Guselkumab selectively inhibits IL-23 by targeting its p19 subunit [Source 1]...",
  "sources": [
    {
      "pmid": "28057360",
      "title": "Efficacy and safety of guselkumab...",
      "year": "2017",
      "score": 0.5329
    }
  ]
}
```

---

## Evaluation Results (RAGAS)

Results from ablation study across chunking strategies and retrieval settings:

| Config | Chunk Size | Overlap | Embed Model | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|---|---|---|
| Best run | — | — | — | — | — | — |
| ... | | | | | | |

*Run `experiments/chunking_ablation.ipynb` to populate this table with your results.*

---

## Project Structure

```
pharma-rag/
├── app/
│   ├── main.py              # FastAPI app
│   └── rag_engine.py        # Retrieval + generation core
├── eval/
│   └── ragas_eval.py        # RAGAS evaluation pipeline
├── experiments/
│   └── chunking_ablation.ipynb  # MLflow ablation study
├── ui/
│   └── streamlit_app.py     # Streamlit chat interface
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **TF-IDF cosine retrieval** — lightweight lexical retrieval with no heavyweight model download at startup.
- **OpenAI-compatible client path** — one runtime integration that works with Groq or OpenAI without a separate Groq SDK dependency.
- **`temperature=0.2`** — low temperature for factual, consistent medical answers.
- **`answer ONLY from context`** guardrail — anti-hallucination prompt instruction validated by RAGAS faithfulness metric.
