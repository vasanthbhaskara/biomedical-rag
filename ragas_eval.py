# ============================================================
# eval/ragas_eval.py — RAGAS evaluation pipeline
#
# Metrics:
#   • faithfulness      — does the answer stay grounded in context?
#   • answer_relevancy  — does the answer actually address the question?
#   • context_precision — are the retrieved chunks actually useful?
#
# Usage:
#   python eval/ragas_eval.py
# ============================================================

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper

from rag_engine import init_engine, retrieve, generate


# ── Evaluation questions ──────────────────────────────────────
EVAL_QUESTIONS = [
    "What is the mechanism of action of guselkumab?",
    "What are the most common adverse effects of Tremfya?",
    "How effective is guselkumab for psoriatic arthritis?",
    "What is the recommended dosage and injection schedule for guselkumab?",
    "How does guselkumab compare to adalimumab in clinical trials?",
    "What cytokine does guselkumab selectively inhibit?",
    "What were the results of the VOYAGE 1 trial?",
    "Is guselkumab safe for long-term use?",
]


def build_eval_dataset(questions: list[str], k: int = 4) -> Dataset:
    """Run RAG on each question and collect inputs/outputs for RAGAS."""
    rows = {
        "question":  [],
        "answer":    [],
        "contexts":  [],
        "reference": [],   # RAGAS needs this field; we leave it empty
    }

    for q in questions:
        print(f"  Evaluating: {q[:60]}...")
        context_chunks = retrieve(q, k=k)
        answer = generate(q, context_chunks)

        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append([c["content"] for c in context_chunks])
        rows["reference"].append("")   # no ground truth — unsupervised eval

    return Dataset.from_dict(rows)


def run_evaluation(k: int = 4) -> dict:
    """Full eval pipeline — returns metric scores."""
    print("\n[eval] Initialising RAG engine...")
    init_engine()

    print(f"\n[eval] Building eval dataset ({len(EVAL_QUESTIONS)} questions)...")
    dataset = build_eval_dataset(EVAL_QUESTIONS, k=k)

    print("\n[eval] Skipping LLM-based RAGAS evaluation (Groq removed)...")
    return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}


if __name__ == "__main__":
    run_evaluation()
