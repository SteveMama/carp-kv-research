from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

import main


LONG_BENCH_ZIP = Path(__file__).resolve().parent / "benchmark_data" / "longbench_data.zip"
DEFAULT_TASKS = ["passage_retrieval_en", "2wikimqa", "hotpotqa"]


def load_task_rows(task: str, limit: int) -> list[dict]:
    with zipfile.ZipFile(LONG_BENCH_ZIP) as zf:
        with zf.open(f"data/{task}.jsonl") as handle:
            rows = []
            for raw in handle:
                rows.append(json.loads(raw))
                if len(rows) >= limit:
                    break
    return rows


def split_context_into_chunks(context: str) -> list[str]:
    pattern = re.compile(r"(?=(Paragraph \d+:|Passage \d+:))")
    splits = pattern.split(context)
    chunks: list[str] = []
    i = 1
    while i < len(splits):
        label = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ""
        chunks.append((label + body).strip())
        i += 2
    if chunks:
        return chunks
    # Fallback for tasks without explicit labels.
    paras = [part.strip() for part in context.split("\n\n") if part.strip()]
    if paras:
        return paras
    return [context.strip()]


def task_prompt(task: str, question: str) -> str:
    if task == "passage_retrieval_en":
        return (
            "Find the paragraph that best matches the description. "
            "Answer with only the paragraph label such as 'Paragraph 7'.\n"
            f"Question: {question}"
        )
    if task == "passage_count":
        return (
            "Count the matching paragraphs. "
            "Answer with only the final integer.\n"
            f"Question: {question}"
        )
    return (
        "Answer using only the provided context. "
        "Return only the exact short answer.\n"
        f"Question: {question}"
    )


def any_answer_match(prediction: str, answers: list[str]) -> bool:
    return any(main.is_answer_correct(prediction, answer) for answer in answers)


def task_top_k(task: str) -> int:
    if task in {"2wikimqa", "hotpotqa"}:
        return 2
    return 1


def gold_chunk_indices(task: str, chunks: list[str], answers: list[str]) -> list[int]:
    hits: list[int] = []
    if task == "passage_retrieval_en":
        labels = {answer.lower() for answer in answers}
        for idx, chunk in enumerate(chunks):
            prefix = chunk.split(":", 1)[0].strip().lower()
            if prefix in labels:
                hits.append(idx)
        return hits

    normalized_answers = [main.normalize_answer(answer) for answer in answers]
    for idx, chunk in enumerate(chunks):
        norm_chunk = main.normalize_answer(chunk)
        if any(answer in norm_chunk for answer in normalized_answers):
            hits.append(idx)
    return hits


def evaluate_task_retrieval(task: str, rows: list[dict]) -> dict[str, dict[str, float]]:
    encoder = main.MiniLMEncoder(main.HF_MINILM_SNAPSHOT)
    prompts = [row["input"] for row in rows]
    query_embeddings = encoder.encode(prompts, batch_size=8).astype(np.float32)

    all_chunk_texts: list[str] = []
    chunk_groups: list[list[str]] = []
    gold_groups: list[list[int]] = []
    for row in rows:
        chunks = split_context_into_chunks(row["context"])
        chunk_groups.append(chunks)
        gold_groups.append(gold_chunk_indices(task, chunks, row["answers"]))
        all_chunk_texts.extend(chunks)
    all_chunk_embeddings = encoder.encode(all_chunk_texts, batch_size=16).astype(np.float32)

    offset = 0
    chunk_embedding_groups: list[np.ndarray] = []
    for chunks in chunk_groups:
        chunk_embedding_groups.append(all_chunk_embeddings[offset : offset + len(chunks)])
        offset += len(chunks)

    results = {
        "full": {"top1": 0, "top4": 0},
        "q4": {"top1": 0, "top4": 0},
        "turbo_proxy": {"top1": 0, "top4": 0},
        "qaware_sparse": {"top1": 0, "top4": 0},
        "qaware_graph": {"top1": 0, "top4": 0},
    }

    for row, query_vec, chunks, chunk_embs, gold in zip(rows, query_embeddings, chunk_groups, chunk_embedding_groups, gold_groups):
        if not gold:
            continue
        pseudo_conv = main.ConversationExample(turns=chunks, queries=[], latest_turn_for_key={})
        for backend_name in results:
            backend = main.build_backend(
                backend_name,
                pseudo_conv,
                chunk_embs,
                query_embeddings,
                graph_params={"bridge_scale": 0.8, "subject_scale": 1.0},
            )
            scores = backend.score(query_vec, None, row["input"])
            order = np.argsort(-scores)
            results[backend_name]["top1"] += int(order[0] in gold)
            results[backend_name]["top4"] += int(any(idx in gold for idx in order[:4]))

    total = len(rows)
    return {
        name: {"top1": values["top1"] / total, "top4": values["top4"] / total}
        for name, values in results.items()
    }


def evaluate_task(task: str, rows: list[dict]) -> dict[str, dict[str, float]]:
    encoder = main.MiniLMEncoder(main.HF_MINILM_SNAPSHOT)
    prompts = [row["input"] for row in rows]
    query_embeddings = encoder.encode(prompts, batch_size=8).astype(np.float32)

    all_chunk_texts: list[str] = []
    chunk_groups: list[list[str]] = []
    for row in rows:
        chunks = split_context_into_chunks(row["context"])
        chunk_groups.append(chunks)
        all_chunk_texts.extend(chunks)
    all_chunk_embeddings = encoder.encode(all_chunk_texts, batch_size=16).astype(np.float32)

    offset = 0
    chunk_embedding_groups: list[np.ndarray] = []
    for chunks in chunk_groups:
        chunk_embedding_groups.append(all_chunk_embeddings[offset : offset + len(chunks)])
        offset += len(chunks)

    generator = main.KVPressGenerator()
    streaming_press = generator.StreamingLLMPress(compression_ratio=0.5)
    expected_press = generator.ExpectedAttentionPress(
        compression_ratio=0.5,
        n_future_positions=64,
        n_sink=4,
    )

    results = {
        "causal_full": {"correct": 0, "time": 0.0, "tokens": 0},
        "streaming_press": {"correct": 0, "time": 0.0, "tokens": 0},
        "expected_press": {"correct": 0, "time": 0.0, "tokens": 0},
        "qagraph": {"correct": 0, "time": 0.0, "tokens": 0},
    }

    for row, query_vec, chunk_texts, chunk_embs in zip(rows, query_embeddings, chunk_groups, chunk_embedding_groups):
        question = task_prompt(task, row["input"])
        answers = row["answers"]
        full_context = main.conversation_to_context(chunk_texts)

        answer, dt = generator.answer(full_context, question)
        results["causal_full"]["correct"] += int(any_answer_match(answer, answers))
        results["causal_full"]["time"] += dt
        results["causal_full"]["tokens"] += generator.token_count(full_context)

        answer, dt = generator.answer(full_context, question, press=streaming_press)
        results["streaming_press"]["correct"] += int(any_answer_match(answer, answers))
        results["streaming_press"]["time"] += dt
        results["streaming_press"]["tokens"] += generator.token_count(full_context)

        answer, dt = generator.answer(full_context, question, press=expected_press)
        results["expected_press"]["correct"] += int(any_answer_match(answer, answers))
        results["expected_press"]["time"] += dt
        results["expected_press"]["tokens"] += generator.token_count(full_context)

        qagraph = main.QueryAwareGraphMemory(
            turns_text=chunk_texts,
            turns=chunk_embs,
            initial_queries=query_embeddings,
            rank=8,
            sparse_k=2,
            query_decay=0.92,
            graph_alpha=1.25,
            top_seed_k=4,
            bridge_scale=0.8,
            subject_scale=1.0,
        )
        scores = qagraph.score(query_vec, None, row["input"])
        idx = main.select_qagraph_turns(qagraph, scores, row["input"], task_top_k(task))
        retrieved_context = main.conversation_to_context([chunk_texts[i] for i in idx])
        answer, dt = generator.answer(retrieved_context, question)
        results["qagraph"]["correct"] += int(any_answer_match(answer, answers))
        results["qagraph"]["time"] += dt
        results["qagraph"]["tokens"] += generator.token_count(retrieved_context)

    total = len(rows)
    return {
        name: {
            "exact_match": values["correct"] / total,
            "avg_s": values["time"] / total,
            "avg_ctx_tokens": values["tokens"] / total,
        }
        for name, values in results.items()
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Run a small public LongBench subset comparison.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--mode", choices=["generation", "retrieval"], default="retrieval")
    args = parser.parse_args()

    print(f"Using LongBench archive: {LONG_BENCH_ZIP}")
    for task in args.tasks:
        rows = load_task_rows(task, args.limit)
        result = evaluate_task_retrieval(task, rows) if args.mode == "retrieval" else evaluate_task(task, rows)
        print(f"\n=== {task} ({len(rows)} examples) ===")
        for name, metrics in result.items():
            if args.mode == "retrieval":
                print(f"{name:16s} top1={metrics['top1']:.4f} top4={metrics['top4']:.4f}")
            else:
                print(
                    f"{name:16s} "
                    f"exact_match={metrics['exact_match']:.4f} "
                    f"avg_s={metrics['avg_s']:.3f} "
                    f"avg_ctx_tokens={metrics['avg_ctx_tokens']:.1f}"
                )


if __name__ == "__main__":
    main_cli()
