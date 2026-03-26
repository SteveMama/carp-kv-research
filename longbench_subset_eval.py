from __future__ import annotations

import argparse
import gc
import json
import re
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from main import (
    FullMemory,
    HF_MINILM_SNAPSHOT,
    HF_QWEN_SMALL,
    KVPressGenerator,
    MiniLMEncoder,
    QueryAwareGraphMemory,
    QueryAwareSparseMemory,
    RESULTS_DIR,
    QueryExample,
    ConversationExample,
    normalize_answer,
    select_qagraph_turns,
    top_indices,
)

KV_PRESS_ROOT = Path(__file__).resolve().parent / "kvpress"
if str(KV_PRESS_ROOT) not in sys.path:
    sys.path.insert(0, str(KV_PRESS_ROOT))

try:
    from kvpress import PyramidKVPress, SnapKVPress
except Exception:
    PyramidKVPress = None
    SnapKVPress = None

try:
    from llmlingua import PromptCompressor
except Exception:
    PromptCompressor = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


LONG_BENCH_ROOT = Path(__file__).resolve().parent / "LongBenchRepo" / "LongBench"
LONG_BENCH_DATA_DIR = LONG_BENCH_ROOT / "data"
LONG_BENCH_CONFIG_DIR = LONG_BENCH_ROOT / "config"
LONG_BENCH_JSONL_LOG = RESULTS_DIR / "longbench_subset_log.jsonl"
LONG_BENCH_MD_LOG = Path(__file__).resolve().parent / "LONG_BENCH_EXPERIMENTS.md"

DEFAULT_TASKS = ["qasper", "multifieldqa_en", "2wikimqa"]


@dataclass
class LongBenchItem:
    dataset: str
    input: str
    context: str
    answers: list[str]
    length: int
    all_classes: list[str]
    item_id: str


@dataclass
class LearnedReranker:
    weights: np.ndarray
    bias: float


@dataclass
class LearnedRouter:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray


def current_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 10**9:
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0 / 1024.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_longbench_items(
    dataset: str,
    max_samples: int,
    max_length_words: int | None = None,
) -> list[LongBenchItem]:
    path = LONG_BENCH_DATA_DIR / f"{dataset}.jsonl"
    items: list[LongBenchItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if max_length_words is not None and raw["length"] > max_length_words:
                continue
            items.append(
                LongBenchItem(
                    dataset=raw["dataset"],
                    input=raw["input"],
                    context=raw["context"],
                    answers=raw["answers"],
                    length=raw["length"],
                    all_classes=raw.get("all_classes") or [],
                    item_id=raw["_id"],
                )
            )
    items.sort(key=lambda item: item.length)
    return items[:max_samples]


def load_prompt_config() -> tuple[dict[str, str], dict[str, int]]:
    prompt_map = load_json(LONG_BENCH_CONFIG_DIR / "dataset2prompt.json")
    maxlen_map = load_json(LONG_BENCH_CONFIG_DIR / "dataset2maxlen.json")
    return prompt_map, maxlen_map


def make_question_prompt(dataset: str, question: str, prompt_template: str) -> str:
    if "{context}" in prompt_template:
        prefix, suffix = prompt_template.split("{context}", 1)
        prefix = re.sub(r"(?i)(article|passages?|text):\s*$", "", prefix).strip()
        suffix = suffix.format(input=question).strip()
        merged = "\n\n".join(part for part in [prefix, suffix] if part)
        return merged
    return prompt_template.format(input=question).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", text.lower())


def sentence_window_chunks(text: str, window_words: int) -> list[str]:
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if current and count + len(words) > window_words:
            chunks.append(" ".join(current).strip())
            current = []
            count = 0
        current.append(sentence.strip())
        count += len(words)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def split_context_into_chunks(dataset: str, context: str, chunk_words: int) -> list[str]:
    context = normalize_whitespace(context)
    if dataset in {"2wikimqa", "hotpotqa", "musique"}:
        pieces = re.split(r"(?=Passage\s+\d+:)", context)
        chunks = [piece.strip() for piece in pieces if piece.strip()]
    else:
        pieces = re.split(r"\n\s*\n", context)
        chunks = [piece.strip() for piece in pieces if piece.strip()]

    if len(chunks) <= 1:
        chunks = sentence_window_chunks(context, chunk_words)

    refined: list[str] = []
    for chunk in chunks:
        if len(chunk.split()) <= chunk_words:
            refined.append(chunk)
            continue
        refined.extend(sentence_window_chunks(chunk, chunk_words))

    return refined if refined else [context]


def truncate_middle_by_tokens(tokenizer, text: str, max_tokens: int) -> tuple[str, bool, int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    original = len(token_ids)
    if original <= max_tokens:
        return text, False, original
    left = max_tokens // 2
    right = max_tokens - left
    kept = token_ids[:left] + token_ids[-right:]
    return tokenizer.decode(kept, skip_special_tokens=True), True, original


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gold_tokens), 1)
    return (2 * precision * recall) / (precision + recall)


def score_prediction(dataset: str, prediction: str, answers: list[str], all_classes: list[str]) -> float:
    del all_classes
    if dataset in {"qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "narrativeqa", "musique"}:
        return max(qa_f1_score(prediction, answer) for answer in answers)
    return float(any(normalize_answer(prediction) == normalize_answer(answer) for answer in answers))


def answer_with_press(
    generator: KVPressGenerator,
    context: str,
    question_prompt: str,
    max_new_tokens: int,
    press=None,
) -> tuple[str, float]:
    start = time.perf_counter()
    result = generator.pipe(
        context,
        question=question_prompt,
        press=press,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.perf_counter() - start
    return result["answer"].strip(), elapsed


def build_chunk_backends(
    chunks: list[str],
    chunk_embeddings: np.ndarray,
    query_embedding: np.ndarray,
) -> dict[str, object]:
    conversation = ConversationExample(turns=chunks, queries=[QueryExample(text="", answer="", support_turn=0)])
    query_bank = np.expand_dims(query_embedding, axis=0)
    return {
        "dense_topk": FullMemory(chunk_embeddings),
        "qaware_topk": QueryAwareSparseMemory(
            turns=chunk_embeddings,
            initial_queries=query_bank,
            ledger_turns={},
            rank=8,
            sparse_k=2,
            query_decay=0.92,
        ),
        "qagraph_topk": QueryAwareGraphMemory(
            turns_text=chunks,
            turns=chunk_embeddings,
            initial_queries=query_bank,
            ledger_turns={},
            rank=8,
            sparse_k=2,
            query_decay=0.92,
            graph_alpha=1.25,
            top_seed_k=4,
            bridge_scale=0.8,
            subject_scale=1.0,
        ),
        "_conversation": conversation,
    }


def build_bm25(chunks: list[str]):
    if BM25Okapi is None:
        return None
    tokenized = [bm25_tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized)


def compact_context(chunks: list[str], indices: Iterable[int]) -> str:
    return "\n\n".join(chunks[i] for i in indices)


def should_use_graph(dataset: str, chunks: list[str]) -> bool:
    if dataset in {"2wikimqa", "hotpotqa", "musique"}:
        return True
    return any(chunk.lstrip().startswith("Passage ") for chunk in chunks[: min(len(chunks), 4)])


def preferred_chunk_words(dataset: str, context: str, base_chunk_words: int) -> int:
    if dataset in {"2wikimqa", "hotpotqa", "musique"} or "Passage 1:" in context:
        return base_chunk_words
    return max(80, base_chunk_words // 2)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    mean = float(scores.mean())
    std = float(scores.std())
    if std < 1e-6:
        return scores - mean
    return (scores - mean) / std


def answer_in_chunk(chunk: str, answers: list[str]) -> bool:
    normalized_chunk = normalize_answer(chunk)
    if not normalized_chunk:
        return False
    for answer in answers:
        normalized_answer = normalize_answer(answer)
        if normalized_answer and normalized_answer in normalized_chunk:
            return True
    return False


def build_feature_matrix(
    dataset: str,
    chunks: list[str],
    query_text: str,
    dense_scores: np.ndarray,
    qaware_scores: np.ndarray,
    qagraph_scores: np.ndarray,
    bm25_scores: np.ndarray | None,
) -> np.ndarray:
    dense_n = normalize_scores(dense_scores)
    qaware_n = normalize_scores(qaware_scores)
    qagraph_n = normalize_scores(qagraph_scores)
    if bm25_scores is None:
        bm25_n = np.zeros_like(dense_n)
    else:
        bm25_n = normalize_scores(bm25_scores)
    lengths = np.array([len(chunk.split()) for chunk in chunks], dtype=np.float32)
    length_n = normalize_scores(lengths)
    chunk_positions = np.linspace(0.0, 1.0, len(chunks), dtype=np.float32) if chunks else np.array([], dtype=np.float32)
    query_tokens = set(bm25_tokenize(query_text))
    overlap = np.array(
        [
            len(query_tokens & set(bm25_tokenize(chunk))) / max(len(query_tokens), 1)
            for chunk in chunks
        ],
        dtype=np.float32,
    )
    overlap_n = normalize_scores(overlap)
    passage_flag = np.array(
        [1.0 if chunk.lstrip().startswith("Passage ") else 0.0 for chunk in chunks],
        dtype=np.float32,
    )
    graph_flag = float(should_use_graph(dataset, chunks))
    graph_flags = np.full_like(dense_n, graph_flag)
    features = np.stack(
        [
            dense_n,
            qaware_n,
            qagraph_n,
            bm25_n,
            dense_n * qaware_n,
            dense_n * qagraph_n,
            qaware_n * bm25_n,
            graph_flags,
            length_n,
            overlap_n,
            chunk_positions,
            passage_flag,
            dense_n * overlap_n,
            bm25_n * overlap_n,
        ],
        axis=1,
    )
    return features.astype(np.float32)


def fit_learned_reranker(
    examples: list[tuple[np.ndarray, np.ndarray]],
    steps: int = 600,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> LearnedReranker | None:
    if not examples:
        return None
    pair_diffs: list[np.ndarray] = []
    point_x: list[np.ndarray] = []
    point_y: list[np.ndarray] = []

    for features, labels in examples:
        pos_idx = np.where(labels > 0.5)[0]
        neg_idx = np.where(labels <= 0.5)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        point_x.append(features)
        point_y.append(labels.astype(np.float32))
        for i in pos_idx:
            neg_scores = features[neg_idx, 0]
            hard_order = neg_idx[np.argsort(-neg_scores)[: min(8, len(neg_idx))]]
            for j in hard_order:
                pair_diffs.append(features[i] - features[j])

    if not pair_diffs or not point_x:
        return None

    x_pair = np.stack(pair_diffs).astype(np.float32)
    x = np.concatenate(point_x, axis=0).astype(np.float32)
    y = np.concatenate(point_y, axis=0).astype(np.float32)
    positive = float(y.sum())
    negative = float(len(y) - positive)

    weights = np.zeros(x_pair.shape[1], dtype=np.float32)
    bias = 0.0
    pos_weight = negative / positive
    sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)

    for _ in range(steps):
        pair_logits = x_pair @ weights
        pair_probs = 1.0 / (1.0 + np.exp(-np.clip(pair_logits, -30.0, 30.0)))
        pair_error = pair_probs - 1.0
        grad_pair = (x_pair.T @ pair_error) / len(x_pair)

        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = (probs - y) * sample_weight
        grad_point = (x.T @ error) / len(y)
        grad_w = 0.75 * grad_pair + 0.25 * grad_point + l2 * weights
        grad_b = float(error.mean()) * 0.25
        weights -= lr * grad_w
        bias -= lr * grad_b

    return LearnedReranker(weights=weights, bias=bias)


def select_learned_turns(
    reranker: LearnedReranker,
    features: np.ndarray,
    retrieval_top_k: int,
) -> list[int]:
    logits = features @ reranker.weights + reranker.bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    ordered = np.argsort(-probs)
    if len(ordered) < 2:
        return ordered[:retrieval_top_k].tolist()
    margin = float(probs[ordered[0]] - probs[ordered[1]])
    extra_k = 0
    if margin < 0.15:
        extra_k = 2
    if margin < 0.05:
        extra_k = 4
    final_k = min(len(ordered), retrieval_top_k + extra_k)
    return ordered[:final_k].tolist()


def learned_probabilities(
    reranker: LearnedReranker,
    features: np.ndarray,
) -> np.ndarray:
    logits = features @ reranker.weights + reranker.bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def select_hybrid_turns(
    dataset: str,
    chunks: list[str],
    retrieval_top_k: int,
    dense_scores: np.ndarray,
    qaware_scores: np.ndarray,
    qagraph_scores: np.ndarray,
    bm25_scores: np.ndarray | None,
) -> list[int]:
    dense_n = normalize_scores(dense_scores)
    qaware_n = normalize_scores(qaware_scores)
    qagraph_n = normalize_scores(qagraph_scores)
    if bm25_scores is None:
        bm25_n = np.zeros_like(dense_n)
    else:
        bm25_n = normalize_scores(bm25_scores)

    if should_use_graph(dataset, chunks):
        hybrid = 0.45 * dense_n + 0.15 * qaware_n + 0.30 * qagraph_n + 0.10 * bm25_n
    else:
        hybrid = 0.55 * dense_n + 0.25 * qaware_n + 0.05 * qagraph_n + 0.15 * bm25_n

    ordered = np.argsort(-hybrid)
    if hybrid.size < 2:
        return ordered[:retrieval_top_k].tolist()

    margin = float(hybrid[ordered[0]] - hybrid[ordered[1]])
    extra_k = 0
    if margin < 0.15:
        extra_k = 2
    if margin < 0.05:
        extra_k = 4
    final_k = min(len(hybrid), retrieval_top_k + extra_k)
    return ordered[:final_k].tolist()


def selection_utility(indices: list[int], labels: np.ndarray) -> float:
    if not indices:
        return 0.0
    picked = labels[indices]
    return float(picked.max()) + 0.1 * float(picked.sum()) - 0.01 * len(indices)


def build_router_features(
    dataset: str,
    chunks: list[str],
    dense_scores: np.ndarray,
    qaware_scores: np.ndarray,
    qagraph_scores: np.ndarray,
    bm25_scores: np.ndarray | None,
    learned_probs: np.ndarray,
    learned_idx: list[int],
    hybrid_idx: list[int],
) -> np.ndarray:
    dense_n = normalize_scores(dense_scores)
    qaware_n = normalize_scores(qaware_scores)
    qagraph_n = normalize_scores(qagraph_scores)
    bm25_n = np.zeros_like(dense_n) if bm25_scores is None else normalize_scores(bm25_scores)

    def top_margin(scores: np.ndarray) -> float:
        ordered = np.sort(scores)[::-1]
        if len(ordered) < 2:
            return float(ordered[0]) if len(ordered) == 1 else 0.0
        return float(ordered[0] - ordered[1])

    learned_order = np.sort(learned_probs)[::-1]
    learned_margin = float(learned_order[0] - learned_order[1]) if len(learned_order) > 1 else float(learned_order[0])
    jaccard = 0.0
    if learned_idx or hybrid_idx:
        inter = len(set(learned_idx) & set(hybrid_idx))
        union = len(set(learned_idx) | set(hybrid_idx))
        jaccard = inter / max(union, 1)

    features = np.array(
        [
            float(should_use_graph(dataset, chunks)),
            float(len(chunks)),
            top_margin(dense_n),
            top_margin(qaware_n),
            top_margin(qagraph_n),
            top_margin(bm25_n),
            learned_margin,
            float(learned_probs.max()) if learned_probs.size else 0.0,
            jaccard,
            float(np.mean([len(chunks[i].split()) for i in learned_idx])) if learned_idx else 0.0,
            float(np.mean([len(chunks[i].split()) for i in hybrid_idx])) if hybrid_idx else 0.0,
        ],
        dtype=np.float32,
    )
    return features


def fit_learned_router(
    examples: list[tuple[np.ndarray, float]],
    steps: int = 400,
    lr: float = 0.1,
    l2: float = 1e-3,
) -> LearnedRouter | None:
    if not examples:
        return None
    x = np.stack([feat for feat, _ in examples]).astype(np.float32)
    y = np.array([label for _, label in examples], dtype=np.float32)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives < 1.0 or negatives < 1.0:
        return None
    feature_mean = x.mean(axis=0).astype(np.float32)
    feature_std = x.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    x = ((x - feature_mean) / feature_std).astype(np.float32)
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = 0.0
    pos_weight = negatives / positives
    sample_weight = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    for _ in range(steps):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        error = (probs - y) * sample_weight
        grad_w = (x.T @ error) / len(y) + l2 * weights
        grad_b = float(error.mean())
        weights -= lr * grad_w
        bias -= lr * grad_b
    return LearnedRouter(weights=weights, bias=bias, feature_mean=feature_mean, feature_std=feature_std)


def route_selection(
    router: LearnedRouter | None,
    router_features: np.ndarray,
    learned_idx: list[int],
    hybrid_idx: list[int],
) -> list[int]:
    if router is None:
        return hybrid_idx if router_features[0] > 0.5 else learned_idx
    normed = ((router_features.astype(np.float32) - router.feature_mean) / router.feature_std).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-float(normed @ router.weights + router.bias)))
    return learned_idx if prob >= 0.5 else hybrid_idx


def ensure_logs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def maybe_init_llmlingua(model_name: str):
    if PromptCompressor is None:
        return None
    return PromptCompressor(
        model_name=model_name,
        device_map="cpu",
        use_llmlingua2=True,
    )


def collect_reranker_examples(
    dataset: str,
    items: list[LongBenchItem],
    encoder: MiniLMEncoder,
    chunk_words: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    for item in items:
        effective_chunk_words = preferred_chunk_words(dataset, item.context, chunk_words)
        chunks = split_context_into_chunks(dataset, item.context, chunk_words=effective_chunk_words)
        labels = np.array([1.0 if answer_in_chunk(chunk, item.answers) else 0.0 for chunk in chunks], dtype=np.float32)
        if labels.sum() < 1:
            continue
        chunk_embeddings = encoder.encode(chunks, batch_size=16).astype(np.float32)
        query_embedding = encoder.encode([item.input], batch_size=1)[0].astype(np.float32)
        backends = build_chunk_backends(chunks, chunk_embeddings, query_embedding)
        bm25 = build_bm25(chunks)
        dense_scores = backends["dense_topk"].score(query_embedding, None, item.input)
        qaware_scores = backends["qaware_topk"].score(query_embedding, None, item.input)
        qagraph_scores = backends["qagraph_topk"].score(query_embedding, None, item.input)
        bm25_scores = None
        if bm25 is not None:
            bm25_scores = np.asarray(bm25.get_scores(bm25_tokenize(item.input)), dtype=np.float32)
        features = build_feature_matrix(
            dataset=dataset,
            chunks=chunks,
            query_text=item.input,
            dense_scores=dense_scores,
            qaware_scores=qaware_scores,
            qagraph_scores=qagraph_scores,
            bm25_scores=bm25_scores,
        )
        examples.append((features, labels))
    return examples


def collect_router_examples(
    dataset: str,
    items: list[LongBenchItem],
    encoder: MiniLMEncoder,
    chunk_words: int,
    retrieval_top_k: int,
    learned_reranker: LearnedReranker | None,
) -> list[tuple[np.ndarray, float]]:
    if learned_reranker is None:
        return []
    examples: list[tuple[np.ndarray, float]] = []
    for item in items:
        effective_chunk_words = preferred_chunk_words(dataset, item.context, chunk_words)
        chunks = split_context_into_chunks(dataset, item.context, chunk_words=effective_chunk_words)
        labels = np.array([1.0 if answer_in_chunk(chunk, item.answers) else 0.0 for chunk in chunks], dtype=np.float32)
        if labels.sum() < 1:
            continue
        chunk_embeddings = encoder.encode(chunks, batch_size=16).astype(np.float32)
        query_embedding = encoder.encode([item.input], batch_size=1)[0].astype(np.float32)
        backends = build_chunk_backends(chunks, chunk_embeddings, query_embedding)
        bm25 = build_bm25(chunks)
        dense_scores = backends["dense_topk"].score(query_embedding, None, item.input)
        qaware_scores = backends["qaware_topk"].score(query_embedding, None, item.input)
        qagraph_scores = backends["qagraph_topk"].score(query_embedding, None, item.input)
        bm25_scores = None
        if bm25 is not None:
            bm25_scores = np.asarray(bm25.get_scores(bm25_tokenize(item.input)), dtype=np.float32)
        feature_matrix = build_feature_matrix(
            dataset=dataset,
            chunks=chunks,
            query_text=item.input,
            dense_scores=dense_scores,
            qaware_scores=qaware_scores,
            qagraph_scores=qagraph_scores,
            bm25_scores=bm25_scores,
        )
        learned_probs = learned_probabilities(learned_reranker, feature_matrix)
        learned_idx = select_learned_turns(learned_reranker, feature_matrix, retrieval_top_k)
        hybrid_idx = select_hybrid_turns(
            dataset=dataset,
            chunks=chunks,
            retrieval_top_k=retrieval_top_k,
            dense_scores=dense_scores,
            qaware_scores=qaware_scores,
            qagraph_scores=qagraph_scores,
            bm25_scores=bm25_scores,
        )
        learned_utility = selection_utility(learned_idx, labels)
        hybrid_utility = selection_utility(hybrid_idx, labels)
        if abs(learned_utility - hybrid_utility) < 1e-6:
            continue
        router_features = build_router_features(
            dataset=dataset,
            chunks=chunks,
            dense_scores=dense_scores,
            qaware_scores=qaware_scores,
            qagraph_scores=qagraph_scores,
            bm25_scores=bm25_scores,
            learned_probs=learned_probs,
            learned_idx=learned_idx,
            hybrid_idx=hybrid_idx,
        )
        label = 1.0 if learned_utility > hybrid_utility else 0.0
        examples.append((router_features, label))
    return examples


def append_longbench_log(record: dict) -> None:
    ensure_logs()
    with LONG_BENCH_JSONL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    lines = [
        f"## {record['timestamp']}",
        f"- Notes: {record['notes'] or 'n/a'}",
        f"- Tasks: {', '.join(record['tasks'])}",
        f"- Samples/task: {record['config']['max_samples_per_task']}",
        f"- Full-context token cap: {record['config']['full_max_context_tokens']}",
        f"- Top-k chunks: {record['config']['retrieval_top_k']}",
        f"- Peak RSS (MB): {record['peak_rss_mb']:.1f}",
    ]
    for task, task_result in record["task_results"].items():
        valid_methods = [
            item for item in task_result["methods"].items() if item[1]["score"] is not None
        ]
        if valid_methods:
            best = max(valid_methods, key=lambda item: item[1]["score"])
            lines.append(
                f"- {task}: best `{best[0]}` score={best[1]['score']:.2f}, "
                f"samples={task_result['samples']}, trunc_rate={task_result['truncation_rate']:.2f}"
            )
        else:
            lines.append(
                f"- {task}: no successful methods, samples={task_result['samples']}, "
                f"trunc_rate={task_result['truncation_rate']:.2f}"
            )
    lines.append("")
    with LONG_BENCH_MD_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def evaluate_task(
    dataset: str,
    items: list[LongBenchItem],
    encoder: MiniLMEncoder,
    generator: KVPressGenerator,
    llmlingua_compressor,
    learned_reranker: LearnedReranker | None,
    learned_router: LearnedRouter | None,
    prompt_template: str,
    max_new_tokens: int,
    retrieval_top_k: int,
    full_max_context_tokens: int,
    chunk_words: int,
    compression_ratio: float,
) -> dict:
    streaming_press = generator.StreamingLLMPress(compression_ratio=compression_ratio)
    expected_press = generator.ExpectedAttentionPress(
        compression_ratio=compression_ratio,
        n_future_positions=64,
        n_sink=4,
    )
    snapkv_press = SnapKVPress(compression_ratio=compression_ratio) if SnapKVPress is not None else None
    pyramidkv_press = PyramidKVPress(compression_ratio=compression_ratio) if PyramidKVPress is not None else None
    methods = {
        "causal_full": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "streaming_press": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "expected_press": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "snapkv": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "pyramidkv": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "bm25_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "dense_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "qaware_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "qagraph_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "adaptive_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "hybrid_mix_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "learned_mix_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "task_adaptive_mix_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "routed_mix_topk": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
        "llmlingua2": {"score_sum": 0.0, "latency_sum": 0.0, "ctx_tokens_sum": 0, "count": 0},
    }
    truncated = 0
    chunk_stats: list[int] = []

    for index, item in enumerate(items, start=1):
        question_prompt = make_question_prompt(dataset, item.input, prompt_template)
        full_context, was_truncated, original_tokens = truncate_middle_by_tokens(
            generator.tokenizer,
            item.context,
            full_max_context_tokens,
        )
        truncated += int(was_truncated)

        answer, dt = answer_with_press(
            generator=generator,
            context=full_context,
            question_prompt=question_prompt,
            press=None,
            max_new_tokens=max_new_tokens,
        )
        methods["causal_full"]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
        methods["causal_full"]["latency_sum"] += dt
        methods["causal_full"]["ctx_tokens_sum"] += min(original_tokens, full_max_context_tokens)
        methods["causal_full"]["count"] += 1

        answer, dt = answer_with_press(
            generator=generator,
            context=full_context,
            question_prompt=question_prompt,
            press=streaming_press,
            max_new_tokens=max_new_tokens,
        )
        methods["streaming_press"]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
        methods["streaming_press"]["latency_sum"] += dt
        methods["streaming_press"]["ctx_tokens_sum"] += min(original_tokens, full_max_context_tokens)
        methods["streaming_press"]["count"] += 1

        answer, dt = answer_with_press(
            generator=generator,
            context=full_context,
            question_prompt=question_prompt,
            press=expected_press,
            max_new_tokens=max_new_tokens,
        )
        methods["expected_press"]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
        methods["expected_press"]["latency_sum"] += dt
        methods["expected_press"]["ctx_tokens_sum"] += min(original_tokens, full_max_context_tokens)
        methods["expected_press"]["count"] += 1

        if snapkv_press is not None:
            try:
                answer, dt = answer_with_press(
                    generator=generator,
                    context=full_context,
                    question_prompt=question_prompt,
                    press=snapkv_press,
                    max_new_tokens=max_new_tokens,
                )
                methods["snapkv"]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
                methods["snapkv"]["latency_sum"] += dt
                methods["snapkv"]["ctx_tokens_sum"] += min(original_tokens, full_max_context_tokens)
                methods["snapkv"]["count"] += 1
            except Exception:
                snapkv_press = None

        if pyramidkv_press is not None:
            try:
                answer, dt = answer_with_press(
                    generator=generator,
                    context=full_context,
                    question_prompt=question_prompt,
                    press=pyramidkv_press,
                    max_new_tokens=max_new_tokens,
                )
                methods["pyramidkv"]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
                methods["pyramidkv"]["latency_sum"] += dt
                methods["pyramidkv"]["ctx_tokens_sum"] += min(original_tokens, full_max_context_tokens)
                methods["pyramidkv"]["count"] += 1
            except Exception:
                pyramidkv_press = None

        effective_chunk_words = preferred_chunk_words(dataset, item.context, chunk_words)
        chunks = split_context_into_chunks(dataset, item.context, chunk_words=effective_chunk_words)
        chunk_stats.append(len(chunks))
        chunk_embeddings = encoder.encode(chunks, batch_size=16).astype(np.float32)
        query_embedding = encoder.encode([item.input], batch_size=1)[0].astype(np.float32)
        backends = build_chunk_backends(chunks, chunk_embeddings, query_embedding)
        bm25 = build_bm25(chunks)

        dense_idx = top_indices(backends["dense_topk"].score(query_embedding, None, item.input), retrieval_top_k)
        qaware_backend = backends["qaware_topk"]
        qaware_idx = top_indices(qaware_backend.score(query_embedding, None, item.input), retrieval_top_k)
        qagraph_backend = backends["qagraph_topk"]
        qagraph_scores = qagraph_backend.score(query_embedding, None, item.input)
        qagraph_idx = select_qagraph_turns(qagraph_backend, qagraph_scores, item.input, retrieval_top_k)
        bm25_idx: list[int] = []
        if bm25 is not None:
            bm25_scores = bm25.get_scores(bm25_tokenize(item.input))
            bm25_idx = np.argsort(-np.asarray(bm25_scores))[:retrieval_top_k].tolist()
        else:
            bm25_scores = None
        adaptive_idx: list[int]
        if should_use_graph(dataset, chunks):
            adaptive_idx = qagraph_idx
        else:
            adaptive_k = min(2, retrieval_top_k)
            adaptive_idx = top_indices(qaware_backend.score(query_embedding, None, item.input), adaptive_k)
        hybrid_idx = select_hybrid_turns(
            dataset=dataset,
            chunks=chunks,
            retrieval_top_k=retrieval_top_k,
            dense_scores=backends["dense_topk"].score(query_embedding, None, item.input),
            qaware_scores=qaware_backend.score(query_embedding, None, item.input),
            qagraph_scores=qagraph_scores,
            bm25_scores=None if bm25_scores is None else np.asarray(bm25_scores, dtype=np.float32),
        )
        feature_matrix = build_feature_matrix(
            dataset=dataset,
            chunks=chunks,
            query_text=item.input,
            dense_scores=backends["dense_topk"].score(query_embedding, None, item.input),
            qaware_scores=qaware_backend.score(query_embedding, None, item.input),
            qagraph_scores=qagraph_scores,
            bm25_scores=None if bm25_scores is None else np.asarray(bm25_scores, dtype=np.float32),
        )
        if learned_reranker is not None:
            learned_idx = select_learned_turns(learned_reranker, feature_matrix, retrieval_top_k)
            learned_probs = learned_probabilities(learned_reranker, feature_matrix)
        else:
            learned_idx = hybrid_idx
            learned_probs = np.zeros(len(chunks), dtype=np.float32)
        task_adaptive_idx = hybrid_idx if should_use_graph(dataset, chunks) else learned_idx
        router_features = build_router_features(
            dataset=dataset,
            chunks=chunks,
            dense_scores=backends["dense_topk"].score(query_embedding, None, item.input),
            qaware_scores=qaware_backend.score(query_embedding, None, item.input),
            qagraph_scores=qagraph_scores,
            bm25_scores=None if bm25_scores is None else np.asarray(bm25_scores, dtype=np.float32),
            learned_probs=learned_probs,
            learned_idx=learned_idx,
            hybrid_idx=hybrid_idx,
        )
        routed_idx = route_selection(learned_router, router_features, learned_idx, hybrid_idx)

        for method_name, indices in {
            "bm25_topk": bm25_idx if bm25_idx else dense_idx,
            "dense_topk": dense_idx,
            "qaware_topk": qaware_idx,
            "qagraph_topk": qagraph_idx,
            "adaptive_topk": adaptive_idx,
            "hybrid_mix_topk": hybrid_idx,
            "learned_mix_topk": learned_idx,
            "task_adaptive_mix_topk": task_adaptive_idx,
            "routed_mix_topk": routed_idx,
        }.items():
            compact = compact_context(chunks, indices)
            answer, dt = answer_with_press(
                generator=generator,
                context=compact,
                question_prompt=question_prompt,
                press=None,
                max_new_tokens=max_new_tokens,
            )
            methods[method_name]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
            methods[method_name]["latency_sum"] += dt
            methods[method_name]["ctx_tokens_sum"] += generator.token_count(compact)
            methods[method_name]["count"] += 1

        if llmlingua_compressor is not None:
            target_tokens = max(128, int(min(original_tokens, full_max_context_tokens) * compression_ratio))
            start = time.perf_counter()
            compressed = llmlingua_compressor.compress_prompt(
                chunks,
                instruction="",
                question=item.input,
                target_token=target_tokens,
                use_sentence_level_filter=False,
                use_context_level_filter=True,
                use_token_level_filter=True,
                keep_first_sentence=0,
                keep_last_sentence=0,
                force_reserve_digit=True,
            )
            compression_dt = time.perf_counter() - start
            compressed_context = compressed["compressed_prompt"]
            answer, dt = answer_with_press(
                generator=generator,
                context=compressed_context,
                question_prompt=question_prompt,
                press=None,
                max_new_tokens=max_new_tokens,
            )
            methods["llmlingua2"]["score_sum"] += score_prediction(dataset, answer, item.answers, item.all_classes)
            methods["llmlingua2"]["latency_sum"] += compression_dt + dt
            methods["llmlingua2"]["ctx_tokens_sum"] += generator.token_count(compressed_context)
            methods["llmlingua2"]["count"] += 1

        print(
            f"{dataset:16s} sample={index}/{len(items)} "
            f"len={item.length} rss_mb={current_rss_mb():.1f} "
            f"chunks={len(chunks)} full_tokens={min(original_tokens, full_max_context_tokens)}"
        )
        del chunk_embeddings
        del backends
        gc.collect()

    summary = {
        "samples": len(items),
        "truncation_rate": truncated / max(len(items), 1),
        "avg_chunks": float(np.mean(chunk_stats)) if chunk_stats else 0.0,
        "methods": {},
    }
    for name, stats in methods.items():
        if stats["count"] == 0:
            summary["methods"][name] = {
                "score": None,
                "avg_s": None,
                "avg_ctx_tokens": None,
            }
            continue
        summary["methods"][name] = {
            "score": round(100.0 * stats["score_sum"] / stats["count"], 2),
            "avg_s": round(stats["latency_sum"] / stats["count"], 3),
            "avg_ctx_tokens": round(stats["ctx_tokens_sum"] / stats["count"], 1),
        }
    return summary


def run(args: argparse.Namespace) -> None:
    try:
        import torch

        torch.set_num_threads(args.torch_threads)
    except Exception:
        pass

    prompt_map, maxlen_map = load_prompt_config()
    encoder = MiniLMEncoder(HF_MINILM_SNAPSHOT)
    generator = KVPressGenerator(model_name=args.model_name)
    llmlingua_compressor = maybe_init_llmlingua(args.llmlingua_model) if args.run_llmlingua else None

    rng = np.random.default_rng(args.seed)
    task_splits: dict[str, tuple[list[LongBenchItem], list[LongBenchItem]]] = {}
    reranker_training_examples: list[tuple[np.ndarray, np.ndarray]] = []

    for dataset in args.tasks:
        raw_items = load_longbench_items(
            dataset=dataset,
            max_samples=args.max_samples_per_task + args.calibration_samples_per_task,
            max_length_words=args.max_length_words,
        )
        if not raw_items:
            continue
        order = np.arange(len(raw_items))
        rng.shuffle(order)
        shuffled = [raw_items[i] for i in order]
        calibration_items = shuffled[: args.calibration_samples_per_task]
        eval_items = shuffled[args.calibration_samples_per_task : args.calibration_samples_per_task + args.max_samples_per_task]
        task_splits[dataset] = (calibration_items, eval_items)
        if calibration_items:
            reranker_training_examples.extend(
                collect_reranker_examples(
                    dataset=dataset,
                    items=calibration_items,
                    encoder=encoder,
                    chunk_words=args.chunk_words,
                )
            )

    learned_reranker = fit_learned_reranker(reranker_training_examples)
    if learned_reranker is not None:
        print("learned_reranker weights", np.round(learned_reranker.weights, 3).tolist(), "bias", round(learned_reranker.bias, 3))
    router_training_examples: list[tuple[np.ndarray, float]] = []
    if learned_reranker is not None:
        for dataset in args.tasks:
            calibration_items, _ = task_splits.get(dataset, ([], []))
            if calibration_items:
                router_training_examples.extend(
                    collect_router_examples(
                        dataset=dataset,
                        items=calibration_items,
                        encoder=encoder,
                        chunk_words=args.chunk_words,
                        retrieval_top_k=args.retrieval_top_k,
                        learned_reranker=learned_reranker,
                    )
                )
    learned_router = fit_learned_router(router_training_examples)
    if learned_router is not None:
        print("learned_router weights", np.round(learned_router.weights, 3).tolist(), "bias", round(learned_router.bias, 3))

    task_results: dict[str, dict] = {}
    peak_rss_mb = current_rss_mb()

    for dataset in args.tasks:
        calibration_items, items = task_splits.get(dataset, ([], []))
        if not items:
            print(f"{dataset:16s} skipped (no items under length cap)")
            continue
        print(
            f"{dataset:16s} loaded samples={len(items)} "
            f"min_len={min(item.length for item in items)} max_len={max(item.length for item in items)} "
            f"calibration={len(calibration_items)}"
        )
        task_results[dataset] = evaluate_task(
            dataset=dataset,
            items=items,
            encoder=encoder,
            generator=generator,
            llmlingua_compressor=llmlingua_compressor,
            learned_reranker=learned_reranker,
            learned_router=learned_router,
            prompt_template=prompt_map[dataset],
            max_new_tokens=min(args.max_new_tokens_cap, maxlen_map[dataset]),
            retrieval_top_k=args.retrieval_top_k,
            full_max_context_tokens=args.full_max_context_tokens,
            chunk_words=args.chunk_words,
            compression_ratio=args.compression_ratio,
        )
        peak_rss_mb = max(peak_rss_mb, current_rss_mb())
        print(f"{dataset:16s} results")
        for method, values in task_results[dataset]["methods"].items():
            if values["score"] is None:
                print(f"  {method:16s} unavailable")
                continue
            print(
                f"  {method:16s} score={values['score']:.2f} "
                f"avg_s={values['avg_s']:.3f} avg_ctx_tokens={values['avg_ctx_tokens']:.1f}"
            )
        gc.collect()

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": args.notes,
        "tasks": args.tasks,
        "config": {
            "seed": args.seed,
            "max_samples_per_task": args.max_samples_per_task,
            "calibration_samples_per_task": args.calibration_samples_per_task,
            "max_length_words": args.max_length_words,
            "full_max_context_tokens": args.full_max_context_tokens,
            "retrieval_top_k": args.retrieval_top_k,
            "chunk_words": args.chunk_words,
            "compression_ratio": args.compression_ratio,
            "model_name": args.model_name,
            "torch_threads": args.torch_threads,
        },
        "peak_rss_mb": peak_rss_mb,
        "task_results": task_results,
    }
    append_longbench_log(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-safe LongBench subset evaluation for Qwen + kvpress baselines.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--max-samples-per-task", type=int, default=2)
    parser.add_argument("--calibration-samples-per-task", type=int, default=0)
    parser.add_argument("--max-length-words", type=int, default=5500)
    parser.add_argument("--full-max-context-tokens", type=int, default=4096)
    parser.add_argument("--retrieval-top-k", type=int, default=4)
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--model-name", type=str, default=HF_QWEN_SMALL)
    parser.add_argument(
        "--run-llmlingua",
        action="store_true",
        help="Run LLMLingua-2 prompt compression baseline on CPU.",
    )
    parser.add_argument(
        "--llmlingua-model",
        type=str,
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    )
    parser.add_argument("--max-new-tokens-cap", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
