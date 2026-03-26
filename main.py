from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Iterable

import numpy as np


HF_MINILM_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/"
    / "fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"
)
HF_FLAN_T5_SMALL = "google/flan-t5-small"
HF_QWEN_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSONL_LOG = RESULTS_DIR / "experiment_log.jsonl"
MARKDOWN_LOG = Path(__file__).resolve().parent / "EXPERIMENT_LOG.md"


@dataclass
class QueryExample:
    text: str
    answer: str
    support_turn: int
    memory_key: str | None = None
    difficulty: str = "direct"


@dataclass
class ConversationExample:
    turns: list[str]
    queries: list[QueryExample]
    latest_turn_for_key: dict[str, int] = field(default_factory=dict)


class MiniLMEncoder:
    def __init__(self, model_path: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        all_embeddings: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                tokens = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, dim=1)
                all_embeddings.append(pooled.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)


class FlanGenerator:
    def __init__(self, model_name: str = HF_FLAN_T5_SMALL):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
        self.model.eval()

    def answer(self, context_turns: list[str], question: str, max_new_tokens: int = 16) -> str:
        prompt = (
            "Answer with only the exact short answer.\n\n"
            "Conversation excerpts:\n"
            + "\n".join(f"- {turn}" for turn in context_turns)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text.strip()


class KVPressGenerator:
    def __init__(self, model_name: str = HF_QWEN_SMALL):
        kvpress_root = Path(__file__).resolve().parent / "kvpress"
        if str(kvpress_root) not in sys.path:
            sys.path.insert(0, str(kvpress_root))

        import torch
        from kvpress import ExpectedAttentionPress, KVPressTextGenerationPipeline, StreamingLLMPress
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.ExpectedAttentionPress = ExpectedAttentionPress
        self.StreamingLLMPress = StreamingLLMPress
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.pipe = KVPressTextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

    def answer(
        self,
        context: str,
        question: str,
        press=None,
        max_new_tokens: int = 16,
    ) -> tuple[str, float]:
        prompt_question = (
            "Using only the provided conversation context, answer with only the exact short answer. "
            "Do not explain.\n"
            f"Question: {question}"
        )
        start = time.perf_counter()
        result = self.pipe(
            context,
            question=prompt_question,
            press=press,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - start
        return result["answer"].strip(), elapsed

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class FullMemory:
    def __init__(self, turns: np.ndarray):
        self.turns = turns

    def score(self, query: np.ndarray, memory_key: str | None = None, query_text: str | None = None) -> np.ndarray:
        del memory_key
        del query_text
        return self.turns @ query

    def observe(self, query: np.ndarray) -> None:
        del query


class QuantizedMemory:
    def __init__(self, turns: np.ndarray, bits: int = 4):
        self.turns = self._quantize_rows(turns, bits)

    @staticmethod
    def _quantize_rows(x: np.ndarray, bits: int) -> np.ndarray:
        scale = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
        levels = 2**bits
        q = np.round(((x / scale) + 1.0) * (levels - 1) / 2.0)
        out = (q * 2.0 / (levels - 1) - 1.0) * scale
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out / norms

    def score(self, query: np.ndarray, memory_key: str | None = None, query_text: str | None = None) -> np.ndarray:
        del memory_key
        del query_text
        return self.turns @ query

    def observe(self, query: np.ndarray) -> None:
        del query


class TurboProxyMemory:
    def __init__(self, turns: np.ndarray, bits: int = 3, seed: int = 0):
        dim = turns.shape[1]
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(dim)
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)

        turns_rot = turns[:, self.perm] * self.signs
        self.approx_rot = self._quantize_rows(turns_rot, bits)
        residual = turns_rot - self.approx_rot
        self.residual_scale = np.linalg.norm(residual, axis=1, keepdims=True) / np.sqrt(dim)
        self.residual_sign = np.where(residual >= 0.0, 1.0, -1.0).astype(np.float32)

        approx = np.empty_like(self.approx_rot)
        approx[:, self.perm] = self.approx_rot * self.signs
        self.approx = approx
        self.dim = dim

    @staticmethod
    def _quantize_rows(x: np.ndarray, bits: int) -> np.ndarray:
        scale = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
        levels = 2**bits
        q = np.round(((x / scale) + 1.0) * (levels - 1) / 2.0)
        return (q * 2.0 / (levels - 1) - 1.0) * scale

    def score(self, query: np.ndarray, memory_key: str | None = None, query_text: str | None = None) -> np.ndarray:
        del memory_key
        del query_text
        query_rot = query[self.perm] * self.signs
        base = self.approx @ query
        correction = (self.residual_scale[:, 0] * (self.residual_sign @ query_rot)) / np.sqrt(self.dim)
        return base + correction

    def observe(self, query: np.ndarray) -> None:
        del query


class QueryAwareSparseMemory:
    def __init__(
        self,
        turns: np.ndarray,
        initial_queries: np.ndarray,
        ledger_turns: dict[str, int] | None = None,
        rank: int = 8,
        sparse_k: int = 2,
        query_decay: float = 0.92,
        recency_alpha: float = 0.015,
        ledger_boost: float = 2.5,
    ):
        self.turns = turns
        self.rank = rank
        self.sparse_k = sparse_k
        self.query_decay = query_decay
        self.recency_alpha = recency_alpha
        self.ledger_boost = ledger_boost
        self.ledger_turns = ledger_turns or {}
        self.c_x = (turns.T @ turns) / max(len(turns), 1)
        self.recency = 1.0 + recency_alpha * np.arange(len(turns), dtype=np.float32)
        dim = turns.shape[1]
        if len(initial_queries) == 0:
            self.c_q = np.eye(dim, dtype=np.float32)
        else:
            self.c_q = (initial_queries.T @ initial_queries) / len(initial_queries)
        self._rebuild()

    def _rebuild(self) -> None:
        score_matrix = self.c_x @ self.c_q @ self.c_x
        eigvals, eigvecs = np.linalg.eigh((score_matrix + score_matrix.T) / 2.0)
        basis = eigvecs[:, np.argsort(eigvals)[-self.rank :]]
        coeffs = self.turns @ basis
        approx = coeffs @ basis.T
        residual = self.turns - approx
        sparse = np.zeros_like(residual)
        idx = np.argpartition(np.abs(residual), -self.sparse_k, axis=1)[:, -self.sparse_k :]
        rows = np.arange(len(residual))[:, None]
        sparse[rows, idx] = residual[rows, idx]
        self.basis = basis
        self.coeffs = coeffs
        self.sparse = sparse

    def score(self, query: np.ndarray, memory_key: str | None = None, query_text: str | None = None) -> np.ndarray:
        del query_text
        scores = (self.coeffs @ (self.basis.T @ query) + self.sparse @ query) * self.recency
        if memory_key and memory_key in self.ledger_turns:
            scores[self.ledger_turns[memory_key]] += self.ledger_boost
        return scores

    def observe(self, query: np.ndarray) -> None:
        self.c_q = self.query_decay * self.c_q + (1.0 - self.query_decay) * np.outer(query, query)
        self._rebuild()


class QueryAwareGraphMemory(QueryAwareSparseMemory):
    def __init__(
        self,
        turns_text: list[str],
        turns: np.ndarray,
        initial_queries: np.ndarray,
        ledger_turns: dict[str, int] | None = None,
        rank: int = 8,
        sparse_k: int = 2,
        query_decay: float = 0.92,
        recency_alpha: float = 0.015,
        ledger_boost: float = 1.2,
        graph_alpha: float = 0.9,
        top_seed_k: int = 3,
        bridge_scale: float = 1.6,
        subject_scale: float = 2.5,
    ):
        self.turns_text = turns_text
        self.graph_alpha = graph_alpha
        self.top_seed_k = top_seed_k
        self.bridge_scale = bridge_scale
        self.subject_scale = subject_scale
        super().__init__(
            turns=turns,
            initial_queries=initial_queries,
            ledger_turns=ledger_turns,
            rank=rank,
            sparse_k=sparse_k,
            query_decay=query_decay,
            recency_alpha=recency_alpha,
            ledger_boost=ledger_boost,
        )
        self.graph = self._build_graph(turns_text)

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        lowered = text.lower()
        entities = set(re.findall(r"inv-\d+", lowered))
        entities.update(re.findall(r"\b[a-z]{4,}\b", lowered))
        stop = {
            "user",
            "assistant",
            "current",
            "latest",
            "final",
            "active",
            "client",
            "project",
            "note",
            "notes",
            "draft",
            "update",
            "earlier",
            "correction",
            "launch",
            "check",
            "checkin",
            "dashboard",
            "theme",
            "archive",
            "documents",
            "finance",
            "reference",
            "working",
            "should",
            "still",
            "there",
            "remember",
            "lightweight",
            "aligned",
            "timeline",
            "weekly",
            "review",
            "sprint",
            "scope",
            "control",
            "prioritize",
            "recent",
            "conflict",
        }
        return {entity for entity in entities if entity not in stop}

    @staticmethod
    def _extract_subject_entities(text: str) -> set[str]:
        lowered = text.lower()
        subjects = set(re.findall(r"\bclient\s+([a-z0-9\-]+)", lowered))
        subjects.update(re.findall(r"\bproject\s+([a-z0-9\-]+)", lowered))
        subjects.update(re.findall(r"\bfor\s+project\s+([a-z0-9\-]+)", lowered))
        subjects.update(re.findall(r"\bfor\s+client\s+([a-z0-9\-]+)", lowered))
        return subjects

    def _build_graph(self, turns_text: list[str]) -> np.ndarray:
        entity_sets = [self._extract_entities(turn) for turn in turns_text]
        self.entity_sets = entity_sets
        vocab: dict[str, int] = {}
        for entities in entity_sets:
            for entity in entities:
                vocab[entity] = vocab.get(entity, 0) + 1

        n = len(turns_text)
        graph = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                shared = entity_sets[i] & entity_sets[j]
                if not shared:
                    continue
                weight = sum(1.0 / (1.0 + vocab[entity]) for entity in shared)
                if weight <= 0.0:
                    continue
                graph[i, j] = weight
                graph[j, i] = weight

        row_sums = graph.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        self.entity_idf = {entity: 1.0 / (1.0 + count) for entity, count in vocab.items()}
        self.update_bias = np.array(
            [
                1.35
                if any(marker in turn.lower() for marker in ["current", "latest", "final", "active", "corrected", "live"])
                else 1.0
                for turn in turns_text
            ],
            dtype=np.float32,
        )
        self.type_bias = {
            "codename": np.array(["codename" in turn.lower() for turn in turns_text], dtype=np.float32),
            "invoice": np.array(
                [("reference" in turn.lower()) or ("inv-" in turn.lower()) or ("finance" in turn.lower()) for turn in turns_text],
                dtype=np.float32,
            ),
            "color": np.array(
                [("color" in turn.lower()) or ("theme" in turn.lower()) for turn in turns_text],
                dtype=np.float32,
            ),
            "city": np.array(
                [("archive" in turn.lower()) or ("store the documents" in turn.lower()) or ("documents in" in turn.lower()) for turn in turns_text],
                dtype=np.float32,
            ),
            "launch": np.array(
                [("launch" in turn.lower()) or ("check-in day" in turn.lower()) for turn in turns_text],
                dtype=np.float32,
            ),
            "alias": np.array(
                [(" alias " in f" {turn.lower()} ") or ("maps to project" in turn.lower()) or ("refers to client" in turn.lower()) for turn in turns_text],
                dtype=np.float32,
            ),
        }
        return graph / row_sums

    @staticmethod
    def _infer_target_type(query_text: str | None) -> str | None:
        if not query_text:
            return None
        text = query_text.lower()
        if "codename" in text:
            return "codename"
        if "invoice" in text or "finance reference" in text or "live reference" in text or "reference" in text:
            return "invoice"
        if "color" in text or "theme" in text:
            return "color"
        if "archive" in text or "city" in text or "documents" in text:
            return "city"
        if "launch" in text or "day" in text or "check-in" in text:
            return "launch"
        return None

    @staticmethod
    def _infer_source_type(query_text: str | None) -> str | None:
        if not query_text:
            return None
        text = query_text.lower()
        if "alias" in text or "tracked as" in text or "behind alias" in text or "tracked under alias" in text:
            return "alias"
        if "whose finance reference" in text or "with live reference" in text or "invoice code belongs" in text:
            return "invoice"
        if "whose dashboard color" in text or "uses the " in text and "theme" in text:
            return "color"
        return None

    def score(self, query: np.ndarray, memory_key: str | None = None, query_text: str | None = None) -> np.ndarray:
        base_scores = super().score(query, memory_key)
        seed_signal = np.maximum(base_scores, 0.0)
        entity_overlap = np.zeros_like(base_scores)
        query_entities: set[str] = set()
        if query_text:
            query_entities = self._extract_entities(query_text)
            if query_entities:
                entity_overlap = np.array(
                    [
                        sum(self.entity_idf.get(entity, 0.0) for entity in query_entities & turn_entities)
                        for turn_entities in self.entity_sets
                    ],
                    dtype=np.float32,
                )
                seed_signal = seed_signal + 2.0 * entity_overlap
        seed_idx = np.argsort(-seed_signal)[: self.top_seed_k]
        seeds = np.zeros_like(base_scores)
        seeds[seed_idx] = seed_signal[seed_idx]
        target_type = self._infer_target_type(query_text)
        source_type = self._infer_source_type(query_text)
        propagated = (self.graph.T @ seeds) * self.update_bias

        bridge_bonus = np.zeros_like(base_scores)
        if source_type in {"invoice", "color"} and target_type and source_type != target_type:
            source_seed_signal = entity_overlap * (0.25 + 0.75 * self.type_bias[source_type])
            source_idx = np.argsort(-source_seed_signal)[:2]
            source_seeds = np.zeros_like(base_scores)
            source_seeds[source_idx] = source_seed_signal[source_idx]
            bridge_scores = (self.graph.T @ source_seeds) * self.update_bias
            bridge_bonus = self.bridge_scale * bridge_scores * self.type_bias[target_type]
            subject_entities = set()
            for idx in source_idx:
                subject_entities.update(self._extract_subject_entities(self.turns_text[idx]))
            if subject_entities:
                subject_overlap = np.array(
                    [
                        sum(1.0 for entity in subject_entities if entity in turn_text.lower())
                        for turn_text in self.turns_text
                    ],
                    dtype=np.float32,
                )
                bridge_bonus += self.subject_scale * subject_overlap * self.type_bias[target_type]

        if target_type:
            target_bias = 1.0 + 0.75 * self.type_bias[target_type] - 0.35 * self.type_bias["alias"]
            target_bias = np.clip(target_bias, 0.7, None)
        else:
            target_bias = 1.0

        source_suppress = 1.0
        if source_type in {"invoice", "color"} and target_type and source_type != target_type:
            source_suppress = np.clip(1.0 - 0.35 * self.type_bias[source_type], 0.65, None)

        return (base_scores + self.graph_alpha * propagated + bridge_bonus) * self.recency * target_bias * source_suppress


def build_dataset(
    rng: random.Random,
    conversation_count: int,
    turns_per_conversation: int,
    queries_per_conversation: int,
) -> list[ConversationExample]:
    projects = [
        "atlas",
        "beacon",
        "comet",
        "delta",
        "ember",
        "fjord",
        "glyph",
        "harbor",
        "ion",
        "juniper",
        "keystone",
        "lattice",
    ]
    clients = [
        "Acme",
        "Borealis",
        "Cinder",
        "Drift",
        "Eon",
        "Fable",
        "Granite",
        "Helix",
        "Ingot",
        "Jasper",
        "Kite",
        "Lyric",
    ]
    colors = [
        "amber",
        "blue",
        "coral",
        "emerald",
        "indigo",
        "mint",
        "orange",
        "scarlet",
        "silver",
        "teal",
    ]
    cities = [
        "Austin",
        "Boston",
        "Chicago",
        "Denver",
        "Miami",
        "Portland",
        "Seattle",
        "Toronto",
    ]
    codenames = [
        "falcon",
        "helix",
        "marble",
        "nova",
        "onyx",
        "prism",
        "quartz",
        "vector",
        "willow",
        "zephyr",
    ]
    client_aliases = [
        "maple",
        "orbit",
        "raven",
        "sable",
        "tango",
        "vivid",
        "wisp",
        "yonder",
        "zen",
        "cedar",
    ]
    project_aliases = [
        "aurora",
        "biscuit",
        "cobalt",
        "drizzle",
        "emberline",
        "feather",
        "glint",
        "horizon",
        "iris",
        "jolt",
    ]

    filler_templates = [
        "User: I am still thinking about the rollout timeline for {project}.",
        "Assistant: We can keep the milestone notes short and focused for {project}.",
        "User: The weekly review for {client} should stay lightweight.",
        "Assistant: I will remember that the design notes are still in draft form.",
        "User: The action items for the current sprint need tighter scope control.",
        "Assistant: I can keep the summary concise unless you ask for more detail.",
        "User: Please make sure the notes stay aligned with the deployment plan.",
        "Assistant: I will keep the chronology straight when we revisit this later.",
        "User: The earlier draft should not be treated as final until I confirm the change.",
        "Assistant: I will prioritize the most recent correction when the notes conflict.",
    ]

    dataset: list[ConversationExample] = []
    for _ in range(conversation_count):
        project = rng.choice(projects)
        client = rng.choice(clients)
        color = rng.choice(colors)
        city = rng.choice(cities)
        old_color = rng.choice([c for c in colors if c != color])
        old_city = rng.choice([c for c in cities if c != city])
        codename = rng.choice(codenames)
        old_codename = rng.choice([c for c in codenames if c != codename])
        invoice = f"INV-{rng.randint(1000, 9999)}"
        old_invoice = f"INV-{rng.randint(1000, 9999)}"
        launch_day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        old_day = rng.choice([d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"] if d != launch_day])
        client_alias = rng.choice(client_aliases)
        project_alias = rng.choice(project_aliases)

        fact_specs = [
            {
                "key": f"codename::{client}",
                "initial": f"User: Initial note for client {client}: the working codename is {old_codename}.",
                "update": f"User: Correction for client {client}: the active codename is {codename} now.",
                "question": rng.choice(
                    [
                        f"What is the current codename for client {client}?",
                        f"Which codename should I use now for client {client}?",
                        f"What did I most recently rename client {client}'s codename to?",
                    ]
                ),
                "answer": codename,
            },
            {
                "key": f"color::{project}",
                "initial": f"User: Earlier draft for project {project}: the dashboard theme was {old_color}.",
                "update": f"User: Update for project {project}: the current dashboard color is {color}.",
                "question": rng.choice(
                    [
                        f"What is the latest dashboard color for project {project}?",
                        f"Which theme color is current for {project} now?",
                        f"What color did I settle on for the {project} dashboard?",
                    ]
                ),
                "answer": color,
            },
            {
                "key": f"city::{project}",
                "initial": f"User: Old archive plan for {project}: keep the documents in {old_city}.",
                "update": f"User: Final archive update for {project}: store the documents in {city}.",
                "question": rng.choice(
                    [
                        f"Where should the archive for {project} live now?",
                        f"Which city is the current archive location for project {project}?",
                        f"What is the final document city for {project}?",
                    ]
                ),
                "answer": city,
            },
            {
                "key": f"invoice::{client}",
                "initial": f"User: Earlier finance note for {client}: use reference {old_invoice}.",
                "update": f"User: Corrected finance note for {client}: the live reference is {invoice}.",
                "question": rng.choice(
                    [
                        f"What is the current finance reference for client {client}?",
                        f"Which invoice code is active now for {client}?",
                        f"What reference did I finally approve for client {client}?",
                    ]
                ),
                "answer": invoice,
            },
            {
                "key": f"launch::{project}",
                "initial": f"User: The first launch check-in plan for {project} was {old_day}.",
                "update": f"User: Latest launch note for {project}: the real check-in day is {launch_day}.",
                "question": rng.choice(
                    [
                        f"What day is the current launch check-in for {project}?",
                        f"Which day did I finally choose for the {project} launch check-in?",
                        f"What is the latest check-in day for project {project}?",
                    ]
                ),
                "answer": launch_day,
            },
        ]
        relation_specs = [
            {
                "key": f"alias-client::{client_alias}",
                "text": f"User: In the ops notes, alias {client_alias} refers to client {client}.",
            },
            {
                "key": f"alias-project::{project_alias}",
                "text": f"User: In the migration tracker, alias {project_alias} maps to project {project}.",
            },
        ]

        turns: list[str] = []
        latest_turn_for_key: dict[str, int] = {}
        scheduled_facts: list[tuple[int, dict[str, str], str]] = []
        relation_turn_for_key: dict[str, int] = {}
        candidate_slots = sorted(
            rng.sample(range(4, turns_per_conversation - 4), k=len(fact_specs) * 2 + len(relation_specs))
        )
        for idx, spec in enumerate(fact_specs):
            scheduled_facts.append((candidate_slots[idx * 2], spec, "initial"))
            scheduled_facts.append((candidate_slots[idx * 2 + 1], spec, "update"))
        relation_offset = len(fact_specs) * 2
        for idx, spec in enumerate(relation_specs):
            scheduled_facts.append((candidate_slots[relation_offset + idx], spec, "text"))
        scheduled_facts.sort(key=lambda item: item[0])

        fact_idx = 0
        for turn_idx in range(turns_per_conversation):
            if fact_idx < len(scheduled_facts) and turn_idx == scheduled_facts[fact_idx][0]:
                _, spec, phase = scheduled_facts[fact_idx]
                turns.append(spec[phase])
                if phase == "update":
                    latest_turn_for_key[spec["key"]] = len(turns) - 1
                if phase == "text":
                    relation_turn_for_key[spec["key"]] = len(turns) - 1
                fact_idx += 1
                continue
            template = rng.choice(filler_templates)
            turns.append(template.format(project=project, client=client))

        direct_queries = [
            QueryExample(
                text=spec["question"],
                answer=spec["answer"],
                support_turn=latest_turn_for_key[spec["key"]],
                memory_key=spec["key"],
                difficulty="direct",
            )
            for spec in fact_specs
        ]
        hard_queries = [
            QueryExample(
                text=rng.choice(
                    [
                        f"What is the current codename for the client tracked under alias {client_alias}?",
                        f"Which codename belongs now to the client behind alias {client_alias}?",
                    ]
                ),
                answer=codename,
                support_turn=latest_turn_for_key[f"codename::{client}"],
                memory_key=None,
                difficulty="alias",
            ),
            QueryExample(
                text=rng.choice(
                    [
                        f"What is the current finance reference for the client tracked under alias {client_alias}?",
                        f"Which invoice code belongs now to the client behind alias {client_alias}?",
                    ]
                ),
                answer=invoice,
                support_turn=latest_turn_for_key[f"invoice::{client}"],
                memory_key=None,
                difficulty="alias",
            ),
            QueryExample(
                text=rng.choice(
                    [
                        f"What is the current dashboard color for the project tracked as {project_alias}?",
                        f"Which theme color is current for the project behind alias {project_alias}?",
                    ]
                ),
                answer=color,
                support_turn=latest_turn_for_key[f"color::{project}"],
                memory_key=None,
                difficulty="alias",
            ),
            QueryExample(
                text=rng.choice(
                    [
                        f"What is the current codename for the client whose finance reference is {invoice}?",
                        f"Which codename should I use for the client with live reference {invoice}?",
                    ]
                ),
                answer=codename,
                support_turn=latest_turn_for_key[f"codename::{client}"],
                memory_key=None,
                difficulty="multi_hop",
            ),
            QueryExample(
                text=rng.choice(
                    [
                        f"What is the latest archive city for the project whose dashboard color is {color}?",
                        f"Which city stores documents for the project that now uses the {color} theme?",
                    ]
                ),
                answer=city,
                support_turn=latest_turn_for_key[f"city::{project}"],
                memory_key=None,
                difficulty="multi_hop",
            ),
            QueryExample(
                text=rng.choice(
                    [
                        f"What day is the current launch check-in for the project tracked as {project_alias}?",
                        f"Which launch day belongs to the project behind alias {project_alias}?",
                    ]
                ),
                answer=launch_day,
                support_turn=latest_turn_for_key[f"launch::{project}"],
                memory_key=None,
                difficulty="alias",
            ),
        ]
        queries = direct_queries + hard_queries
        rng.shuffle(queries)
        dataset.append(
            ConversationExample(
                turns=turns,
                queries=queries[:queries_per_conversation],
                latest_turn_for_key=latest_turn_for_key,
            )
        )
    return dataset


def top_k_hits(scores: np.ndarray, target_index: int, k: int) -> bool:
    return bool(target_index in np.argsort(-scores)[:k])


def evaluate_backend_detailed(
    dataset: list[ConversationExample],
    turn_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    backend_name: str,
    train_query_bank: np.ndarray,
    graph_params: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    metrics = {
        "top1": 0,
        "top8": 0,
        "top16": 0,
    }
    by_difficulty: dict[str, dict[str, int]] = {}
    total = 0

    for conv, conv_turns, conv_queries in zip(dataset, turn_embeddings, query_embeddings):
        backend = build_backend(backend_name, conv, conv_turns, train_query_bank, graph_params=graph_params)

        for query, query_vec in zip(conv.queries, conv_queries):
            scores = backend.score(query_vec, query.memory_key, query.text)
            top1 = int(int(np.argmax(scores)) == query.support_turn)
            top8 = int(top_k_hits(scores, query.support_turn, 8))
            top16 = int(top_k_hits(scores, query.support_turn, 16))
            metrics["top1"] += top1
            metrics["top8"] += top8
            metrics["top16"] += top16
            diff_metrics = by_difficulty.setdefault(
                query.difficulty,
                {"top1": 0, "top8": 0, "top16": 0, "count": 0},
            )
            diff_metrics["top1"] += top1
            diff_metrics["top8"] += top8
            diff_metrics["top16"] += top16
            diff_metrics["count"] += 1
            total += 1
            backend.observe(query_vec)

    overall = {name: value / max(total, 1) for name, value in metrics.items()}
    by_difficulty_norm = {
        difficulty: {
            "top1": values["top1"] / values["count"],
            "top8": values["top8"] / values["count"],
            "top16": values["top16"] / values["count"],
            "count": values["count"],
        }
        for difficulty, values in by_difficulty.items()
    }
    return {"overall": overall, "by_difficulty": by_difficulty_norm}


def evaluate_backend(
    dataset: list[ConversationExample],
    turn_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    backend_name: str,
    train_query_bank: np.ndarray,
    graph_params: dict[str, float] | None = None,
) -> dict[str, float]:
    return evaluate_backend_detailed(
        dataset=dataset,
        turn_embeddings=turn_embeddings,
        query_embeddings=query_embeddings,
        backend_name=backend_name,
        train_query_bank=train_query_bank,
        graph_params=graph_params,
    )["overall"]


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^(the current|current|the final|final)\s+", "", text)
    text = re.sub(r"^(answer:)\s*", "", text)
    text = re.sub(r"[^\w\- ]+", " ", text)
    return " ".join(text.split())


def is_answer_correct(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    target = normalize_answer(gold)
    if pred == target:
        return True
    return re.search(rf"(?<!\w){re.escape(target)}(?!\w)", pred) is not None


def top_indices(scores: np.ndarray, k: int) -> list[int]:
    return np.argsort(-scores)[:k].tolist()


def build_backend(
    backend_name: str,
    conversation: ConversationExample,
    turns: np.ndarray,
    train_query_bank: np.ndarray,
    graph_params: dict[str, float] | None = None,
):
    graph_params = graph_params or {}
    if backend_name == "full":
        return FullMemory(turns)
    if backend_name == "q4":
        return QuantizedMemory(turns, bits=4)
    if backend_name == "turbo_proxy":
        return TurboProxyMemory(turns, bits=3, seed=17)
    if backend_name == "qaware_sparse":
        return QueryAwareSparseMemory(
            turns,
            initial_queries=train_query_bank,
            ledger_turns=conversation.latest_turn_for_key,
            rank=8,
            sparse_k=2,
            query_decay=0.92,
        )
    if backend_name == "qaware_graph":
        return QueryAwareGraphMemory(
            turns_text=conversation.turns,
            turns=turns,
            initial_queries=train_query_bank,
            ledger_turns=conversation.latest_turn_for_key,
            rank=8,
            sparse_k=2,
            query_decay=0.92,
            graph_alpha=1.25,
            top_seed_k=4,
            bridge_scale=graph_params.get("bridge_scale", 1.6),
            subject_scale=graph_params.get("subject_scale", 2.5),
        )
    raise ValueError(f"Unknown backend: {backend_name}")


def conversation_to_context(turns: list[str]) -> str:
    return "\n".join(turns)


def select_qagraph_turns(
    backend: QueryAwareGraphMemory,
    scores: np.ndarray,
    query_text: str,
    requested_k: int,
) -> list[int]:
    candidate_pool = top_indices(scores, max(8, requested_k))
    target_type = backend._infer_target_type(query_text)
    source_type = backend._infer_source_type(query_text)

    selected: list[int] = []
    if target_type:
        target_hits = [idx for idx in candidate_pool if backend.type_bias[target_type][idx] > 0]
        if target_hits:
            selected.append(target_hits[0])

    if not selected:
        selected.append(candidate_pool[0])

    if source_type and source_type != target_type:
        source_hits = [idx for idx in candidate_pool if backend.type_bias[source_type][idx] > 0 and idx not in selected]
        if source_hits:
            selected.insert(0, source_hits[0])
    elif source_type == "alias":
        alias_hits = [idx for idx in candidate_pool if backend.type_bias["alias"][idx] > 0 and idx not in selected]
        if alias_hits:
            selected.insert(0, alias_hits[0])

    for idx in candidate_pool:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max(requested_k, 2):
            break
    return selected[: max(requested_k, 2 if source_type else requested_k)]


def calibrate_graph_params(
    dataset: list[ConversationExample],
    turn_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    train_query_bank: np.ndarray,
) -> dict[str, float]:
    best = {"bridge_scale": 1.6, "subject_scale": 2.5}
    best_score = (-1.0, -1.0)
    bridge_values = [0.8, 1.2, 1.6, 2.0, 2.4]
    subject_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    for bridge_scale in bridge_values:
        for subject_scale in subject_values:
            result = evaluate_backend_detailed(
                dataset=dataset,
                turn_embeddings=turn_embeddings,
                query_embeddings=query_embeddings,
                backend_name="qaware_graph",
                train_query_bank=train_query_bank,
                graph_params={"bridge_scale": bridge_scale, "subject_scale": subject_scale},
            )
            by_diff = result["by_difficulty"]
            multi = by_diff.get("multi_hop", {}).get("top1", 0.0)
            overall = result["overall"]["top1"]
            candidate_score = (multi, overall)
            if candidate_score > best_score:
                best_score = candidate_score
                best = {"bridge_scale": bridge_scale, "subject_scale": subject_scale}
    return best


def evaluate_generator(
    dataset: list[ConversationExample],
    turn_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    train_query_bank: np.ndarray,
    max_queries: int,
) -> None:
    generator = FlanGenerator()
    counts = {"full_context": 0, "q4_top1": 0, "turbo_proxy_top1": 0, "qaware_top1": 0, "qagraph_top1": 0}
    total = 0

    for conv, conv_turns, conv_queries in zip(dataset, turn_embeddings, query_embeddings):
        backends = {
            "q4_top1": build_backend("q4", conv, conv_turns, train_query_bank),
            "turbo_proxy_top1": build_backend("turbo_proxy", conv, conv_turns, train_query_bank),
            "qaware_top1": build_backend("qaware_sparse", conv, conv_turns, train_query_bank),
            "qagraph_top1": build_backend("qaware_graph", conv, conv_turns, train_query_bank),
        }
        for query, query_vec in zip(conv.queries, conv_queries):
            answer_full = generator.answer(conv.turns, query.text)
            counts["full_context"] += int(is_answer_correct(answer_full, query.answer))

            for name, backend in backends.items():
                idx = top_indices(backend.score(query_vec, query.memory_key, query.text), 1)
                retrieved = [conv.turns[i] for i in idx]
                answer = generator.answer(retrieved, query.text)
                counts[name] += int(is_answer_correct(answer, query.answer))
                backend.observe(query_vec)

            total += 1
            if total >= max_queries:
                print(
                        f"full_context   exact_match={counts['full_context'] / total:.4f}\n"
                        f"q4_top1        exact_match={counts['q4_top1'] / total:.4f}\n"
                        f"turbo_proxy    exact_match={counts['turbo_proxy_top1'] / total:.4f}\n"
                        f"qaware_top1    exact_match={counts['qaware_top1'] / total:.4f}\n"
                        f"qagraph_top1   exact_match={counts['qagraph_top1'] / total:.4f}"
                    )
                return

    if total:
        print(
            f"full_context   exact_match={counts['full_context'] / total:.4f}\n"
            f"q4_top1        exact_match={counts['q4_top1'] / total:.4f}\n"
            f"turbo_proxy    exact_match={counts['turbo_proxy_top1'] / total:.4f}\n"
            f"qaware_top1    exact_match={counts['qaware_top1'] / total:.4f}\n"
            f"qagraph_top1   exact_match={counts['qagraph_top1'] / total:.4f}"
        )


def evaluate_kvpress_generator(
    dataset: list[ConversationExample],
    turn_embeddings: list[np.ndarray],
    query_embeddings: list[np.ndarray],
    train_query_bank: np.ndarray,
    model_name: str,
    max_queries: int,
    qaware_top_k: int,
    compression_ratio: float,
    graph_params: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    generator = KVPressGenerator(model_name=model_name)
    counts = {
        "causal_full": 0,
        "streaming_press": 0,
        "expected_press": 0,
        "qaware_topk": 0,
        "qagraph_topk": 0,
    }
    elapsed = {name: 0.0 for name in counts}
    prompt_tokens = {name: 0 for name in counts}
    total = 0

    streaming_press = generator.StreamingLLMPress(compression_ratio=compression_ratio)
    expected_press = generator.ExpectedAttentionPress(
        compression_ratio=compression_ratio,
        n_future_positions=64,
        n_sink=4,
    )

    for conv, conv_turns, conv_queries in zip(dataset, turn_embeddings, query_embeddings):
        qaware = build_backend("qaware_sparse", conv, conv_turns, train_query_bank, graph_params=graph_params)
        qagraph = build_backend("qaware_graph", conv, conv_turns, train_query_bank, graph_params=graph_params)
        full_context = conversation_to_context(conv.turns)
        for query, query_vec in zip(conv.queries, conv_queries):
            answer, dt = generator.answer(full_context, query.text)
            counts["causal_full"] += int(is_answer_correct(answer, query.answer))
            elapsed["causal_full"] += dt
            prompt_tokens["causal_full"] += generator.token_count(full_context)

            answer, dt = generator.answer(full_context, query.text, press=streaming_press)
            counts["streaming_press"] += int(is_answer_correct(answer, query.answer))
            elapsed["streaming_press"] += dt
            prompt_tokens["streaming_press"] += generator.token_count(full_context)

            answer, dt = generator.answer(full_context, query.text, press=expected_press)
            counts["expected_press"] += int(is_answer_correct(answer, query.answer))
            elapsed["expected_press"] += dt
            prompt_tokens["expected_press"] += generator.token_count(full_context)

            idx = top_indices(qaware.score(query_vec, query.memory_key, query.text), qaware_top_k)
            retrieved_context = conversation_to_context([conv.turns[i] for i in idx])
            answer, dt = generator.answer(retrieved_context, query.text)
            counts["qaware_topk"] += int(is_answer_correct(answer, query.answer))
            elapsed["qaware_topk"] += dt
            prompt_tokens["qaware_topk"] += generator.token_count(retrieved_context)
            qaware.observe(query_vec)

            qagraph_scores = qagraph.score(query_vec, query.memory_key, query.text)
            idx = select_qagraph_turns(qagraph, qagraph_scores, query.text, qaware_top_k)
            retrieved_context = conversation_to_context([conv.turns[i] for i in idx])
            answer, dt = generator.answer(retrieved_context, query.text)
            counts["qagraph_topk"] += int(is_answer_correct(answer, query.answer))
            elapsed["qagraph_topk"] += dt
            prompt_tokens["qagraph_topk"] += generator.token_count(retrieved_context)
            qagraph.observe(query_vec)

            total += 1
            if total >= max_queries:
                break
        if total >= max_queries:
            break

    if not total:
        return {}

    results: dict[str, dict[str, float]] = {}
    for name in ["causal_full", "streaming_press", "expected_press", "qaware_topk", "qagraph_topk"]:
        results[name] = {
            "exact_match": counts[name] / total,
            "avg_s": elapsed[name] / total,
            "avg_ctx_tokens": prompt_tokens[name] / total,
        }
        print(
            f"{name:16s} "
            f"exact_match={results[name]['exact_match']:.4f} "
            f"avg_s={results[name]['avg_s']:.3f} "
            f"avg_ctx_tokens={results[name]['avg_ctx_tokens']:.1f}"
        )
    return results


def ensure_log_paths() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def append_experiment_log(
    args: argparse.Namespace,
    retrieval_results: dict[str, dict[str, dict[str, float]]],
    kvpress_results: dict[str, dict[str, float]] | None,
) -> None:
    ensure_log_paths()
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": args.notes,
        "config": {
            "seed": args.seed,
            "train_conversations": args.train_conversations,
            "test_conversations": args.test_conversations,
            "turns_per_conversation": args.turns_per_conversation,
            "queries_per_conversation": args.queries_per_conversation,
            "kvpress_queries": args.kvpress_queries,
            "kvpress_top_k": args.kvpress_top_k,
            "kvpress_compression_ratio": args.kvpress_compression_ratio,
            "kvpress_model": args.kvpress_model,
        },
        "retrieval": retrieval_results,
        "kvpress_generation": kvpress_results,
    }
    with JSONL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    best_retrieval = max(retrieval_results.items(), key=lambda item: item[1]["overall"]["top1"])
    lines = [
        f"## {record['timestamp']}",
        f"- Notes: {args.notes or 'n/a'}",
        f"- Config: train={args.train_conversations}, test={args.test_conversations}, turns={args.turns_per_conversation}, queries={args.queries_per_conversation}",
        f"- Best retrieval: `{best_retrieval[0]}` top1={best_retrieval[1]['overall']['top1']:.4f}, top8={best_retrieval[1]['overall']['top8']:.4f}",
    ]
    if kvpress_results:
        best_generation = max(kvpress_results.items(), key=lambda item: item[1]["exact_match"])
        lines.append(
            f"- Best generation: `{best_generation[0]}` exact_match={best_generation[1]['exact_match']:.4f}, "
            f"avg_s={best_generation[1]['avg_s']:.3f}, avg_ctx_tokens={best_generation[1]['avg_ctx_tokens']:.1f}"
        )
    lines.append("")
    with MARKDOWN_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def encode_dataset(
    encoder: MiniLMEncoder,
    dataset: list[ConversationExample],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    turn_embeddings: list[np.ndarray] = []
    query_embeddings: list[np.ndarray] = []
    all_queries: list[np.ndarray] = []

    for example in dataset:
        conv_turn_emb = encoder.encode(example.turns, batch_size=32)
        conv_query_emb = encoder.encode([q.text for q in example.queries], batch_size=16)
        turn_embeddings.append(conv_turn_emb.astype(np.float32))
        query_embeddings.append(conv_query_emb.astype(np.float32))
        all_queries.extend(conv_query_emb.astype(np.float32))

    if all_queries:
        query_bank = np.stack(all_queries)
    else:
        query_bank = np.empty((0, turn_embeddings[0].shape[1]), dtype=np.float32)
    return turn_embeddings, query_embeddings, query_bank


def run_experiment(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    train_data = build_dataset(
        rng=rng,
        conversation_count=args.train_conversations,
        turns_per_conversation=args.turns_per_conversation,
        queries_per_conversation=args.queries_per_conversation,
    )
    test_data = build_dataset(
        rng=rng,
        conversation_count=args.test_conversations,
        turns_per_conversation=args.turns_per_conversation,
        queries_per_conversation=args.queries_per_conversation,
    )

    encoder = MiniLMEncoder(HF_MINILM_SNAPSHOT)
    train_turns, train_queries, train_query_bank = encode_dataset(encoder, train_data)
    graph_params = calibrate_graph_params(
        dataset=train_data,
        turn_embeddings=train_turns,
        query_embeddings=train_queries,
        train_query_bank=train_query_bank,
    )
    print(
        f"graph_params   bridge_scale={graph_params['bridge_scale']:.2f} "
        f"subject_scale={graph_params['subject_scale']:.2f}"
    )
    test_turns, test_queries, _ = encode_dataset(encoder, test_data)

    retrieval_results: dict[str, dict[str, dict[str, float]]] = {}
    for name in ["full", "q4", "turbo_proxy", "qaware_sparse", "qaware_graph"]:
        result = evaluate_backend_detailed(
            dataset=test_data,
            turn_embeddings=test_turns,
            query_embeddings=test_queries,
            backend_name=name,
            train_query_bank=train_query_bank,
            graph_params=graph_params,
        )
        retrieval_results[name] = result
        scores = result["overall"]
        print(
            f"{name:14s} "
            f"top1={scores['top1']:.4f} "
            f"top8={scores['top8']:.4f} "
            f"top16={scores['top16']:.4f}"
        )

    if args.run_generator:
        evaluate_generator(
            dataset=test_data,
            turn_embeddings=test_turns,
            query_embeddings=test_queries,
            train_query_bank=train_query_bank,
            max_queries=args.generator_queries,
            graph_params=graph_params,
        )

    kvpress_results: dict[str, dict[str, float]] | None = None
    if args.run_kvpress:
        kvpress_results = evaluate_kvpress_generator(
            dataset=test_data,
            turn_embeddings=test_turns,
            query_embeddings=test_queries,
            train_query_bank=train_query_bank,
            model_name=args.kvpress_model,
            max_queries=args.kvpress_queries,
            qaware_top_k=args.kvpress_top_k,
            compression_ratio=args.kvpress_compression_ratio,
            graph_params=graph_params,
        )

    if args.log_experiment:
        append_experiment_log(args, retrieval_results, kvpress_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a query-aware sparse conversation memory against direct quantization."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-conversations", type=int, default=18)
    parser.add_argument("--test-conversations", type=int, default=12)
    parser.add_argument("--turns-per-conversation", type=int, default=48)
    parser.add_argument("--queries-per-conversation", type=int, default=4)
    parser.add_argument("--run-generator", action="store_true")
    parser.add_argument("--generator-queries", type=int, default=12)
    parser.add_argument("--run-kvpress", action="store_true")
    parser.add_argument("--kvpress-model", type=str, default=HF_QWEN_SMALL)
    parser.add_argument("--kvpress-queries", type=int, default=8)
    parser.add_argument("--kvpress-top-k", type=int, default=1)
    parser.add_argument("--kvpress-compression-ratio", type=float, default=0.5)
    parser.add_argument("--log-experiment", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
