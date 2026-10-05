"""
Bonus Challenge: HybridMemoryAgent implementation.
Combines Episodic Memory (Vector DB with Qdrant + BM25 RRF)
and Stable User Profile / Streaming Activity (Feast Feature Store).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False


class HybridMemoryAgent:
    """
    Personal AI Memory Agent combining:
    1. Episodic Memory: Qdrant vector store + BM25 sparse index with RRF fusion.
    2. User Profile & Realtime Activity: Feast online feature store.
    """

    def __init__(
        self,
        collection_name: str = "user_memories",
        feast_repo_path: str | Path | None = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        vector_dim: int = 384,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.embedder = TextEmbedding(model_name=embedding_model)

        # In-memory Qdrant client for zero-dependency local execution
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
        )

        # Local storage for BM25 and memory tracking
        self.memories: list[dict[str, Any]] = []
        self._point_id_counter = 0

        # Feast Feature Store connection
        self.fs = None
        if FEAST_AVAILABLE:
            repo_path = Path(feast_repo_path or (Path(__file__).resolve().parent.parent / "app" / "feast_repo"))
            if (repo_path / "feature_store.yaml").exists():
                try:
                    self.fs = FeatureStore(repo_path=str(repo_path))
                except Exception as e:
                    print(f"[Warning] Failed to initialize Feast FeatureStore from {repo_path}: {e}")

    def remember(
        self,
        text: str,
        user_id: str = "u_001",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a new piece of episodic memory for a specific user.
        Performs sentence/paragraph-aware chunking, generates embeddings,
        and indexes into both Qdrant (with user_id filter payload) and BM25.
        """
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if not raw_chunks:
            raw_chunks = [text.strip()]

        for chunk in raw_chunks:
            self._point_id_counter += 1
            point_id = self._point_id_counter

            # Generate embedding
            vector = next(self.embedder.embed([chunk])).tolist()

            payload = {
                "user_id": user_id,
                "text": chunk,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                payload.update(metadata)

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )

            self.memories.append({
                "id": point_id,
                "user_id": user_id,
                "text": chunk,
                "payload": payload,
                "tokens": chunk.lower().split(),
            })

    def _get_user_features(self, user_id: str) -> dict[str, Any]:
        """Fetch stable profile and recent velocity features from Feast online store."""
        default_features = {
            "user_id": user_id,
            "reading_speed_wpm": 200,
            "preferred_language": "vi",
            "topic_affinity": "cloud",
            "queries_last_hour": 12,
            "distinct_topics_24h": 4,
        }

        if not self.fs:
            return default_features

        try:
            requested_features = [
                "user_profile_features:reading_speed_wpm",
                "user_profile_features:preferred_language",
                "user_profile_features:topic_affinity",
                "query_velocity_features:queries_last_hour",
                "query_velocity_features:distinct_topics_24h",
            ]
            online_res = self.fs.get_online_features(
                features=requested_features,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()

            res = {
                "user_id": user_id,
                "reading_speed_wpm": online_res.get("reading_speed_wpm", [200])[0] or 200,
                "preferred_language": online_res.get("preferred_language", ["vi"])[0] or "vi",
                "topic_affinity": online_res.get("topic_affinity", ["cloud"])[0] or "cloud",
                "queries_last_hour": online_res.get("queries_last_hour", [12])[0] or 12,
                "distinct_topics_24h": online_res.get("distinct_topics_24h", [4])[0] or 4,
            }
            return res
        except Exception as e:
            # Fallback gracefully
            return default_features

    def recall(
        self,
        query: str,
        user_id: str = "u_001",
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> str:
        """
        Retrieve relevant episodic memories + user profile features,
        and assemble a rich contextual prompt for downstream generation.
        """
        # 1. Fetch user features from Feast online store
        profile = self._get_user_features(user_id)

        # 2. Filter candidate memories for this user
        user_memories = [m for m in self.memories if m["user_id"] == user_id]
        if not user_memories:
            return self._format_context(profile, [], query)

        # 3. BM25 Retrieval with punctuation-clean word tokens
        import re
        tokenized_corpus = [re.findall(r"\w+", m["text"].lower()) for m in user_memories]
        bm25 = BM25Okapi(tokenized_corpus)
        q_tokens = re.findall(r"\w+", query.lower())
        kw_scores = bm25.get_scores(q_tokens)
        ranked_kw = sorted(range(len(kw_scores)), key=lambda i: -kw_scores[i])
        kw_ranked_ids = [user_memories[i]["id"] for i in ranked_kw]

        # 4. Dense Vector Retrieval (filtered by user_id in Qdrant)
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )
        vector_res = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=user_filter,
            limit=len(user_memories),
        ).points
        vec_ranked_ids = [p.id for p in vector_res]

        # 5. RRF Fusion (rank is 1-based, k=60)
        rrf_scores: dict[int, float] = {}
        for rank, p_id in enumerate(kw_ranked_ids, start=1):
            rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, p_id in enumerate(vec_ranked_ids, start=1):
            rrf_scores[p_id] = rrf_scores.get(p_id, 0.0) + 1.0 / (rrf_k + rank)

        # Personalization boost: apply gentle affinity boost when query asks for recommendation or affinity matches
        user_affinity = str(profile.get("topic_affinity", "")).lower()
        id_to_mem = {m["id"]: m for m in user_memories}

        boosted_scores: list[tuple[int, float]] = []
        is_recommendation = any(k in query.lower() for k in ["recommend", "gợi ý", "sở thích"])
        for p_id, score in rrf_scores.items():
            mem_text = id_to_mem[p_id]["text"].lower()
            if is_recommendation and user_affinity and user_affinity in mem_text:
                score *= 1.20  # Boost when user specifically asks for recommendations based on interest
            boosted_scores.append((p_id, score))

        # Sort and take top_k
        top_candidates = sorted(boosted_scores, key=lambda kv: -kv[1])[:top_k]
        top_memories = [
            {
                "id": p_id,
                "score": score,
                "text": id_to_mem[p_id]["text"],
            }
            for p_id, score in top_candidates
        ]

        # 6. Assemble Final Context
        return self._format_context(profile, top_memories, query)

    def _format_context(
        self,
        profile: dict[str, Any],
        memories: list[dict[str, Any]],
        query: str,
    ) -> str:
        """Format retrieved features and episodic memory into clean prompt context."""
        lines = [
            "================================================================================",
            f"QUERY: {query}",
            "================================================================================",
            "[STABLE USER PROFILE & REALTIME SIGNALS (Feast Feature Store)]",
            f"  • User ID: {profile.get('user_id', 'unknown')}",
            f"  • Preferred Language: {profile.get('preferred_language', 'vi')} (adapts response style)",
            f"  • Reading Speed: {profile.get('reading_speed_wpm', 200)} wpm",
            f"  • Core Topic Affinity: {profile.get('topic_affinity', 'general')}",
            f"  • Recent Activity Velocity: {profile.get('queries_last_hour', 0)} queries in last 1h",
            f"  • Topic Diversity: {profile.get('distinct_topics_24h', 1)} distinct topics in 24h",
            "",
            "[RETRIEVED EPISODIC MEMORIES (Qdrant + BM25 + RRF Hybrid Fusion)]",
        ]

        if not memories:
            lines.append("  (No relevant episodic memories found)")
        else:
            for i, m in enumerate(memories, start=1):
                preview = m["text"].replace("\n", " ")
                lines.append(f"  {i}. [Score: {m['score']:.4f}] {preview}")

        lines.extend([
            "",
            "[AGENT INSTRUCTION CONTEXT]",
            f"  Generate an answer personalized to the user's affinity ({profile.get('topic_affinity')}) "
            f"in {profile.get('preferred_language', 'vi')}, grounding strictly on episodic memories.",
            "================================================================================",
        ])

        return "\n".join(lines)
