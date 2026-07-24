"""FAISS-based retrieval index for RAGMat-OOD.

Implements ``FAISSIndex`` with:
- L2-normalised embeddings for cosine similarity via inner product (IndexFlatIP).
- Train-partition-only indexing enforced at the API level.
- Separate index per (representation, property, split) triple.
- Named: ``faiss_{tier}_{repr}_{property}_{split}.index``

Critical rules:
- ``build()`` MUST only be called with train-partition embeddings.
- Never share an index across property types or splits.
- Always L2-normalise embeddings before ``build()`` and before ``query()``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import urllib.request

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    """Best-effort debug reporting for the retrieval silent failure session."""
    env_path = Path(__file__).resolve().parents[2] / ".dbg" / "retrieval-silent-failure.env"
    url = "http://127.0.0.1:7777/event"
    session_id = "retrieval-silent-failure"
    try:
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        payload = {
            "sessionId": session_id,
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
        }
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
        ).read()
    except Exception:
        pass


class FAISSIndex:
    """Exact cosine similarity retrieval index using FAISS IndexFlatIP.

    Args:
        dim: Embedding dimension.
        property_name: Property this index was built for (integrity label).
        split_name: Split this index was built for (integrity label).
    """

    def __init__(self, dim: int, property_name: str, split_name: str) -> None:
        self.dim = dim
        self.property_name = property_name
        self.split_name = split_name
        self._index: faiss.IndexFlatIP | None = None
        self._material_ids: list[str] = []
        self._built: bool = False

    # ── Public API ──────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, material_ids: list[str]) -> None:
        """Build the FAISS index from train-partition embeddings.

        Embeddings are L2-normalised before indexing to enable cosine
        similarity via inner product.

        Args:
            embeddings: Train-partition embeddings ``(N, dim)``.  Must be
                extracted from the training partition ONLY.
            material_ids: Corresponding material IDs (same order).

        Raises:
            ValueError: If ``embeddings.shape[1] != self.dim``.
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings of dim={self.dim}, "
                f"got {embeddings.shape[1]}"
            )
        if len(material_ids) != embeddings.shape[0]:
            raise ValueError(
                "embeddings and material_ids must have the same length"
            )

        emb = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(emb)  # In-place L2 normalisation

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(emb)
        self._material_ids = list(material_ids)
        self._built = True

        logger.info(
            "FAISSIndex built: %d vectors, dim=%d, property=%s, split=%s",
            self._index.ntotal,
            self.dim,
            self.property_name,
            self.split_name,
        )

    def query(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Retrieve top-k nearest neighbours for each query embedding.

        Query embeddings are L2-normalised before searching.

        Args:
            query_embeddings: Query embeddings ``(M, dim)``.
            top_k: Number of neighbours to retrieve.

        Returns:
            Tuple ``(scores, ids)`` where:
            - ``scores``: Shape ``(M, top_k)`` cosine similarity scores.
            - ``ids``: Nested list of shape ``(M, top_k)`` material ID strings.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if not self._built:
            raise RuntimeError(
                "FAISSIndex is not built. Call build() or load() first."
            )
        query = np.ascontiguousarray(query_embeddings.astype(np.float32))
        # #region debug-point B:faiss-query-entry
        _debug_report(
            "B",
            "ragmat/retrieval/faiss_index.py:query",
            "[DEBUG] FAISS query called",
            {
                "query_shape": list(query.shape),
                "top_k": int(top_k),
                "dim": int(self.dim),
                "built": bool(self._built),
                "index_ntotal": int(self._index.ntotal if self._index is not None else 0),
            },
        )
        # #endregion
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, top_k)  # (M, k)

        result_ids: list[list[str]] = []
        for row in indices:
            row_ids = [
                self._material_ids[int(i)] if 0 <= int(i) < len(self._material_ids) else ""
                for i in row
            ]
            result_ids.append(row_ids)
        # #region debug-point B:faiss-query-result
        _debug_report(
            "B",
            "ragmat/retrieval/faiss_index.py:query",
            "[DEBUG] FAISS query returned",
            {
                "scores_shape": list(scores.shape),
                "indices_shape": list(indices.shape),
                "first_score_row": scores[0].tolist() if len(scores) else [],
                "first_id_row": result_ids[0] if result_ids else [],
            },
        )
        # #endregion

        return scores, result_ids

    def save(self, path: str) -> None:
        """Save the FAISS index and material ID list to disk.

        Args:
            path: Base path (without extension). Two files are written:
                ``{path}.index`` and ``{path}_ids.json``.
        """
        if not self._built:
            raise RuntimeError("Cannot save: index not built.")
        index_path = str(path) + ".index"
        ids_path = str(path) + "_ids.json"
        faiss.write_index(self._index, index_path)
        with open(ids_path, "w") as f:
            json.dump(self._material_ids, f)
        logger.info("FAISSIndex saved to %s", index_path)

    def load(self, path: str) -> None:
        """Load a previously saved FAISS index from disk.

        Args:
            path: Base path (without extension). Expects ``{path}.index``
                and ``{path}_ids.json``.
        """
        index_path = str(path) + ".index"
        ids_path = str(path) + "_ids.json"
        self._index = faiss.read_index(index_path)
        with open(ids_path) as f:
            self._material_ids = json.load(f)
        self._built = True
        logger.info(
            "FAISSIndex loaded: %d vectors from %s",
            self._index.ntotal,
            index_path,
        )

    @staticmethod
    def index_name(
        tier: int,
        representation: str,
        property_name: str,
        split_name: str,
    ) -> str:
        """Return the canonical file base name for a FAISS index.

        Example: ``faiss_tier1_cgcnn_band_gap_element_out``

        Args:
            tier: Experiment tier (0 or 1).
            representation: ``"matminer"`` or ``"cgcnn"``.
            property_name: ``"formation_energy"`` or ``"band_gap"``.
            split_name: ``"iid"``, ``"family_out"``, or ``"element_out"``.

        Returns:
            Base filename string (no extension).
        """
        return f"faiss_tier{tier}_{representation}_{property_name}_{split_name}"
