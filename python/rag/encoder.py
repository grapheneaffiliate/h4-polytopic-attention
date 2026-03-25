"""
H4 Document Encoder — Encode text into E8 lattice memory for geometric retrieval.

Each document chunk becomes an 8D embedding stored in a Voronoi cell.
The encoding uses golden-angle spiral placement based on token content,
ensuring that retrieval and attention share the same geometric space
through the E8→H4 projection (cos(π/5) = φ/2).

No separate embedding model needed. The same geometry handles both.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from h4_polytopic_attention import E8LatticeIndex, PHI, PHI_INV


@dataclass
class Chunk:
    """A document chunk stored in E8 lattice memory."""
    text: str
    doc_id: str
    chunk_idx: int
    token_ids: List[int]


class H4DocumentEncoder:
    """
    Encode documents into E8 lattice memory for retrieval.

    Each chunk becomes an 8D embedding stored in an E8 Voronoi cell.
    The encoding uses golden-angle spiral placement based on token
    frequencies, ensuring geometric consistency with H4 attention.

    The 8D embedding captures:
    - dims 0-3: semantic content (weighted token frequency features)
    - dims 4-7: positional/structural features (chunk position, doc features)
    """

    def __init__(self, stoi: Dict[str, int], chunk_size: int = 256, overlap: int = 64):
        """
        Args:
            stoi: string-to-index vocabulary mapping
            chunk_size: tokens per chunk
            overlap: overlap between consecutive chunks
        """
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.vocab_size = len(stoi)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.lattice = E8LatticeIndex(max_cell_size=240)
        self.chunks: List[Chunk] = []
        self._address_counter = 0

        # Precompute per-token 8D embeddings using golden-angle spiral
        self._token_embeddings = self._build_token_embeddings()

    def _build_token_embeddings(self) -> np.ndarray:
        """Build 8D embeddings for each token using golden-angle spiral."""
        embs = np.zeros((self.vocab_size, 8))
        for i in range(self.vocab_size):
            # Golden-angle placement in 8D: pairs of (cos, sin) at φ-scaled frequencies
            for d in range(4):
                angle = i * 2 * math.pi * PHI_INV * (PHI ** (-d / 4))
                embs[i, 2 * d] = math.cos(angle)
                embs[i, 2 * d + 1] = math.sin(angle)
            # Normalize
            norm = np.linalg.norm(embs[i])
            if norm > 1e-10:
                embs[i] /= norm
        return embs

    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token IDs using the vocabulary."""
        return [self.stoi.get(c, 0) for c in text]

    def _chunk_tokens(self, token_ids: List[int]) -> List[List[int]]:
        """Split token list into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            chunks.append(token_ids[start:end])
            start += self.chunk_size - self.overlap
            if end == len(token_ids):
                break
        return chunks

    def _embed_chunk(self, token_ids: List[int], chunk_idx: int, n_chunks: int) -> np.ndarray:
        """
        Compute 8D embedding for a chunk.

        Combines token frequency features (dims 0-3) with
        positional features (dims 4-7).
        """
        emb = np.zeros(8)

        # Semantic: weighted average of token embeddings
        if token_ids:
            token_embs = self._token_embeddings[token_ids]  # (n_tokens, 8)
            # Weight by position in chunk (later tokens slightly higher weight)
            weights = np.array([1.0 + 0.5 * PHI_INV * (i / len(token_ids))
                                for i in range(len(token_ids))])
            weights /= weights.sum()
            emb[:4] = (token_embs[:, :4].T @ weights)

        # Positional: chunk position and length features
        pos_frac = chunk_idx / max(n_chunks - 1, 1) if n_chunks > 1 else 0.5
        len_frac = len(token_ids) / self.chunk_size
        angle1 = pos_frac * 2 * math.pi * PHI_INV
        angle2 = len_frac * math.pi * PHI_INV
        emb[4] = math.cos(angle1)
        emb[5] = math.sin(angle1)
        emb[6] = math.cos(angle2)
        emb[7] = math.sin(angle2)

        # Normalize to unit sphere
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb /= norm
        return emb

    def encode_document(self, text: str, doc_id: str):
        """
        Chunk document, encode each chunk as 8D vector, store in E8 lattice.

        Args:
            text: document text
            doc_id: unique identifier for the document
        """
        token_ids = self._text_to_tokens(text)
        token_chunks = self._chunk_tokens(token_ids)
        n_chunks = len(token_chunks)

        for i, chunk_tokens in enumerate(token_chunks):
            # Compute 8D embedding
            embedding = self._embed_chunk(chunk_tokens, i, n_chunks)

            # Store in E8 lattice
            address = self._address_counter
            self._address_counter += 1
            self.lattice.insert(embedding, value=float(address), address=address)

            # Store chunk metadata
            chunk_text = ''.join(self.itos.get(t, '?') for t in chunk_tokens)
            self.chunks.append(Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_idx=i,
                token_ids=chunk_tokens,
            ))

    def retrieve(self, query_text: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Encode query as 8D vector, find k nearest chunks in E8 lattice.

        Returns:
            List of (chunk, distance) tuples sorted by E8 distance.
        """
        query_tokens = self._text_to_tokens(query_text)
        query_emb = self._embed_chunk(query_tokens, 0, 1)

        results = self.lattice.query_nearest(query_emb, k=k, search_neighbors=True)

        retrieved = []
        for dist_sq, value, addr in results:
            idx = int(addr)
            if idx < len(self.chunks):
                retrieved.append((self.chunks[idx], dist_sq))

        return retrieved

    def retrieve_with_h4(self, query_text: str, k: int = 5):
        """
        Retrieve chunks AND return their H4 projections for attention.

        Returns:
            chunks: list of Chunk objects
            h4_keys: (k, 4) array — 4D projections for direct attention use
            e8_embeddings: (k, 8) array — full 8D embeddings
            distances: (k,) array — E8 distances to query
        """
        query_tokens = self._text_to_tokens(query_text)
        query_emb = self._embed_chunk(query_tokens, 0, 1)

        results = self.lattice.query_nearest(query_emb, k=k, search_neighbors=True)

        chunks = []
        h4_keys = []
        e8_embs = []
        distances = []

        for dist_sq, value, addr in results:
            idx = int(addr)
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunks.append(chunk)
                distances.append(dist_sq)

                # Reconstruct 8D embedding
                emb = self._embed_chunk(
                    chunk.token_ids, chunk.chunk_idx,
                    sum(1 for c in self.chunks if c.doc_id == chunk.doc_id)
                )
                e8_embs.append(emb)

                # Project to H4 via E8→H4 projection
                h4_key = self.lattice.project_to_h4(emb)
                h4_keys.append(h4_key)

        return (
            chunks,
            np.array(h4_keys) if h4_keys else np.zeros((0, 4)),
            np.array(e8_embs) if e8_embs else np.zeros((0, 8)),
            np.array(distances) if distances else np.zeros(0),
        )

    def stats(self) -> Dict:
        """Return encoder statistics."""
        lattice_stats = self.lattice.stats()
        return {
            'n_chunks': len(self.chunks),
            'n_documents': len(set(c.doc_id for c in self.chunks)),
            'lattice_cells': lattice_stats.get('occupied_cells', 0),
            'lattice_utilization': lattice_stats.get('utilization', 0),
            'avg_chunk_len': (
                sum(len(c.token_ids) for c in self.chunks) / len(self.chunks)
                if self.chunks else 0
            ),
        }
