"""
H4 Geometric RAG Pipeline — Unified retrieval + generation on CPU.

The E8 lattice handles retrieval (O(1) + 240 neighbors).
The H4 attention handles generation (O(log t) via ChamberTree).
The E8→H4 projection (cos(π/5) = φ/2) connects them geometrically.

No GPU. No separate embedding model. No vector database.
One geometric system handles both retrieval and generation.
"""

import time
import math
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from h4_language_model import H4LanguageModel
from rag.encoder import H4DocumentEncoder


@dataclass
class RAGResult:
    """Result from a RAG query."""
    answer: str
    sources: List[Dict]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    context_length: int
    chunks_retrieved: int


class H4RAGPipeline:
    """
    Complete question-answering pipeline on CPU.

    1. Encode document collection into E8 lattice memory
    2. Given question, retrieve relevant chunks via lattice search
    3. Concatenate question + retrieved chunks as context
    4. Generate answer via H4 attention model (ternary, ChamberTree)
    """

    def __init__(
        self,
        vocab_size: int,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        use_bitlinear: bool = True,
        chunk_size: int = 256,
        overlap: int = 64,
        max_context: int = 1024,
    ):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = vocab_size
        self.max_context = max_context

        # Document encoder with E8 lattice
        self.encoder = H4DocumentEncoder(
            stoi=stoi,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # H4 language model (ternary weights for CPU efficiency)
        self.model = H4LanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_value=16,
            d_ffn=d_model * 4,
            max_seq_len=max_context,
            dropout=0.0,
            use_bitlinear=use_bitlinear,
        )
        self.model.eval()

    def index_document(self, text: str, doc_id: str):
        """Add a document to the retrieval index."""
        self.encoder.encode_document(text, doc_id)

    def index_directory(self, doc_dir: str):
        """Index all .txt files in a directory."""
        count = 0
        for fname in sorted(os.listdir(doc_dir)):
            if fname.endswith('.txt'):
                path = os.path.join(doc_dir, fname)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                self.index_document(text, doc_id=fname)
                count += 1
        return count

    def _encode_text(self, text: str) -> torch.Tensor:
        """Convert text to token ID tensor."""
        ids = [self.stoi.get(c, 0) for c in text]
        return torch.tensor([ids], dtype=torch.long)

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        return ''.join(self.itos.get(i.item(), '?') for i in token_ids)

    def answer(
        self,
        question: str,
        k: int = 5,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> RAGResult:
        """
        Answer a question using retrieved context.

        1. Retrieve top-k chunks from E8 lattice
        2. Build prompt: [retrieved chunks] [SEP] [question] [SEP]
        3. Generate answer token-by-token using H4 attention
        """
        t_start = time.perf_counter()

        # Step 1: Retrieve
        t_ret_start = time.perf_counter()
        retrieved = self.encoder.retrieve(question, k=k)
        t_retrieval = (time.perf_counter() - t_ret_start) * 1000

        # Step 2: Build context
        context_parts = []
        sources = []
        for chunk, dist in retrieved:
            context_parts.append(chunk.text)
            sources.append({
                'doc_id': chunk.doc_id,
                'chunk_idx': chunk.chunk_idx,
                'distance': float(dist),
                'preview': chunk.text[:80],
            })

        # Format: [context chunks] | [question] |
        sep = ' | '
        context_text = sep.join(context_parts) + sep + question + sep

        # Truncate to max context
        if len(context_text) > self.max_context - max_tokens:
            context_text = context_text[-(self.max_context - max_tokens):]

        # Step 3: Generate
        t_gen_start = time.perf_counter()
        input_ids = self._encode_text(context_text)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k_sample=20,
            )

        # Extract generated tokens (after the input)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        answer_text = self._decode_tokens(generated_ids)
        t_generation = (time.perf_counter() - t_gen_start) * 1000

        t_total = (time.perf_counter() - t_start) * 1000
        n_generated = len(generated_ids)
        tps = n_generated / (t_generation / 1000) if t_generation > 0 else 0

        return RAGResult(
            answer=answer_text,
            sources=sources,
            retrieval_time_ms=t_retrieval,
            generation_time_ms=t_generation,
            total_time_ms=t_total,
            tokens_generated=n_generated,
            tokens_per_second=tps,
            context_length=input_ids.shape[1],
            chunks_retrieved=len(retrieved),
        )

    def stats(self) -> Dict:
        """Pipeline statistics."""
        encoder_stats = self.encoder.stats()
        model_params = self.model.count_params()
        return {
            **encoder_stats,
            'model_params': model_params,
            'max_context': self.max_context,
            'use_bitlinear': self.model.use_bitlinear,
        }
