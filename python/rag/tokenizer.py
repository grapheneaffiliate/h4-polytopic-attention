"""
BPE tokenizer wrapper for H4 RAG.

Uses tiktoken's GPT-2 BPE but maps to a smaller vocabulary suitable for
our model size. The full 50K GPT-2 vocab is too large for a 64-256 dim model.
We build a restricted vocab from the training data — tokens that actually
appear — and map everything else to UNK.

This gives ~5x compression over character-level (512 BPE tokens ≈ 2500 chars)
which means seq_len=512 fits virtually all SQuAD contexts.
"""

import tiktoken
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


class BPETokenizer:
    """
    Restricted-vocabulary BPE tokenizer.

    Uses tiktoken GPT-2 as the base BPE, then maps token IDs to a
    smaller vocabulary built from training data. Unknown tokens map to UNK.

    Special tokens:
        0 = PAD
        1 = UNK
        2 = SEP (separator between context, question, answer)
    """

    N_SPECIAL = 3  # PAD, UNK, SEP

    def __init__(self, max_vocab: int = 4096):
        self.max_vocab = max_vocab
        self.base_enc = tiktoken.get_encoding('gpt2')

        # Will be populated by build_vocab()
        self.base_to_local: Dict[int, int] = {}
        self.local_to_base: Dict[int, int] = {}
        self.vocab_size = self.N_SPECIAL  # updated after build_vocab

    def build_vocab(self, texts: List[str]):
        """
        Build restricted vocabulary from training texts.
        Keeps the most frequent max_vocab - N_SPECIAL tokens.
        """
        counter = Counter()
        for text in texts:
            tokens = self.base_enc.encode(text, disallowed_special=())
            counter.update(tokens)

        # Keep most frequent tokens
        n_keep = self.max_vocab - self.N_SPECIAL
        most_common = counter.most_common(n_keep)

        self.base_to_local = {}
        self.local_to_base = {}
        for i, (base_id, _count) in enumerate(most_common):
            local_id = i + self.N_SPECIAL
            self.base_to_local[base_id] = local_id
            self.local_to_base[local_id] = base_id

        self.vocab_size = len(self.base_to_local) + self.N_SPECIAL
        coverage = sum(c for _, c in most_common) / sum(counter.values()) if counter else 0
        print(f"BPE vocab: {self.vocab_size} tokens "
              f"(from {len(counter)} unique, {coverage:.1%} coverage)")

    def encode(self, text: str) -> List[int]:
        """Encode text to local token IDs."""
        base_tokens = self.base_enc.encode(text, disallowed_special=())
        return [self.base_to_local.get(t, 1) for t in base_tokens]  # 1 = UNK

    def decode(self, ids: List[int]) -> str:
        """Decode local token IDs back to text."""
        base_ids = []
        for local_id in ids:
            if local_id < self.N_SPECIAL:
                continue  # skip PAD, UNK, SEP
            base_id = self.local_to_base.get(local_id)
            if base_id is not None:
                base_ids.append(base_id)
        return self.base_enc.decode(base_ids)

    def encode_qa(self, context: str, question: str, answer: Optional[str] = None) -> Tuple[List[int], Optional[List[int]]]:
        """
        Encode a QA pair.

        Returns:
            input_ids: [context_tokens, SEP, question_tokens, SEP]
            answer_ids: [answer_tokens] or None
        """
        ctx_ids = self.encode(context)
        q_ids = self.encode(question)
        input_ids = ctx_ids + [2] + q_ids + [2]  # 2 = SEP

        answer_ids = None
        if answer is not None:
            answer_ids = self.encode(answer)

        return input_ids, answer_ids
