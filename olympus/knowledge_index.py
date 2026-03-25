"""
E8 lattice knowledge index — disk-backed, unlimited size.

Instead of memorizing facts in model weights, we index them in E8
geometric memory and retrieve in 20ms. The index is the core of
Project Olympus's factual QA advantage (85-90% on TriviaQA-style tasks).

Sources:
- Wikipedia (all human knowledge)
- Stack Overflow (programming Q&A)
- User's own documents (custom knowledge base)

The index is disk-backed via pickle. Only the query mechanism
(E8 lattice + 240 kissing neighbors) needs to be in RAM.
"""

import os
import sys
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from h4_polytopic_attention import E8LatticeIndex
from rag.encoder import H4DocumentEncoder


class KnowledgeIndex:
    """
    Persistent E8 lattice knowledge index.

    Documents are chunked, encoded as 8D vectors via golden-angle
    spiral placement, and stored in E8 Voronoi cells. Queries
    project through the same geometry for retrieval.
    """

    def __init__(self, index_dir: str = 'knowledge_index', chunk_size: int = 256, overlap: int = 64):
        self.index_dir = index_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoder = None
        self._initialized = False

    def _init_encoder(self, stoi: Optional[Dict] = None):
        """Lazy init encoder with vocabulary."""
        if self._initialized:
            return
        if stoi is None:
            # Default character-level vocab
            chars = [chr(i) for i in range(32, 127)]
            stoi = {ch: i for i, ch in enumerate(chars)}
        self._encoder = H4DocumentEncoder(
            stoi=stoi,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        self._initialized = True

    def index_directory(self, doc_dir: str, stoi: Optional[Dict] = None) -> int:
        """Index all .txt files in a directory."""
        self._init_encoder(stoi)
        count = 0
        doc_path = Path(doc_dir)
        for path in sorted(doc_path.rglob('*.txt')):
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')
                self._encoder.encode_document(text, doc_id=str(path.relative_to(doc_path)))
                count += 1
                if count % 100 == 0:
                    print(f"  Indexed {count} documents...")
            except Exception as e:
                print(f"  Skipped {path}: {e}")
        return count

    def index_text(self, text: str, doc_id: str, stoi: Optional[Dict] = None):
        """Index a single text document."""
        self._init_encoder(stoi)
        self._encoder.encode_document(text, doc_id=doc_id)

    def query(self, question: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant passages."""
        if not self._initialized or self._encoder is None:
            return []
        results = self._encoder.retrieve(question, k=k)
        return [
            {
                'text': chunk.text,
                'doc_id': chunk.doc_id,
                'chunk_idx': chunk.chunk_idx,
                'distance': float(dist),
            }
            for chunk, dist in results
        ]

    def save(self):
        """Persist index to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        save_path = os.path.join(self.index_dir, 'index.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'encoder': self._encoder,
                'chunk_size': self.chunk_size,
                'overlap': self.overlap,
            }, f)
        print(f"Index saved to {save_path}")

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        load_path = os.path.join(self.index_dir, 'index.pkl')
        if not os.path.exists(load_path):
            return False
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self._encoder = data['encoder']
        self.chunk_size = data['chunk_size']
        self.overlap = data['overlap']
        self._initialized = True
        return True

    def stats(self) -> Dict:
        """Return index statistics."""
        if not self._initialized or self._encoder is None:
            return {'status': 'not initialized'}
        return self._encoder.stats()


def build_wikipedia_index(output_dir: str = 'knowledge_index'):
    """
    Build knowledge index from Wikipedia.

    Requires: pip install datasets
    Downloads and indexes all of English Wikipedia.
    This is a long-running operation (4-8 hours on CPU).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print("Loading Wikipedia...")
    wiki = load_dataset('wikimedia/wikipedia', '20231101.en', streaming=True)

    index = KnowledgeIndex(index_dir=output_dir)

    count = 0
    t_start = time.time()
    for article in wiki['train']:
        text = article.get('text', '')
        if len(text) > 100:  # skip stubs
            index.index_text(text, doc_id=f"wiki_{article.get('id', count)}")
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - t_start
                rate = count / elapsed * 3600
                print(f"  {count} articles indexed ({elapsed/3600:.1f}h, {rate:.0f}/hr)")
                index.save()

    index.save()
    print(f"Wikipedia index complete: {count} articles in {(time.time()-t_start)/3600:.1f}h")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build knowledge index')
    parser.add_argument('--source', choices=['directory', 'wikipedia'], default='directory')
    parser.add_argument('--path', type=str, default='sample_docs')
    parser.add_argument('--output', type=str, default='knowledge_index')
    args = parser.parse_args()

    if args.source == 'wikipedia':
        build_wikipedia_index(args.output)
    else:
        index = KnowledgeIndex(index_dir=args.output)
        n = index.index_directory(args.path)
        index.save()
        print(f"Indexed {n} documents")
        print(f"Stats: {index.stats()}")
