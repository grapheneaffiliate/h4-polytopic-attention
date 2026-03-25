"""
Interactive demo: ask questions about a document collection.

Usage:
    python demo.py --docs path/to/documents/
    python demo.py --docs path/to/documents/ --question "What is..."

No GPU. No API key. No cloud account. Just a laptop and documents.

Example:
    $ python demo.py --docs sample_docs/

    Indexed 3 documents (47 chunks) in 0.12s
    Model: H4 ternary, d_model=256, 4 layers, 8 heads

    > What is the golden ratio?

    Answer: The golden ratio phi equals one plus...

    Sources:
      [1] math_notes.txt (chunk 2, dist=0.342)
      [2] geometry.txt (chunk 7, dist=0.518)

    Retrieval: 0.8ms | Generation: 340ms | 128 tokens at 376 tok/s
    Context: 512 tokens (3.1% scanned by ChamberTree)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.pipeline import H4RAGPipeline


def build_vocab_from_docs(doc_dir: str):
    """Build character-level vocabulary from all documents in a directory."""
    all_text = ""
    for fname in os.listdir(doc_dir):
        if fname.endswith('.txt'):
            path = os.path.join(doc_dir, fname)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                all_text += f.read()

    # Add common chars that might appear in questions but not docs
    all_text += " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:;()|"

    chars = sorted(list(set(all_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return len(chars), stoi, itos


def create_sample_docs(doc_dir: str):
    """Create sample documents if none provided."""
    os.makedirs(doc_dir, exist_ok=True)

    docs = {
        'golden_ratio.txt': (
            "The golden ratio, often denoted by the Greek letter phi, is a special number "
            "approximately equal to 1.618. It appears throughout mathematics, art, and nature. "
            "Two quantities are in the golden ratio if their ratio is the same as the ratio of "
            "their sum to the larger of the two quantities. The golden ratio is an irrational "
            "number that is a solution to the quadratic equation x squared equals x plus one. "
            "The golden ratio is closely related to the Fibonacci sequence. As Fibonacci numbers "
            "increase, the ratio of consecutive Fibonacci numbers approaches the golden ratio. "
            "For example, 8 divided by 5 equals 1.6, and 13 divided by 8 equals 1.625, which "
            "are close approximations. The golden ratio appears in the geometry of pentagons "
            "and in the arrangement of leaves and petals in many plants. "
        ),
        'polytopes.txt': (
            "A polytope is a geometric object with flat sides in any number of dimensions. "
            "In two dimensions, polytopes are polygons. In three dimensions, they are polyhedra. "
            "In four dimensions, they are called polychora or 4-polytopes. The 600-cell is a "
            "regular 4-polytope with 120 vertices, 720 edges, 1200 triangular faces, and 600 "
            "tetrahedral cells. It is the four-dimensional analog of the icosahedron. The 600-cell "
            "has the H4 symmetry group, which contains 14400 elements. This is the largest finite "
            "reflection group in four dimensions. The vertices of the 600-cell can be expressed "
            "using the golden ratio phi. The coordinates include values like phi over two and "
            "one over two phi. The 600-cell is dual to the 120-cell, which has 600 vertices. "
            "Together they form the most symmetric structures possible in four dimensions. "
        ),
        'e8_lattice.txt': (
            "The E8 lattice is the densest sphere packing in eight dimensions. This was proven "
            "by Maryna Viazovska in 2016. The E8 lattice has a kissing number of 240, meaning "
            "each sphere touches exactly 240 others. The lattice can be decomposed as the union "
            "of two cosets: D8 and D8 shifted by the vector one half in all coordinates. The "
            "E8 lattice is connected to the H4 polytope through a remarkable projection. The "
            "Coxeter element of E8 has eigenvalues that include cosine of pi over five, which "
            "equals phi over two. When the 240 roots of E8 are projected along these eigenspaces, "
            "they map to the vertices of H4 polytopes. This projection preserves the golden ratio "
            "structure, connecting eight-dimensional lattice geometry to four-dimensional polytope "
            "symmetry. The E8 lattice is used in coding theory, string theory, and as a memory "
            "addressing scheme where Voronoi cells provide natural bucket boundaries. "
        ),
    }

    for fname, content in docs.items():
        path = os.path.join(doc_dir, fname)
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

    return doc_dir


def main():
    parser = argparse.ArgumentParser(description='H4 Geometric RAG Demo')
    parser.add_argument('--docs', type=str, default=None, help='Path to document directory')
    parser.add_argument('--question', type=str, default=None, help='Single question (non-interactive)')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--max_tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--k', type=int, default=3, help='Number of chunks to retrieve')
    parser.add_argument('--ternary', action='store_true', default=True, help='Use ternary weights')
    args = parser.parse_args()

    # Use sample docs if none provided
    if args.docs is None:
        sample_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'sample_docs')
        args.docs = create_sample_docs(sample_dir)
        print(f"Created sample documents in {args.docs}")

    if not os.path.isdir(args.docs):
        print(f"Error: {args.docs} is not a directory")
        return

    # Build vocabulary from documents
    print("Building vocabulary...")
    vocab_size, stoi, itos = build_vocab_from_docs(args.docs)
    print(f"Vocabulary: {vocab_size} characters")

    # Create pipeline
    print(f"Creating H4 RAG pipeline (d_model={args.d_model}, {args.n_layers} layers, "
          f"{'ternary' if args.ternary else 'float'})...")
    pipeline = H4RAGPipeline(
        vocab_size=vocab_size,
        stoi=stoi,
        itos=itos,
        d_model=args.d_model,
        n_heads=8,
        n_layers=args.n_layers,
        use_bitlinear=args.ternary,
        max_context=512,
    )

    # Index documents
    t0 = time.perf_counter()
    n_docs = pipeline.index_directory(args.docs)
    t_index = time.perf_counter() - t0
    stats = pipeline.stats()
    print(f"Indexed {n_docs} documents ({stats['n_chunks']} chunks) in {t_index:.2f}s")
    print(f"Lattice utilization: {stats['lattice_utilization']:.1%}")
    params = stats['model_params']
    print(f"Model: {params['trainable']:,} trainable params, "
          f"{params['buffers']:,} buffer elements")
    print()

    if args.question:
        # Single question mode
        result = pipeline.answer(args.question, k=args.k, max_tokens=args.max_tokens)
        _print_result(result)
    else:
        # Interactive mode
        print("Ask questions about your documents. Type 'quit' to exit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question or question.lower() in ('quit', 'exit', 'q'):
                break

            result = pipeline.answer(question, k=args.k, max_tokens=args.max_tokens)
            _print_result(result)
            print()


def _print_result(result):
    """Pretty-print a RAG result."""
    print(f"\nAnswer: {result.answer[:500]}")
    print(f"\nSources:")
    for i, src in enumerate(result.sources):
        print(f"  [{i+1}] {src['doc_id']} (chunk {src['chunk_idx']}, "
              f"dist={src['distance']:.3f})")
        print(f"      {src['preview'][:60]}...")
    print(f"\nRetrieval: {result.retrieval_time_ms:.1f}ms | "
          f"Generation: {result.generation_time_ms:.1f}ms | "
          f"{result.tokens_generated} tokens at {result.tokens_per_second:.0f} tok/s")
    print(f"Context: {result.context_length} tokens")


if __name__ == '__main__':
    main()
