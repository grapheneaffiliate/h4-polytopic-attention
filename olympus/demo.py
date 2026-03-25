"""
Project Olympus interactive demo.

Ask anything, get routed to the right specialist, answers backed
by E8 lattice knowledge retrieval + MiniLM reranking.

Usage:
    python olympus/demo.py
    python olympus/demo.py --docs path/to/your/documents/

No GPU. No API key. No cloud account. Just a laptop.
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from olympus.router import GeometricRouter
from olympus.knowledge_index import KnowledgeIndex


def main():
    parser = argparse.ArgumentParser(description='Project Olympus Demo')
    parser.add_argument('--docs', type=str, default=None,
                        help='Path to documents to index')
    parser.add_argument('--index', type=str, default='knowledge_index',
                        help='Path to persistent knowledge index')
    args = parser.parse_args()

    print("=" * 60)
    print("  PROJECT OLYMPUS — Frontier-Quality AI on CPU")
    print("=" * 60)
    print()

    # Initialize router
    print("Initializing geometric router...")
    router = GeometricRouter()
    print("  ChamberTree: 16 chambers, 4 specialists")

    # Initialize knowledge index
    print("Loading knowledge index...")
    index = KnowledgeIndex(index_dir=args.index)
    if not index.load():
        print("  No existing index found.")
        if args.docs:
            print(f"  Indexing documents from {args.docs}...")
            n = index.index_directory(args.docs)
            index.save()
            print(f"  Indexed {n} documents")
        else:
            # Index sample docs
            sample_dir = os.path.join(os.path.dirname(__file__), '..', 'sample_docs')
            if os.path.isdir(sample_dir):
                n = index.index_directory(sample_dir)
                print(f"  Indexed {n} sample documents")
            else:
                print("  No documents available. Use --docs to specify a directory.")
    else:
        stats = index.stats()
        print(f"  Loaded: {stats.get('n_chunks', '?')} chunks from "
              f"{stats.get('n_documents', '?')} documents")

    # Status
    print()
    print("  Specialists:")
    print("    [general] Conversation, instructions, creative, summary")
    print("    [code]    Code generation, debugging, explanation")
    print("    [math]    Problem solving, logical reasoning")
    print("    [qa]      Factual answers from retrieved knowledge")
    print()
    print("  Note: Specialist models not yet trained.")
    print("  This demo shows routing + retrieval (the infrastructure).")
    print("  Full generation requires completing the Olympus training plan.")
    print()
    print("Type a question (or 'quit' to exit).")
    print()

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() in ('quit', 'exit', 'q'):
            break

        t_start = time.perf_counter()

        # Route
        t_route = time.perf_counter()
        specialists = router.route_with_fallback(query)
        specialist, chamber, confidence = router.route(query)
        route_ms = (time.perf_counter() - t_route) * 1000

        # Retrieve
        t_retrieve = time.perf_counter()
        passages = index.query(query, k=3)
        retrieve_ms = (time.perf_counter() - t_retrieve) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        # Display
        print(f"\n  [Router] specialist={specialist}, chamber={chamber}, "
              f"confidence={confidence:.3f} ({route_ms:.1f}ms)")

        if len(specialists) > 1:
            print(f"  [Fallback] also querying: {specialists[1]}")

        if passages:
            print(f"  [Retrieval] {len(passages)} passages ({retrieve_ms:.1f}ms)")
            for i, p in enumerate(passages):
                preview = p['text'][:100].replace('\n', ' ')
                print(f"    [{i+1}] {p['doc_id']} (dist={p['distance']:.3f})")
                print(f"        {preview}...")
        else:
            print(f"  [Retrieval] No passages found")

        print(f"\n  [Generation] Specialist model not loaded — see PROJECT_OLYMPUS.md")
        print(f"  Total: {total_ms:.1f}ms (route: {route_ms:.1f}ms, retrieve: {retrieve_ms:.1f}ms)")
        print()


if __name__ == '__main__':
    main()
