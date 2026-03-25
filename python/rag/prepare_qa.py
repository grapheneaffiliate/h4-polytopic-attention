"""
Download and prepare QA training data for H4 RAG.

Uses a simple extractive QA format:
- Input: [context] | [question] |
- Target: [answer]

Data sources (in order of preference):
1. SQuAD-style QA pairs generated from the sample documents
2. Downloaded SQuAD 2.0 dev set (small, freely available)

For CPU training with 2-minute budget, we need small data that
trains fast. The sample doc QA pairs are ideal for proving the
pipeline works; SQuAD provides real benchmark numbers.
"""

import json
import os
import sys
import random
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def generate_sample_qa() -> List[Dict]:
    """
    Generate QA pairs from the sample documents.
    These are hand-crafted to match the sample_docs content.
    The model's job: learn to extract the answer from the context.
    """
    qa_pairs = [
        # golden_ratio.txt
        {"context": "The golden ratio, often denoted by the Greek letter phi, is a special number approximately equal to 1.618.",
         "question": "What is the golden ratio approximately equal to?",
         "answer": "1.618"},
        {"context": "Two quantities are in the golden ratio if their ratio is the same as the ratio of their sum to the larger of the two quantities.",
         "question": "When are two quantities in the golden ratio?",
         "answer": "if their ratio is the same as the ratio of their sum to the larger"},
        {"context": "The golden ratio is closely related to the Fibonacci sequence. As Fibonacci numbers increase, the ratio of consecutive Fibonacci numbers approaches the golden ratio.",
         "question": "How is the golden ratio related to Fibonacci numbers?",
         "answer": "the ratio of consecutive Fibonacci numbers approaches the golden ratio"},
        {"context": "The golden ratio appears in the geometry of pentagons and in the arrangement of leaves and petals in many plants.",
         "question": "Where does the golden ratio appear in nature?",
         "answer": "in the arrangement of leaves and petals in many plants"},

        # polytopes.txt
        {"context": "The 600-cell is a regular 4-polytope with 120 vertices, 720 edges, 1200 triangular faces, and 600 tetrahedral cells.",
         "question": "How many vertices does the 600-cell have?",
         "answer": "120"},
        {"context": "The 600-cell has the H4 symmetry group, which contains 14400 elements. This is the largest finite reflection group in four dimensions.",
         "question": "How many elements does the H4 symmetry group contain?",
         "answer": "14400"},
        {"context": "The 600-cell is dual to the 120-cell, which has 600 vertices.",
         "question": "What is the 600-cell dual to?",
         "answer": "the 120-cell"},
        {"context": "A polytope is a geometric object with flat sides in any number of dimensions.",
         "question": "What is a polytope?",
         "answer": "a geometric object with flat sides in any number of dimensions"},

        # e8_lattice.txt
        {"context": "The E8 lattice is the densest sphere packing in eight dimensions. This was proven by Maryna Viazovska in 2016.",
         "question": "Who proved E8 is the densest sphere packing?",
         "answer": "Maryna Viazovska"},
        {"context": "The E8 lattice has a kissing number of 240, meaning each sphere touches exactly 240 others.",
         "question": "What is the kissing number of E8?",
         "answer": "240"},
        {"context": "The Coxeter element of E8 has eigenvalues that include cosine of pi over five, which equals phi over two.",
         "question": "What eigenvalue connects E8 to the golden ratio?",
         "answer": "cosine of pi over five, which equals phi over two"},
        {"context": "When the 240 roots of E8 are projected along these eigenspaces, they map to the vertices of H4 polytopes.",
         "question": "What happens when E8 roots are projected along the eigenspaces?",
         "answer": "they map to the vertices of H4 polytopes"},
    ]

    return qa_pairs


def prepare_training_data(
    qa_pairs: List[Dict],
    val_fraction: float = 0.2,
) -> Tuple[List[Dict], List[Dict]]:
    """Split QA pairs into train and validation sets."""
    random.seed(42)
    pairs = list(qa_pairs)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_fraction))
    return pairs[n_val:], pairs[:n_val]


def format_qa_for_training(qa_pair: Dict, sep: str = " | ") -> Tuple[str, str]:
    """
    Format a QA pair for character-level training.

    Input: [context] | [question] |
    Target: [answer]

    The model learns to generate the answer given context + question.
    """
    input_text = qa_pair['context'] + sep + qa_pair['question'] + sep
    target_text = qa_pair['answer']
    return input_text, target_text


def download_squad_dev():
    """
    Download SQuAD 2.0 dev set for real benchmark evaluation.
    Returns list of QA dicts with context/question/answer.
    """
    import urllib.request

    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'squad_dev.json')

    if not os.path.exists(cache_path):
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        print(f"Downloading SQuAD 2.0 dev set...")
        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"Saved to {cache_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            return []

    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qa_pairs = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                if qa.get('is_impossible', False):
                    continue
                if qa['answers']:
                    answer = qa['answers'][0]['text']
                    qa_pairs.append({
                        'context': context[:500],  # truncate long contexts
                        'question': qa['question'],
                        'answer': answer,
                    })

    return qa_pairs


if __name__ == '__main__':
    print("Generating sample QA pairs...")
    pairs = generate_sample_qa()
    train, val = prepare_training_data(pairs)
    print(f"Sample QA: {len(train)} train, {len(val)} val")

    for p in pairs[:3]:
        inp, tgt = format_qa_for_training(p)
        print(f"\nInput:  {inp[:80]}...")
        print(f"Target: {tgt}")

    print("\nAttempting SQuAD download...")
    squad = download_squad_dev()
    if squad:
        print(f"SQuAD 2.0 dev: {len(squad)} answerable questions")
    else:
        print("SQuAD not available (offline?). Using sample QA only.")
