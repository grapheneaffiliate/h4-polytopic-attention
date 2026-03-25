"""
Data download and tokenization pipeline for H4 Polytopic Attention experiments.

Supports multiple datasets with automatic download and caching:
  - synthetic: Fibonacci-structured phrases (no download needed)
  - shakespeare: Tiny Shakespeare (~1MB character-level text)
  - tinystories: TinyStories from HuggingFace (real children's stories)

All datasets return the same interface:
    (train_data, val_data, vocab_size, stoi, itos)
"""

import os
import sys
import json
import torch
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

DATASETS = {
    'synthetic': {
        'source': 'synthetic',
        'description': 'Fibonacci-structured phrases (built-in)',
    },
    'shakespeare': {
        'source': 'url',
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'filename': 'shakespeare.txt',
        'description': 'Tiny Shakespeare (~1MB, character-level)',
    },
    'tinystories': {
        'source': 'huggingface',
        'path': 'roneneldan/TinyStories',
        'split': 'train',
        'val_split': 'validation',
        'filename': 'tinystories.txt',
        'val_filename': 'tinystories_val.txt',
        'description': 'TinyStories (HuggingFace, real children\'s stories)',
        # Fallback URL if HF datasets library is not installed
        'fallback_url': None,  # Too large for raw URL fallback
    },
}


def _ensure_data_dir():
    """Create data/ directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _download_url(url, filepath):
    """Download a file from URL using urllib (stdlib)."""
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  Saved to {filepath} ({os.path.getsize(filepath)} bytes)")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def _generate_synthetic_text():
    """Generate synthetic text with Fibonacci-structured repetitions."""
    base_phrases = [
        "the golden ratio appears in nature ",
        "fibonacci numbers grow exponentially ",
        "symmetry underlies all of physics ",
        "the icosahedron has twenty faces ",
        "phi equals one plus one over phi ",
        "geometry is the language of space ",
        "five fold symmetry cannot tile a plane ",
        "the dodecahedron has twelve faces ",
    ]
    text = ""
    a, b = 1, 1
    for _ in range(200):
        phrase = base_phrases[a % len(base_phrases)]
        text += phrase * (b % 3 + 1)
        a, b = b, a + b
    return text


def _load_shakespeare():
    """Download and return Tiny Shakespeare text."""
    _ensure_data_dir()
    cfg = DATASETS['shakespeare']
    filepath = os.path.join(DATA_DIR, cfg['filename'])

    if not os.path.exists(filepath):
        if not _download_url(cfg['url'], filepath):
            print("Shakespeare download failed, falling back to synthetic data.")
            return None

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded Shakespeare: {len(text):,} chars")
    return text


def _load_tinystories():
    """Load TinyStories from HuggingFace datasets or cached files."""
    _ensure_data_dir()
    cfg = DATASETS['tinystories']
    train_path = os.path.join(DATA_DIR, cfg['filename'])
    val_path = os.path.join(DATA_DIR, cfg['val_filename'])

    # Check cache first
    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        with open(val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
        print(f"Loaded TinyStories from cache: train={len(train_text):,} chars, val={len(val_text):,} chars")
        return train_text, val_text

    # Try HuggingFace datasets library
    try:
        from datasets import load_dataset as hf_load_dataset
        print("Loading TinyStories from HuggingFace (this may take a while)...")
        ds = hf_load_dataset(cfg['path'])

        # Extract text — TinyStories has a 'text' field
        # Limit to first 5M chars for manageability on CPU
        MAX_CHARS = 5_000_000
        train_text = ""
        for item in ds[cfg['split']]:
            train_text += item['text'] + "\n"
            if len(train_text) >= MAX_CHARS:
                train_text = train_text[:MAX_CHARS]
                break

        val_text = ""
        for item in ds[cfg['val_split']]:
            val_text += item['text'] + "\n"
            if len(val_text) >= MAX_CHARS // 10:
                val_text = val_text[:MAX_CHARS // 10]
                break

        # Cache to disk
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(train_text)
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write(val_text)

        print(f"TinyStories loaded and cached: train={len(train_text):,} chars, val={len(val_text):,} chars")
        return train_text, val_text

    except ImportError:
        print("HuggingFace 'datasets' library not installed.")
        print("Install with: pip install datasets")
        print("Falling back to synthetic data.")
        return None
    except Exception as e:
        print(f"Failed to load TinyStories: {e}")
        print("Falling back to synthetic data.")
        return None


def prepare_char_dataset(text, val_text=None):
    """Prepare character-level dataset from text.

    Returns:
        (train_data, val_data, vocab_size, stoi, itos)
    """
    if val_text is not None:
        # Pre-split data: build vocab from both
        all_text = text + val_text
    else:
        all_text = text

    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    if val_text is not None:
        train_data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        val_data = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    else:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

    return train_data, val_data, vocab_size, stoi, itos


def load_dataset(name='shakespeare'):
    """Load a dataset by name. Returns raw text (or tuple for pre-split datasets).

    For use with train_cpu.py's load_text_data() replacement.

    Args:
        name: 'synthetic', 'shakespeare', or 'tinystories'

    Returns:
        text (str) for single-text datasets, or
        (train_text, val_text) for pre-split datasets, or
        None on failure (caller should fall back to synthetic)
    """
    if name == 'synthetic':
        return _generate_synthetic_text()
    elif name == 'shakespeare':
        return _load_shakespeare()
    elif name == 'tinystories':
        return _load_tinystories()
    else:
        print(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
        return None


def load_and_prepare(name='shakespeare'):
    """Full pipeline: download, tokenize, return ready-to-train tensors.

    Returns:
        (train_data, val_data, vocab_size, stoi, itos)
    """
    result = load_dataset(name)

    if result is None:
        # Fall back to synthetic
        print("Using synthetic fallback data.")
        text = _generate_synthetic_text()
        return prepare_char_dataset(text)

    if isinstance(result, tuple):
        # Pre-split dataset (e.g., TinyStories)
        train_text, val_text = result
        return prepare_char_dataset(train_text, val_text)
    else:
        # Single text, will be split 90/10
        return prepare_char_dataset(result)


def list_datasets():
    """Print available datasets."""
    print("Available datasets:")
    for name, cfg in DATASETS.items():
        cached = ""
        if cfg['source'] == 'url':
            path = os.path.join(DATA_DIR, cfg.get('filename', ''))
            if os.path.exists(path):
                cached = f" [cached: {os.path.getsize(path):,} bytes]"
        elif cfg['source'] == 'huggingface':
            path = os.path.join(DATA_DIR, cfg.get('filename', ''))
            if os.path.exists(path):
                cached = f" [cached: {os.path.getsize(path):,} bytes]"
        print(f"  {name:15s} — {cfg['description']}{cached}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare datasets for H4 experiments')
    parser.add_argument('dataset', nargs='?', default='shakespeare',
                        choices=list(DATASETS.keys()),
                        help='Dataset to prepare (default: shakespeare)')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    args = parser.parse_args()

    if args.list:
        list_datasets()
        sys.exit(0)

    train_data, val_data, vocab_size, stoi, itos = load_and_prepare(args.dataset)
    print(f"\nDataset: {args.dataset}")
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Sample chars: {''.join(itos[i] for i in train_data[:80].tolist())}")
