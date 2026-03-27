"""
GGUF Inference Engine for Olympus Specialists

Fast CPU inference using llama-cpp-python.
Expected speedup: 15-30s (HF fp32) -> 4-8s (GGUF Q4_K_M) per response.

Usage:
    from gguf_inference import GGUFEngine

    engine = GGUFEngine()
    engine.load("code")  # Loads GGUF model
    response = engine.generate("Write a bubble sort in Python")

Install: pip install llama-cpp-python
"""

import os
from pathlib import Path
from typing import Optional

# Monkey-patch llama-cpp-python to handle SmolLM3's custom Jinja tags
try:
    import llama_cpp.llama_chat_format as _fmt
    _orig_jinja_init = _fmt.Jinja2ChatFormatter.__init__
    def _safe_jinja_init(self, *args, **kwargs):
        try:
            _orig_jinja_init(self, *args, **kwargs)
        except Exception:
            kwargs2 = dict(kwargs)
            kwargs2['template'] = (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                "{% endfor %}"
                "<|im_start|>assistant\n"
            )
            _orig_jinja_init(self, *args, **kwargs2)
    _fmt.Jinja2ChatFormatter.__init__ = _safe_jinja_init
except Exception:
    pass

GGUF_DIR = Path(__file__).parent.parent / "checkpoints" / "gguf"
DEFAULT_QUANT = "q4_k_m"

# Chat templates per model family
CHAT_TEMPLATE_CHATML = "<|im_start|>system\nYou are a helpful assistant specialized in {specialist}.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

SPECIALIST_SYSTEM = {
    "general": (
        "general knowledge and conversation. You are Lattice, a local AI assistant "
        "running on the user's computer via Project Olympus. You are a 3B parameter "
        "SmolLM3 model with LoRA specialization. You have three specialist modes "
        "(code, math, QA) and exact computation via transformer-vm. You run locally "
        "with no cloud, no API, no cost. If you don't know something, say so honestly "
        "rather than guessing. Be concise"
    ),
    "code": (
        "writing clean, correct code. Always include a test with assert to verify "
        "correctness. Show code first, explain briefly after. You are the code "
        "specialist of Lattice, a local AI running via Project Olympus"
    ),
    "math": (
        "mathematical reasoning. Show the solution steps and final answer concisely. "
        "You are the math specialist of Lattice, a local AI running via Project Olympus. "
        "For exact arithmetic, the system uses transformer-vm (not you)"
    ),
    "qa": (
        "answering questions accurately. Give a direct answer first, then brief "
        "explanation. You are the QA specialist of Lattice, a local AI running via "
        "Project Olympus. If you don't know a fact, say so rather than guessing"
    ),
}

# Code specialist uses Qwen2.5-Coder-7B if available, otherwise falls back to SmolLM3
QWEN_CODER_GGUF = GGUF_DIR / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"


class GGUFEngine:
    """Manages GGUF model loading and inference for Olympus specialists."""

    def __init__(self, n_ctx=2048, n_threads=None):
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, os.cpu_count() // 2)
        self._models = {}  # specialist_name -> Llama instance
        self._current = None

    def available_specialists(self):
        """Return list of specialists with GGUF files ready."""
        available = []
        for name in ["general", "code", "math", "qa"]:
            path = self._gguf_path(name)
            if path and path.exists():
                available.append(name)
        return available

    def _gguf_path(self, specialist: str) -> Optional[Path]:
        """Find GGUF file for a specialist, trying common quantization names."""
        # Code specialist: prefer Qwen2.5-Coder-7B if available
        if specialist == "code" and QWEN_CODER_GGUF.exists():
            return QWEN_CODER_GGUF

        candidates = [
            GGUF_DIR / f"olympus-{specialist}-{DEFAULT_QUANT}.gguf",
            GGUF_DIR / f"olympus-{specialist}-q5_k_m.gguf",
            GGUF_DIR / f"olympus-{specialist}-q8_0.gguf",
            GGUF_DIR / f"olympus-{specialist}-f16.gguf",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def is_available(self, specialist: str) -> bool:
        """Check if GGUF model exists for this specialist."""
        path = self._gguf_path(specialist)
        return path is not None and path.exists()

    def load(self, specialist: str) -> bool:
        """Load a specialist GGUF model. Returns True on success."""
        if specialist in self._models:
            self._current = specialist
            return True

        path = self._gguf_path(specialist)
        if path is None or not path.exists():
            return False

        try:
            from llama_cpp import Llama
        except ImportError:
            print("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return False

        print(f"Loading GGUF model: {path.name} ({path.stat().st_size / (1024**2):.0f}MB)")

        # Unload previous model to save memory (keep max 1 loaded)
        if self._current and self._current != specialist:
            self._unload(self._current)

        self._models[specialist] = Llama(
            model_path=str(path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        self._current = specialist
        print(f"Loaded {specialist} specialist ({self.n_threads} threads, {self.n_ctx} ctx)")
        return True

    def _unload(self, specialist: str):
        """Unload a model to free memory."""
        if specialist in self._models:
            del self._models[specialist]
            self._current = None

    def generate(
        self,
        prompt: str,
        specialist: Optional[str] = None,
        context: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list] = None,
    ) -> str:
        """Generate a response using the loaded GGUF model."""
        specialist = specialist or self._current
        if specialist is None:
            return "[No specialist loaded]"

        if specialist not in self._models:
            if not self.load(specialist):
                return f"[{specialist} specialist GGUF not available]"

        model = self._models[specialist]

        # Build prompt with chat template
        if context:
            user_msg = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            user_msg = prompt

        full_prompt = CHAT_TEMPLATE_CHATML.format(
            specialist=SPECIALIST_SYSTEM.get(specialist, "general tasks"),
            prompt=user_msg,
        )

        if stop is None:
            stop = ["<|im_end|>", "<|im_start|>"]

        output = model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
        )

        return output["choices"][0]["text"].strip()

    def status(self) -> dict:
        """Return status of all specialists."""
        result = {}
        for name in ["general", "code", "math", "qa"]:
            path = self._gguf_path(name)
            if name in self._models:
                result[name] = "loaded"
            elif path and path.exists():
                size_mb = path.stat().st_size / (1024**2)
                model_name = path.stem
                result[name] = f"ready ({model_name}, {size_mb:.0f}MB)"
            else:
                result[name] = "not converted"
        return result
