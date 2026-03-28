"""
LLM Hypothesis Generator for ARC-AGI puzzles.

Uses the Qwen2.5-Coder-7B specialist (via GGUF) to generate C program
hypotheses from training pair descriptions. The verifier catches bad code
— the model just needs to be right once out of N attempts.

Strategy:
1. Format training pairs as a structured prompt
2. Ask the model to write a C function that transforms input to output
3. Parse the C code from the response
4. Wrap in the standard ARC program template
5. Let the verifier compile and check

The model doesn't need to understand transformer-vm constraints.
We post-process the code: replace multiplication with addition loops,
replace stdlib calls with our runtime helpers, etc.
"""

import logging
import re
import time
from typing import List, Optional

from .grid_io import Grid, format_grid
from .hypothesizer import Hypothesis, ARC_HEADER, _dimensions

logger = logging.getLogger(__name__)

# ── Prompt template ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at solving ARC-AGI puzzles. Each puzzle has training pairs showing input grids transformed to output grids. Your job is to find the transformation rule and implement it as a C function.

The C function interface is:
- Input: grid[] array (row-major, h rows, w cols), values 0-9
- Output: out[] array, set oh (output height) and ow (output width)
- Helper: arc_get(grid, w, row, col) reads a cell
- Helper: arc_set(out, ow, row, col, value) writes a cell
- Helper: arc_fill(out, h, w, value) fills entire grid
- No multiplication operator (use addition loops: for(i=0;i<a;i++) result += b)
- All variables must be int, declared at start of block
- Arrays are pre-allocated as static (grid[900], out[900])

Write ONLY the body of the compute function (the part between the braces).
The code will be inserted into this template:

void compute(const char *input) {
    int *grid = _grid;
    int *out = _out;
    arc_parse_grid(input, grid, _h_arr, _w_arr);
    int h = _h_arr[0], w = _w_arr[0];
    int oh, ow;

    // YOUR CODE HERE

    arc_emit_grid(out, oh, ow);
}"""

TASK_PROMPT_TEMPLATE = """ARC puzzle. Write ONLY the C code body to transform input grid to output grid.

Variables already declared: int *grid (input), int *out (output), int h (rows), int w (cols), int oh, ow.
Use: arc_get(grid, w, row, col) to read, arc_set(out, ow, row, col, val) to write.
Set oh and ow for output dimensions. No multiplication operator - use loops.

EXAMPLE for a color swap puzzle (swap color 1 with color 2):
```c
    oh = h; ow = w;
    int r, c;
    for (r = 0; r < h; r++) {{
        for (c = 0; c < w; c++) {{
            int v = arc_get(grid, w, r, c);
            if (v == 1) v = 2;
            else if (v == 2) v = 1;
            arc_set(out, ow, r, c, v);
        }}
    }}
```

Training pairs:
{pairs_text}

Write ONLY the C code body (no function definition, no includes, no main):"""


def format_pairs_for_llm(task: dict) -> str:
    """Format training pairs as text for the LLM prompt."""
    lines = []
    for i, pair in enumerate(task["train"]):
        inp, out = pair["input"], pair["output"]
        h_i, w_i = _dimensions(inp)
        h_o, w_o = _dimensions(out)
        lines.append(f"Training pair {i+1} ({h_i}x{w_i} -> {h_o}x{w_o}):")
        lines.append(f"Input:")
        for row in inp:
            lines.append("  " + " ".join(str(v) for v in row))
        lines.append(f"Output:")
        for row in out:
            lines.append("  " + " ".join(str(v) for v in row))
        lines.append("")
    return "\n".join(lines)


# ── C code extraction and fixing ─────────────────────────────────

def extract_c_body(response: str) -> Optional[str]:
    """Extract C code body from LLM response.

    The model may generate:
    - Just the body (ideal)
    - A full function definition (void compute/transform/main)
    - A full program with #includes

    We need to extract just the body that goes between our template braces.
    """
    # Try fenced code block first
    blocks = re.findall(r'```(?:c|C)?\s*\n(.*?)```', response, re.DOTALL)
    if blocks:
        code = blocks[-1].strip()
    else:
        # Try to find code-like lines
        code_lines = []
        in_code = False
        for line in response.split('\n'):
            stripped = line.strip()
            if any(stripped.startswith(kw) for kw in
                   ['oh ', 'oh=', 'ow ', 'ow=', 'int ', 'for ', 'for(', 'if ',
                    'if(', 'while', '{', '}', 'arc_', '/*', '//',
                    'void ', 'static ']):
                in_code = True
            if in_code:
                code_lines.append(line)
        code = "\n".join(code_lines)

    if not code:
        return None

    # Remove #include, #define, #ifndef etc.
    code = re.sub(r'#\s*(?:include|define|ifndef|endif|ifdef)\s*[^\n]*', '', code)

    # Remove static array declarations (our template provides these)
    code = re.sub(r'static\s+int\s+\w+\[\d+\]\s*;', '', code)

    # Extract body from function definitions
    # Match: void compute(...) { ... } or void transform_grid(...) { ... } or int main() { ... }
    func_match = re.search(
        r'(?:void|int)\s+\w+\s*\([^)]*\)\s*\{(.*)',
        code, re.DOTALL
    )
    if func_match:
        body = func_match.group(1)
        # Find the matching closing brace
        depth = 1
        end = 0
        for i, ch in enumerate(body):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > 0:
            code = body[:end]
        else:
            code = body  # No matching brace found, take everything

    # Remove arc_parse_grid/arc_emit_grid calls (template handles these)
    code = re.sub(r'arc_parse_grid\([^)]*\)\s*;', '', code)
    code = re.sub(r'arc_emit_grid\([^)]*\)\s*;', '', code)

    # Remove variable declarations that duplicate our template
    code = re.sub(r'int\s+\*grid\s*=\s*_grid\s*;', '', code)
    code = re.sub(r'int\s+\*out\s*=\s*_out\s*;', '', code)
    code = re.sub(r'int\s+h\s*=\s*_h_arr\[0\]\s*[,;]', '', code)
    code = re.sub(r'int\s+w\s*=\s*_w_arr\[0\]\s*;', '', code)

    # Remove duplicate function definitions inside body
    code = re.sub(r'(?:void|int)\s+\w+\s*\([^)]*\)\s*\{', '/* removed nested func */ {', code)

    # Clean up excess whitespace
    code = re.sub(r'\n{3,}', '\n\n', code)

    # Remove trailing unbalanced braces
    code = code.rstrip()
    while code:
        opens = code.count('{')
        closes = code.count('}')
        if closes > opens and code.endswith('}'):
            code = code[:code.rfind('}')].rstrip()
        else:
            break

    return code.strip() if code.strip() else None


def fix_c_code(code: str) -> str:
    """Post-process C code to fix common issues for transformer-vm."""
    # Remove assert calls
    code = re.sub(r'assert\s*\([^)]*\)\s*;', '', code)

    # Remove return 0; (void function)
    code = re.sub(r'return\s+0\s*;', '', code)

    # Remove common stdlib calls
    for fn in ['printf', 'malloc', 'free', 'memset', 'memcpy', 'abs',
               'arc_get_dims', 'transform_grid']:
        code = re.sub(rf'{fn}\s*\([^)]*\)\s*;', f'/* {fn} removed */', code)

    # Remove redeclarations of template variables
    code = re.sub(r'\bint\s+grid\s*\[', '/* int grid[ */', code)
    code = re.sub(r'\bstatic\s+int\s+grid\s*\[', '/* static int grid[ */', code)
    code = re.sub(r'\bint\s+out\s*\[', '/* int out[ */', code)
    code = re.sub(r'\bint\s+w\s*,\s*h\s*;', '/* int w, h; */', code)
    code = re.sub(r'\bint\s+h\s*,\s*w\s*;', '/* int h, w; */', code)
    code = re.sub(r'\bint\s+ow\s*,\s*oh\s*;', '/* int ow, oh; */', code)
    code = re.sub(r'\bint\s+oh\s*,\s*ow\s*;', '/* int oh, ow; */', code)

    # Replace r * w with addition loop helper
    # Common pattern: grid[r * w + c] -> arc_get(grid, w, r, c)
    # This is too complex for regex; leave for model to handle

    return code


def wrap_as_hypothesis(c_body: str, description: str) -> Hypothesis:
    """Wrap extracted C body into a full hypothesis."""
    preamble = f'''/* ARC rule: {description}
 * Generated by LLM hypothesis generator.
 */
#include "{ARC_HEADER}"

/* Static arrays to avoid WASM stack overflow (4KB stack limit) */
static int _grid[900];
static int _out[900];
static int _tmp[32];
static int _qr[900];
static int _qc[900];
static int _h_arr[1];
static int _w_arr[1];

void compute(const char *input) {{
    int *grid = _grid;
    int *out = _out;
    arc_parse_grid(input, grid, _h_arr, _w_arr);
    int h = _h_arr[0], w = _w_arr[0];
    int oh, ow;

{c_body}

    arc_emit_grid(out, oh, ow);
}}
'''
    return Hypothesis("llm_generated", description, preamble, confidence=0.5)


# ── LLM hypothesis generator ────────────────────────────────────

class LLMHypothesizer:
    """Generate ARC hypotheses using a local LLM specialist."""

    def __init__(self, n_ctx=4096, n_threads=None):
        self.engine = None
        self.n_ctx = n_ctx
        self.n_threads = n_threads

    def _ensure_engine(self):
        if self.engine is not None:
            return True
        try:
            from olympus.gguf_inference import GGUFEngine
            self.engine = GGUFEngine(n_ctx=self.n_ctx, n_threads=self.n_threads)
            if not self.engine.load("code"):
                logger.warning("Failed to load code specialist")
                self.engine = None
                return False
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize LLM engine: {e}")
            return False

    def generate_hypotheses(self, task: dict,
                            temperatures: list = None,
                            max_tokens: int = 1500) -> List[Hypothesis]:
        """Generate hypotheses by prompting the LLM at multiple temperatures.

        Returns list of Hypothesis objects (one per successful extraction).
        """
        if not self._ensure_engine():
            return []

        if temperatures is None:
            temperatures = [0.3, 0.6, 0.9]

        pairs_text = format_pairs_for_llm(task)
        prompt = TASK_PROMPT_TEMPLATE.format(pairs_text=pairs_text)

        hypotheses = []
        seen_codes = set()  # Deduplicate

        for temp in temperatures:
            try:
                response = self.engine.generate(
                    prompt,
                    specialist="code",
                    max_tokens=max_tokens,
                    temperature=temp,
                    stop=["<|im_end|>", "<|im_start|>"],
                )

                if not response:
                    continue

                c_body = extract_c_body(response)
                if c_body is None:
                    logger.info(f"No C code extracted at temp={temp}")
                    continue

                c_body = fix_c_code(c_body)

                # Deduplicate
                code_key = c_body.strip()
                if code_key in seen_codes:
                    continue
                seen_codes.add(code_key)

                desc = f"LLM hypothesis (temp={temp})"
                hyp = wrap_as_hypothesis(c_body, desc)
                hypotheses.append(hyp)
                logger.info(f"Generated hypothesis at temp={temp}: {len(c_body)} chars")

            except Exception as e:
                logger.warning(f"LLM generation failed at temp={temp}: {e}")

        return hypotheses


def generate_llm_hypotheses(task: dict, llm: LLMHypothesizer = None,
                            temperatures: list = None) -> List[Hypothesis]:
    """Convenience function for generating LLM hypotheses."""
    if llm is None:
        llm = LLMHypothesizer()
    return llm.generate_hypotheses(task, temperatures=temperatures)
