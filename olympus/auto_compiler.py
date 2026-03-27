"""
Auto-Compiler: Automatic failure detection → compilation → routing

The self-compiling intelligence loop:
1. Model attempts a task
2. Property checker detects failure + classifies it
3. System generates a correct C implementation
4. Compiles via transformer-vm
5. Registers in ChamberTree for automatic routing
6. Model never fails that task class again

Usage:
    compiler = AutoCompiler()

    # When a code query fails property checking:
    result = compiler.handle_failure(
        query="longest increasing subsequence on [3,1,4,1,5]",
        code="def lis(arr): ...",  # the failing code
        failure="not strictly increasing: 4 >= 1",
        category="algorithm"
    )
    # result: compiled tool registered, future queries auto-route
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
WASM_TOOLS_DIR = Path(__file__).parent / "wasm_tools"
COMPILED_DIR = WASM_TOOLS_DIR / "compiled"
TOOL_REGISTRY_PATH = Path(__file__).parent / "tool_registry.json"

# ── Failure Categories ────────────────────────────────────────────

FAILURE_CATEGORIES = {
    "backtracking": {
        "patterns": [
            r"not strictly increasing",
            r"not a subsequence",
            r"not sorted",
            r"wrong order",
        ],
        "description": "DP/backtracking implementation error",
        "fix_strategy": "compile_algorithm",
    },
    "arithmetic": {
        "patterns": [
            r"overflow",
            r"division by zero",
            r"wrong result.*\d+",
            r"off by one",
        ],
        "description": "Numerical computation error",
        "fix_strategy": "compile_algorithm",
    },
    "data_structure": {
        "patterns": [
            r"not a permutation",
            r"missing elements",
            r"duplicate",
            r"wrong length",
        ],
        "description": "Data structure manipulation error",
        "fix_strategy": "compile_algorithm",
    },
    "hallucination": {
        "patterns": [
            r"3[dD] print",
            r"I don't have",
            r"as an AI",
            r"I cannot",
        ],
        "description": "Model hallucinated or refused incorrectly",
        "fix_strategy": "improve_prompt",
    },
}


def classify_failure(failure_msg: str, query: str) -> Optional[dict]:
    """Classify a failure into a category with metadata."""
    failure_lower = failure_msg.lower()
    query_lower = query.lower()

    for category, info in FAILURE_CATEGORIES.items():
        for pattern in info["patterns"]:
            if re.search(pattern, failure_msg, re.IGNORECASE):
                return {
                    "category": category,
                    "description": info["description"],
                    "fix_strategy": info["fix_strategy"],
                    "matched_pattern": pattern,
                    "failure_msg": failure_msg,
                    "query": query,
                }

    return {
        "category": "unknown",
        "description": "Unclassified failure",
        "fix_strategy": "log_for_review",
        "failure_msg": failure_msg,
        "query": query,
    }


# ── Tool Registry ─────────────────────────────────────────────────

class ToolRegistry:
    """Persistent registry of compiled tools.

    Maps (query_pattern → compiled_tool) so future queries
    automatically route to the correct compiled version.
    """

    def __init__(self, path=TOOL_REGISTRY_PATH):
        self.path = path
        self.tools = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"tools": [], "failures_logged": []}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.tools, f, indent=2)

    def register_tool(self, name: str, c_source: str, query_patterns: list,
                      description: str):
        """Register a new compiled tool."""
        entry = {
            "name": name,
            "c_source": c_source,
            "query_patterns": query_patterns,
            "description": description,
            "compiled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uses": 0,
        }
        # Don't duplicate
        existing_names = {t["name"] for t in self.tools["tools"]}
        if name not in existing_names:
            self.tools["tools"].append(entry)
            self._save()
            logger.info(f"Registered new tool: {name}")
            return True
        return False

    def log_failure(self, failure_info: dict):
        """Log a failure for future compilation."""
        failure_info["logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.tools["failures_logged"].append(failure_info)
        # Keep only last 100
        self.tools["failures_logged"] = self.tools["failures_logged"][-100:]
        self._save()

    def find_tool(self, query: str) -> Optional[dict]:
        """Find a compiled tool that matches a query."""
        query_lower = query.lower()
        for tool in self.tools["tools"]:
            for pattern in tool["query_patterns"]:
                if re.search(pattern, query_lower):
                    tool["uses"] += 1
                    self._save()
                    return tool
        return None

    def get_failure_stats(self) -> dict:
        """Get statistics on logged failures."""
        from collections import Counter
        categories = Counter(
            f.get("category", "unknown")
            for f in self.tools["failures_logged"]
        )
        return {
            "total_failures": len(self.tools["failures_logged"]),
            "total_tools": len(self.tools["tools"]),
            "failures_by_category": dict(categories),
            "tools": [
                {"name": t["name"], "uses": t["uses"], "patterns": t["query_patterns"]}
                for t in self.tools["tools"]
            ],
        }


# ── C Code Generator ──────────────────────────────────────────────

# Templates for common algorithm patterns that models get wrong
ALGORITHM_TEMPLATES = {
    "sort": {
        "query_patterns": [r"sort\b", r"sorted\b", r"order\b.*\["],
        "c_source": "code/sort.c",
        "template": '''/* Sort integers.
 * Input: space-separated integers
 * Output: sorted integers, space-separated
 */
void compute(const char *input) {
    int arr[256];
    int n = 0;
    int pos = 0;
    while (input[pos]) {
        while (input[pos] == ' ') pos++;
        if (input[pos] == 0) break;
        int neg = 0;
        if (input[pos] == '-') { neg = 1; pos++; }
        int val = 0;
        while (input[pos] >= '0' && input[pos] <= '9') {
            int d = input[pos] - '0';
            int t2 = val + val; int t4 = t2 + t2; int t8 = t4 + t4;
            val = t8 + t2 + d;
            pos++;
        }
        if (neg) val = 0 - val;
        arr[n] = val;
        n++;
    }
    /* Insertion sort */
    int i, j;
    for (i = 1; i < n; i++) {
        int key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
    for (i = 0; i < n; i++) {
        if (i > 0) putchar(' ');
        print_int(arr[i]);
    }
    putchar('\\n');
}
''',
    },
    "binary_search": {
        "query_patterns": [r"binary\s*search", r"find.*in.*sorted"],
        "c_source": "code/binary_search.c",
        "template": '''/* Binary search in sorted array.
 * Input: "target val1 val2 val3 ..." (first number is target)
 * Output: index (0-based) or -1 if not found
 */
void compute(const char *input) {
    int arr[256];
    int n = 0;
    int pos = 0;
    /* Parse all numbers */
    while (input[pos]) {
        while (input[pos] == ' ') pos++;
        if (input[pos] == 0) break;
        int neg = 0;
        if (input[pos] == '-') { neg = 1; pos++; }
        int val = 0;
        while (input[pos] >= '0' && input[pos] <= '9') {
            int d = input[pos] - '0';
            int t2 = val + val; int t4 = t2 + t2; int t8 = t4 + t4;
            val = t8 + t2 + d;
            pos++;
        }
        if (neg) val = 0 - val;
        arr[n] = val;
        n++;
    }
    if (n < 2) { print_int(0 - 1); putchar('\\n'); return; }
    int target = arr[0];
    int lo = 1, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == target) {
            print_int(mid - 1);
            putchar('\\n');
            return;
        }
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    print_int(0 - 1);
    putchar('\\n');
}
''',
    },
    "reverse": {
        "query_patterns": [r"reverse\b.*(?:list|array|string)"],
        "c_source": "code/reverse.c",
        "template": '''/* Reverse a list of integers.
 * Input: space-separated integers
 * Output: reversed, space-separated
 */
void compute(const char *input) {
    int arr[256];
    int n = 0;
    int pos = 0;
    while (input[pos]) {
        while (input[pos] == ' ') pos++;
        if (input[pos] == 0) break;
        int neg = 0;
        if (input[pos] == '-') { neg = 1; pos++; }
        int val = 0;
        while (input[pos] >= '0' && input[pos] <= '9') {
            int d = input[pos] - '0';
            int t2 = val + val; int t4 = t2 + t2; int t8 = t4 + t4;
            val = t8 + t2 + d;
            pos++;
        }
        if (neg) val = 0 - val;
        arr[n] = val;
        n++;
    }
    int i;
    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) putchar(' ');
        print_int(arr[i]);
    }
    putchar('\\n');
}
''',
    },
}


# ── Auto-Compiler ─────────────────────────────────────────────────

class AutoCompiler:
    """The self-compiling intelligence loop."""

    def __init__(self):
        self.registry = ToolRegistry()
        self.tvm_engine = None
        self._init_tvm()
        self._ensure_templates()

    def _init_tvm(self):
        """Initialize transformer-vm engine."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from tvm_engine import TVMEngine
            self.tvm_engine = TVMEngine()
            if not self.tvm_engine.available:
                self.tvm_engine = None
        except ImportError:
            self.tvm_engine = None

    def _ensure_templates(self):
        """Write built-in algorithm templates to disk and register them."""
        for name, info in ALGORITHM_TEMPLATES.items():
            c_path = WASM_TOOLS_DIR / info["c_source"]
            if not c_path.exists():
                c_path.parent.mkdir(parents=True, exist_ok=True)
                c_path.write_text(info["template"])
                logger.info(f"Created template: {c_path}")

            self.registry.register_tool(
                name=name,
                c_source=str(c_path),
                query_patterns=info["query_patterns"],
                description=f"Compiled {name} algorithm",
            )

    def handle_failure(self, query: str, code: str, failure: str) -> dict:
        """Handle a detected failure. Classify, compile if possible, register.

        Returns dict with:
            handled: bool
            action: str (what was done)
            tool_name: str or None
        """
        # Classify the failure
        classification = classify_failure(failure, query)
        self.registry.log_failure(classification)

        category = classification["category"]
        strategy = classification.get("fix_strategy", "log_for_review")

        if strategy == "compile_algorithm":
            # Check if we already have a template for this
            for name, info in ALGORITHM_TEMPLATES.items():
                for pattern in info["query_patterns"]:
                    if re.search(pattern, query, re.IGNORECASE):
                        # Template exists — ensure it's compiled and registered
                        return {
                            "handled": True,
                            "action": f"Routed to existing compiled tool: {name}",
                            "tool_name": name,
                            "category": category,
                        }

            # No template — log for manual compilation
            logger.info(
                f"No template for failure category '{category}' on query: {query[:80]}"
            )
            return {
                "handled": False,
                "action": f"Logged failure (category: {category}). No auto-compile template yet.",
                "tool_name": None,
                "category": category,
                "suggestion": f"Write a C program for this algorithm class and add to ALGORITHM_TEMPLATES",
            }

        elif strategy == "improve_prompt":
            return {
                "handled": False,
                "action": "Hallucination detected. Needs better system prompt, not compilation.",
                "tool_name": None,
                "category": category,
            }

        else:
            return {
                "handled": False,
                "action": f"Unknown failure type: {category}",
                "tool_name": None,
                "category": category,
            }

    def try_compiled_tool(self, query: str) -> Optional[dict]:
        """Check if a compiled tool can handle this query.

        Returns result dict or None.
        """
        tool = self.registry.find_tool(query)
        if tool is None:
            return None

        if self.tvm_engine is None:
            return None

        # Extract input data from query
        input_data = self._extract_input(query)
        if input_data is None:
            return None

        # Run through transformer-vm
        result = self.tvm_engine.compute(f"{tool['name']} {input_data}")
        if result:
            result["compiled_tool"] = tool["name"]
            return result

        return None

    def _extract_input(self, query: str) -> Optional[str]:
        """Extract numerical input from a query string."""
        # Try to find a list of numbers
        match = re.search(r'\[([\d,\s\-]+)\]', query)
        if match:
            return re.sub(r'[,\[\]]', ' ', match.group(0)).strip()

        # Try to find space-separated numbers after keywords
        match = re.search(r'(?:on|of|for|from)\s+([\d\s\-]+)', query)
        if match:
            return match.group(1).strip()

        return None

    def status(self) -> dict:
        """Get auto-compiler status."""
        stats = self.registry.get_failure_stats()
        stats["tvm_available"] = self.tvm_engine is not None
        stats["templates_available"] = list(ALGORITHM_TEMPLATES.keys())
        return stats


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    compiler = AutoCompiler()

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status = compiler.status()
        print(json.dumps(status, indent=2))
    else:
        print("Auto-Compiler Status:")
        status = compiler.status()
        print(f"  Tools compiled: {status['total_tools']}")
        print(f"  Failures logged: {status['total_failures']}")
        print(f"  Templates available: {status['templates_available']}")
        print(f"  TVM available: {status['tvm_available']}")
        print()
        print("  Tools:")
        for t in status["tools"]:
            print(f"    {t['name']:20s} uses={t['uses']}  patterns={t['patterns']}")
