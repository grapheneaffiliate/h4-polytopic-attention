"""
Phase 3: Hybrid LLM Integration — Claude (Max Plan) + H4 Executor
===================================================================

Uses the Claude Agent SDK to leverage Max plan OAuth authentication.
Claude handles reasoning/planning; H4 executor handles exact computation
via custom MCP tools.

Run:
  py python/hybrid_llm.py --demo     # Non-interactive demo
  py python/hybrid_llm.py            # Interactive chat

Author: Timothy McGirl
"""

import os
import sys
import json
import anyio
from typing import List, Dict, Any

from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

# Import our H4 components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weight_compiler import (
    Program, fibonacci_program, H4Executor,
    StateEncoder, CompiledTransformer,
    generate_600_cell_vertices, h4_simple_roots,
    PHI, PHI_INV,
)
import numpy as np


# ============================================================
# MCP Tool Definitions (custom tools for the H4 executor)
# ============================================================

@tool(
    "h4_fibonacci",
    "Compute the Fibonacci sequence using the H4 polytopic attention transformer executor. Returns F(0) through F(n+1). The computation runs through analytically constructed transformer weights with 4D H4 attention heads.",
    {"n": int},
)
async def h4_fibonacci(args):
    n = args["n"]
    prog = fibonacci_program(min(n, 30))  # cap at 30 to avoid huge numbers
    executor = H4Executor(prog, d_model=32)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = executor.run(max_steps=500)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    # Build sequence from trace
    seen = set()
    fib_sequence = []
    for regs in executor.register_history:
        v = int(regs[1])
        if v not in seen:
            seen.add(v)
            fib_sequence.append(v)

    expected = [0, 1]
    for _ in range(n):
        expected.append(expected[-1] + expected[-2])

    output = {
        "fibonacci_n": n,
        "result_F_n_plus_1": int(result['registers'][1]),
        "expected_F_n_plus_1": expected[min(n, 30) + 1],
        "correct": int(result['registers'][1]) == expected[min(n, 30) + 1],
        "execution_steps": result['steps'],
        "trace_length": result['trace_length'],
        "sequence_from_trace": fib_sequence[:n + 2],
        "final_registers": [int(r) for r in result['registers'][:6]],
        "transformer": {"d_model": 32, "n_heads": 8, "n_layers": 4, "head_dim": "4D H4"},
    }
    return {"content": [{"type": "text", "text": json.dumps(output, indent=2)}]}


@tool(
    "h4_compile_and_run",
    "Compile and run a custom program on the H4 transformer executor. ISA: LOAD (immediate to register), ADD, SUB, MUL (register arithmetic), STORE (copy register), JMP, JNZ, HALT. 8 registers (R0-R7). Pass instructions as a JSON array.",
    {"instructions": str, "max_steps": int},
)
async def h4_compile_and_run(args):
    instructions = json.loads(args["instructions"]) if isinstance(args["instructions"], str) else args["instructions"]
    max_steps = args.get("max_steps", 500)

    prog = Program()
    for instr in instructions:
        prog.add(
            instr.get("opcode", "HALT"),
            a=instr.get("a", 0),
            b=instr.get("b", 0),
            dest=instr.get("dest", 0),
        )

    executor = H4Executor(prog, d_model=32)
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = executor.run(max_steps=max_steps)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    output = {
        "program_length": len(prog),
        "execution_steps": result['steps'],
        "halted": result['halted'],
        "final_registers": [int(r) for r in result['registers']],
    }
    return {"content": [{"type": "text", "text": json.dumps(output, indent=2)}]}


@tool(
    "h4_geometry_info",
    "Get information about the H4 polytope geometry: vertices (600-cell), Coxeter chambers, dot products, golden ratio structure. Use aspect='all' for everything.",
    {"aspect": str},
)
async def h4_geometry_info(args):
    aspect = args["aspect"]
    vertices = generate_600_cell_vertices()
    roots = h4_simple_roots()
    info = {}

    if aspect in ("vertices", "all"):
        info["vertices"] = {
            "count": len(vertices),
            "all_on_unit_sphere": bool(np.allclose(np.linalg.norm(vertices, axis=1), 1.0)),
            "orbits": {
                "orbit_1": "8 vertices: permutations of (+-1, 0, 0, 0)",
                "orbit_2": "16 vertices: (+-1/2, +-1/2, +-1/2, +-1/2)",
                "orbit_3": "96 vertices: even perms of (0, +-1/2, +-phi/2, +-1/(2phi))",
            }
        }

    if aspect in ("chambers", "all"):
        info["coxeter_chambers"] = {
            "symmetry_group": "W(H4)",
            "group_order": 14400,
            "simple_roots": roots.tolist(),
        }

    if aspect in ("dot_products", "all"):
        dots = vertices @ vertices.T
        unique_dots = np.unique(np.round(dots[~np.eye(len(vertices), dtype=bool)].flatten(), 6))
        info["dot_products"] = {
            "unique_count": len(unique_dots),
            "values": sorted(unique_dots.tolist()),
            "phi_half_present": bool(any(abs(d - PHI / 2) < 0.01 for d in unique_dots)),
        }

    if aspect in ("golden_ratio", "all"):
        info["golden_ratio"] = {
            "phi": PHI,
            "phi_inverse": PHI_INV,
            "phi_squared_equals_phi_plus_1": abs(PHI**2 - (PHI + 1)) < 1e-12,
            "appearances": [
                "600-cell vertex coordinates contain phi/2 and 1/(2phi)",
                "Coxeter element eigenvalues involve cos(pi/5) = phi/2",
                "E8->H4 projection uses phi as scaling factor",
                "Fibonacci checkpoints grow with base phi",
            ]
        }

    return {"content": [{"type": "text", "text": json.dumps(info, indent=2)}]}


# ============================================================
# Create MCP Server with our tools
# ============================================================

h4_server = create_sdk_mcp_server(
    "h4-executor",
    tools=[h4_fibonacci, h4_geometry_info, h4_compile_and_run],
)


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """You are a hybrid AI system combining Claude's reasoning with the H4 Polytopic Attention transformer executor for exact computation.

You have access to MCP tools that run programs through a transformer with analytically constructed weights using 4D H4 (600-cell) attention heads. This is based on the paper "H4 Polytopic Attention" by Timothy McGirl.

Key concepts:
- The H4 polytope (600-cell) has 120 vertices on S^3, with 14,400 symmetries
- The golden ratio phi = (1+sqrt(5))/2 appears throughout
- 4D attention heads are quadratically more expressive than 2D heads
- Programs compile into transformer weights analytically (no training)

When asked for computation, use the H4 tools. Explain both results and geometry.

You are talking to Timothy McGirl, the author of this system."""


# ============================================================
# Demo Mode
# ============================================================

async def run_demo():
    print("=" * 60)
    print("H4 Polytopic Attention - Hybrid LLM Demo (Phase 3)")
    print("Claude (Max Plan OAuth) + H4 Transformer Executor")
    print("=" * 60)

    prompts = [
        "Compute Fibonacci(20) using the H4 transformer executor and explain what's happening geometrically.",
        "Show me the golden ratio structure in the H4 polytope.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Demo {i+1}: {prompt}")
        print("=" * 60)

        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"h4-executor": h4_server},
            max_turns=10,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)
                elif isinstance(message, ResultMessage):
                    print(f"\n{message.result}")

    print(f"\n{'=' * 60}")
    print("Demo complete.")


# ============================================================
# Interactive Mode
# ============================================================

async def run_interactive():
    print("=" * 60)
    print("H4 Polytopic Attention - Hybrid LLM (Phase 3)")
    print("Claude (Max Plan OAuth) + H4 Transformer Executor")
    print("=" * 60)
    print("\nType your message (Ctrl+C to exit):\n")

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"h4-executor": h4_server},
        max_turns=15,
    )

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input or user_input.lower() in ('quit', 'exit', 'q'):
                break

            await client.query(user_input)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(f"\nClaude: {block.text}")
                elif isinstance(message, ResultMessage):
                    print(f"\nClaude: {message.result}")
            print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        anyio.run(run_demo)
    else:
        anyio.run(run_interactive)
