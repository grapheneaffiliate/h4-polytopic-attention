"""
H4 Polytopic Attention — MCP Server (Phase 4)
===============================================

Exposes the H4 transformer executor as an MCP server for Claude Code.
Phase 4 adds E8 lattice-indexed RAM with STORE_MEM/LOAD_MEM opcodes.

Usage:
  Add to Claude Code settings.json:
  {
    "mcpServers": {
      "h4-executor": {
        "command": "py",
        "args": ["C:/Users/atchi/h4-polytopic-attention/python/h4_mcp_server.py"]
      }
    }
  }

Author: Timothy McGirl
"""

import sys
import os
import json

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import numpy as np

from weight_compiler import (
    Program, fibonacci_program, H4Executor,
    StateEncoder, CompiledTransformer,
    generate_600_cell_vertices, h4_simple_roots,
    PHI, PHI_INV,
)

server = Server("h4-executor")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="h4_fibonacci",
            description="Compute Fibonacci sequence using the H4 polytopic attention transformer executor. Runs through analytically constructed transformer weights with 4D H4 (600-cell) attention heads. Returns F(0) through F(n+1).",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of Fibonacci iterations (computes up to F(n+1)), max 30"}
                },
                "required": ["n"]
            },
        ),
        Tool(
            name="h4_compile_and_run",
            description=(
                "Compile and run a custom program on the H4 transformer executor. "
                "Phase 4 ISA: LOAD (immediate to register), ADD, SUB, MUL (register ops), "
                "STORE (copy), STORE_MEM (R[a] to E8 lattice at addr R[b]), "
                "LOAD_MEM (E8 lattice at addr R[a] to R[dest]), "
                "JMP, JNZ, HALT. 8 registers R0-R7. "
                "Memory ops use E8 Voronoi cell bucketing with 240 kissing-neighbor lookup."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "array",
                        "description": "List of instructions",
                        "items": {
                            "type": "object",
                            "properties": {
                                "opcode": {
                                    "type": "string",
                                    "enum": ["LOAD", "ADD", "SUB", "MUL", "STORE",
                                             "STORE_MEM", "LOAD_MEM",
                                             "JMP", "JNZ", "HALT"]
                                },
                                "a": {"type": "integer", "description": "First operand (register index or immediate)"},
                                "b": {"type": "integer", "description": "Second operand (register index)"},
                                "dest": {"type": "integer", "description": "Destination register"}
                            },
                            "required": ["opcode"]
                        }
                    },
                    "max_steps": {"type": "integer", "description": "Max execution steps (default 500)"}
                },
                "required": ["instructions"]
            },
        ),
        Tool(
            name="h4_geometry_info",
            description="Get H4 polytope geometry info: 600-cell vertices, Coxeter chambers, dot products, golden ratio structure. Aspects: vertices, chambers, dot_products, golden_ratio, all.",
            inputSchema={
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "enum": ["vertices", "chambers", "dot_products", "golden_ratio", "all"],
                        "description": "Which aspect to query"
                    }
                },
                "required": ["aspect"]
            },
        ),
        Tool(
            name="h4_benchmark",
            description="Benchmark the H4 attention system: encoding throughput and forward pass timing at different trace lengths.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n_steps": {"type": "integer", "description": "Number of steps (default 500)"}
                },
            },
        ),
        Tool(
            name="h4_lattice_memory",
            description=(
                "Phase 4: E8 lattice memory diagnostics. "
                "Run a program that exercises STORE_MEM/LOAD_MEM and return "
                "E8 Voronoi cell utilization stats: occupied cells, bucket distribution, "
                "primary hit rate, kissing number verification (240). "
                "Actions: 'benchmark' (store+load n entries), 'info' (E8 lattice constants)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["benchmark", "info"],
                        "description": "Action to perform"
                    },
                    "n_entries": {
                        "type": "integer",
                        "description": "Number of entries for benchmark (default 1000)"
                    }
                },
                "required": ["action"]
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "h4_fibonacci":
        return await _h4_fibonacci(arguments)
    elif name == "h4_compile_and_run":
        return await _h4_compile_and_run(arguments)
    elif name == "h4_geometry_info":
        return await _h4_geometry_info(arguments)
    elif name == "h4_benchmark":
        return await _h4_benchmark(arguments)
    elif name == "h4_lattice_memory":
        return await _h4_lattice_memory(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _h4_fibonacci(args):
    n = min(args["n"], 30)
    prog = fibonacci_program(n)
    executor = H4Executor(prog, d_model=32)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = executor.run(max_steps=500)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    seen = set()
    fib_seq = []
    for regs in executor.register_history:
        v = int(regs[1])
        if v not in seen:
            seen.add(v)
            fib_seq.append(v)

    expected = [0, 1]
    for _ in range(n):
        expected.append(expected[-1] + expected[-2])

    output = {
        "fibonacci_n": n,
        "result": int(result['registers'][1]),
        "expected": expected[n + 1],
        "correct": int(result['registers'][1]) == expected[n + 1],
        "steps": result['steps'],
        "sequence": fib_seq[:n + 2],
        "registers": [int(r) for r in result['registers'][:6]],
        "transformer": {"d_model": 32, "n_heads": 8, "n_layers": 4, "head_dim": "4D_H4"},
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _h4_compile_and_run(args):
    instructions = args["instructions"]
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
        "steps": result['steps'],
        "halted": result['halted'],
        "registers": [int(r) for r in result['registers']],
        "lattice_memory": result.get('lattice_memory', {}),
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _h4_geometry_info(args):
    aspect = args["aspect"]
    vertices = generate_600_cell_vertices()
    roots = h4_simple_roots()
    info = {}

    if aspect in ("vertices", "all"):
        info["vertices"] = {
            "count": len(vertices),
            "on_unit_sphere": bool(np.allclose(np.linalg.norm(vertices, axis=1), 1.0)),
            "orbits": ["8: perms of (+-1,0,0,0)", "16: (+-1/2)^4", "96: even perms of (0,+-1/2,+-phi/2,+-1/2phi)"],
        }
    if aspect in ("chambers", "all"):
        info["chambers"] = {
            "group": "W(H4)", "order": 14400,
            "simple_roots": [[round(x, 6) for x in r] for r in roots.tolist()],
        }
    if aspect in ("dot_products", "all"):
        dots = vertices @ vertices.T
        unique = np.unique(np.round(dots[~np.eye(len(vertices), dtype=bool)].flatten(), 6))
        info["dot_products"] = {
            "unique_count": len(unique),
            "values": [round(v, 6) for v in sorted(unique.tolist())],
            "has_phi_half": bool(any(abs(d - PHI/2) < 0.01 for d in unique)),
        }
    if aspect in ("golden_ratio", "all"):
        info["golden_ratio"] = {
            "phi": round(PHI, 15), "phi_inv": round(PHI_INV, 15),
            "phi^2 = phi+1": abs(PHI**2 - PHI - 1) < 1e-12,
            "roles": [
                "vertex coordinates",
                "Coxeter eigenvalues",
                "E8->H4 projection (cos(pi/5) = phi/2)",
                "Fibonacci checkpoint spacing",
                "Lattice memory Voronoi cell geometry",
            ],
        }

    return [TextContent(type="text", text=json.dumps(info, indent=2))]


async def _h4_benchmark(args):
    import time
    n = args.get("n_steps", 500)

    encoder = StateEncoder(32)
    transformer = CompiledTransformer(32)
    dummy = type('obj', (object,), {'opcode': 'ADD', 'operand_a': 0, 'operand_b': 1, 'dest': 2})()
    regs = np.zeros(8)

    start = time.time()
    states = [encoder.encode_state(i % 12, regs, dummy, i) for i in range(n)]
    enc_time = time.time() - start

    timings = {}
    for cp in [50, 100, 250, min(n, 500)]:
        if cp > len(states):
            break
        t0 = time.time()
        trace = np.array(states[:cp])
        _ = transformer.forward_layer(trace, transformer.layers[0])
        timings[f"{cp}_steps"] = f"{time.time()-t0:.3f}s"

    output = {
        "n_steps": n,
        "encoding": f"{enc_time:.3f}s ({n/enc_time:.0f} states/s)",
        "forward_pass": timings,
    }
    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _h4_lattice_memory(args):
    import time
    from h4_polytopic_attention import E8LatticeIndex

    action = args["action"]

    if action == "info":
        lattice = E8LatticeIndex()
        proj = lattice.projection_matrix
        output = {
            "e8_lattice": {
                "dimension": 8,
                "kissing_number": len(lattice.kissing_vectors),
                "kissing_vectors_verified": len(lattice.kissing_vectors) == 240,
                "voronoi_cell_structure": "D8 union (D8 + [1/2]^8)",
                "decoder": "O(1) closest-lattice-point",
            },
            "e8_to_h4_projection": {
                "shape": "4x8",
                "eigenvalues": {
                    "cos(pi/5)": round(float(proj[0, 0]), 10),
                    "phi/2": round(float(PHI / 2), 10),
                    "match": abs(float(proj[0, 0]) - PHI / 2) < 1e-10,
                    "cos(2pi/5)": round(float(proj[0, 2]), 10),
                    "1/(2phi)": round(float(PHI_INV / 2), 10),
                },
                "purpose": "Unifies 8D memory addressing with 4D H4 attention geometry",
            },
            "memory_opcodes": {
                "STORE_MEM": "R[a] -> E8 Voronoi cell at address R[b]",
                "LOAD_MEM": "E8 Voronoi cell at address R[a] -> R[dest]",
            },
            "max_cell_size": lattice.max_cell_size,
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    elif action == "benchmark":
        n = args.get("n_entries", 1000)
        lattice = E8LatticeIndex()

        # Store phase
        start = time.time()
        embeddings = []
        for i in range(n):
            emb = np.zeros(8)
            for j in range(4):
                theta = i * PHI_INV * (2 * np.pi) * (j + 1)
                emb[2*j] = np.cos(theta) * (1.0 + i * 0.001)
                emb[2*j + 1] = np.sin(theta) * (1.0 + i * 0.001)
            lattice.insert(emb, value=float(i), address=i)
            embeddings.append(emb)
        store_time = time.time() - start

        # Load phase (query same embeddings back)
        start = time.time()
        hits = 0
        for emb in embeddings:
            results = lattice.query_nearest(emb, k=1)
            if results:
                hits += 1
        load_time = time.time() - start

        stats = lattice.stats()
        output = {
            "benchmark": {
                "n_entries": n,
                "store_time_s": round(store_time, 4),
                "store_rate": f"{n/store_time:.0f} ops/s",
                "load_time_s": round(load_time, 4),
                "load_rate": f"{n/load_time:.0f} ops/s",
                "hit_rate": f"{hits}/{n} ({hits/n*100:.1f}%)",
            },
            "lattice_stats": {
                "total_entries": stats['total_entries'],
                "occupied_cells": stats['occupied_cells'],
                "utilization": f"{stats['utilization']:.1%}",
                "max_bucket_size": stats['max_bucket_size'],
                "avg_bucket_size": round(stats['avg_bucket_size'], 2),
                "primary_hit_rate": f"{stats['primary_hit_rate']:.1%}",
            },
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
