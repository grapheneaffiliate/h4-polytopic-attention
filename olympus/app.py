"""
Lattice — H4 Polytopic Attention Demo App

The interactive frontend to Project Olympus.
Runs the actual router, compiled arithmetic, specialists, and E8 retrieval.

Run:  python olympus/app.py
Open: http://localhost:7860

Works immediately with compiled arithmetic + router.
Specialists activate when checkpoints are downloaded.
"""

import gradio as gr
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

from router import OlympusRouter
from compiled_arithmetic import CompiledStackExecutor
from gguf_inference import GGUFEngine
try:
    from tvm_engine import TVMEngine
    tvm_engine = TVMEngine()
except ImportError:
    tvm_engine = None
from code_verifier import verify_code_response, build_fix_prompt, check_output_properties

# Initialize components
router = OlympusRouter()
compute_engine = CompiledStackExecutor()
gguf_engine = GGUFEngine()

# Track which specialists are available
SPECIALIST_STATUS = {}
gguf_available = gguf_engine.available_specialists()
for name in ['general', 'code', 'math', 'qa']:
    if name in gguf_available:
        SPECIALIST_STATUS[name] = 'ready (GGUF, fast)'
    else:
        adapter_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', f'olympus_{name}', 'final', 'adapter_model.safetensors')
        if name == 'general':
            SPECIALIST_STATUS[name] = 'ready (base SmolLM3)'
        elif os.path.exists(adapter_path):
            SPECIALIST_STATUS[name] = 'ready (LoRA adapter, convert to GGUF for speed)'
        else:
            SPECIALIST_STATUS[name] = 'not loaded (checkpoint missing)'


def process_query(query, history):
    """Full Olympus pipeline."""
    if not query.strip():
        return history, "*Waiting for query...*"

    t_start = time.time()
    pipeline_steps = []

    # Step 0a: transformer-vm (primary compute engine, 30K tok/s, any C program)
    # Computes exact answer. For pure math queries, returns immediately.
    # For "write code" queries, stores the result as ground truth for verification.
    tvm_result = None
    if tvm_engine and tvm_engine.available and tvm_engine.can_handle(query):
        tvm_result = tvm_engine.compute(query)

    # If query is purely computational (no "write", "code", "function", "implement"), return exact answer
    q_lower = query.lower()
    code_request = any(kw in q_lower for kw in ['write', 'code', 'function', 'implement', 'program', 'script', 'class'])
    if tvm_result and not code_request:
        t_total = (time.time() - t_start) * 1000
        pipeline_info = (
            f"### Pipeline\n"
            f"**Method:** TRANSFORMER-VM ({tvm_result.get('engine', 'graph-evaluator')})\n\n"
            f"**Expression:** `{tvm_result['expression']}`\n\n"
            f"**Result:** `{tvm_result['result']}`\n\n"
            f"**Time:** {tvm_result['time_ms']:.1f}ms (total {t_total:.1f}ms)\n\n"
            f"*Exact. C program compiled to WASM, executed through analytical transformer.*"
        )
        answer = f"**{tvm_result['expression']} = {tvm_result['result']}**\n\n*Computed via transformer-vm (exact, analytical transformer weights)*"
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
        return history, pipeline_info

    # Step 0b: Compiled arithmetic (zero-dependency fallback)
    if compute_engine.can_handle(query):
        computation = compute_engine.extract_and_compute(query)
        if computation:
            expr, result, trace = computation
            t_total = (time.time() - t_start) * 1000

            trace_str = "\n".join(f"  {t}" for t in trace)
            pipeline_info = (
                f"### Pipeline\n"
                f"**Method:** Compiled Arithmetic (binary circuit, fallback)\n\n"
                f"**Expression:** `{expr}`\n\n"
                f"**Result:** `{result}`\n\n"
                f"**Time:** {t_total:.1f}ms\n\n"
                f"**Execution trace:**\n```\n{trace_str}\n```\n\n"
                f"*Exact. Zero error. No LLM invoked. (Install transformer-vm for 150x speedup)*"
            )

            answer = f"**{expr} = {result}**\n\n*Computed via compiled binary circuit (24-bit, exact)*"
            history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
            return history, pipeline_info

    # If transformer-vm computed a ground truth for a code request, note it
    if tvm_result and code_request:
        pipeline_steps.append(
            f"**Ground truth (transformer-vm):** `{tvm_result['result']}` ({tvm_result['time_ms']:.0f}ms)"
        )

    # Step 1: Route
    t_route = time.time()
    route_result = router.route(query)
    specialist = route_result['specialist']
    confidence = route_result.get('confidence', 0)
    chamber = route_result.get('chamber', -1)
    t_route_ms = (time.time() - t_route) * 1000

    pipeline_steps.append(
        f"**Router:** {specialist} specialist\n"
        f"  Chamber {chamber}, {confidence:.0%} confidence, {t_route_ms:.1f}ms"
    )

    # Step 2: Retrieve context (for QA queries)
    context = ""
    if specialist in ('qa', 'general') and 'who' in query.lower() or 'what' in query.lower() or 'when' in query.lower() or 'where' in query.lower():
        try:
            from olympus.knowledge_index import KnowledgeIndex
            index = KnowledgeIndex()
            if index.load():
                t_ret = time.time()
                results = index.query(query, k=3)
                t_ret_ms = (time.time() - t_ret) * 1000
                if results:
                    context = "\n\n".join(r['text'][:300] for r in results)
                    pipeline_steps.append(f"**E8 Retrieval:** {len(results)} passages ({t_ret_ms:.1f}ms)")
        except Exception:
            pass

    # Step 3: Generate response
    t_gen = time.time()
    specialist_status = SPECIALIST_STATUS.get(specialist, 'unknown')

    if gguf_engine.is_available(specialist):
        # Fast path: GGUF inference (~4-8s on CPU)
        try:
            answer = gguf_engine.generate(query, specialist=specialist, context=context)
            t_gen_ms = (time.time() - t_gen) * 1000
            pipeline_steps.append(f"**Generation:** {specialist} specialist via GGUF ({t_gen_ms:.0f}ms)")

            # Sprint contract: verify code specialist output
            if specialist == 'code':
                verification = verify_code_response(answer)
                if verification['has_code']:
                    exec_result = verification['execution']
                    if not verification['verified']:
                        # Code crashed — try to fix
                        error_msg = exec_result['stderr'][:200]
                        pipeline_steps.append(f"**Verify:** code CRASHED\n  Error: `{error_msg}`")
                        fix_prompt = build_fix_prompt(query, verification['code'], exec_result['stderr'])
                        try:
                            fix_answer = gguf_engine.generate(fix_prompt, specialist='code')
                            fix_check = verify_code_response(fix_answer)
                            if fix_check['has_code'] and fix_check['verified']:
                                answer = fix_answer
                                exec_result = fix_check['execution']
                                pipeline_steps.append(f"**Fix:** second attempt runs OK")
                            else:
                                answer += f"\n\n---\n*Code crashed. Error: {error_msg}*"
                                pipeline_steps.append(f"**Fix:** second attempt also failed")
                        except Exception:
                            answer += f"\n\n---\n*Code crashed. Error: {error_msg}*"

                    # Property check: verify output satisfies mathematical invariants
                    if exec_result.get('success') and exec_result.get('stdout'):
                        props = check_output_properties(query, exec_result['stdout'])
                        if props['checked']:
                            if props['passed']:
                                pipeline_steps.append(f"**Properties:** all checks passed\n  Output: `{exec_result['stdout'][:200]}`")
                            else:
                                failure_str = "; ".join(props['failures'])
                                pipeline_steps.append(f"**Properties:** FAILED\n  `{failure_str}`")
                                # Try fix with property violation as error
                                fix_prompt = build_fix_prompt(query, verification['code'], f"Output property violation: {failure_str}")
                                try:
                                    fix_answer = gguf_engine.generate(fix_prompt, specialist='code')
                                    fix_check = verify_code_response(fix_answer)
                                    if fix_check['has_code'] and fix_check['verified']:
                                        fix_props = check_output_properties(query, fix_check['execution']['stdout'])
                                        if fix_props['passed']:
                                            answer = fix_answer
                                            pipeline_steps.append(f"**Fix:** corrected! Output: `{fix_check['execution']['stdout'][:200]}`")
                                        else:
                                            answer += f"\n\n---\n*Property check failed: {failure_str}*"
                                            pipeline_steps.append(f"**Fix:** second attempt still violates properties")
                                    else:
                                        answer += f"\n\n---\n*Property check failed: {failure_str}*"
                                except Exception:
                                    answer += f"\n\n---\n*Property check failed: {failure_str}*"
                        elif exec_result.get('stdout'):
                            pipeline_steps.append(f"**Verify:** code ran OK\n  Output: `{exec_result['stdout'][:200]}`")

        except Exception as e:
            answer = f"*[GGUF generation error: {str(e)[:100]}]*"
            pipeline_steps.append(f"**Generation:** GGUF error")
    elif 'not loaded' in specialist_status:
        answer = (
            f"*[{specialist} specialist: checkpoint not downloaded yet]*\n\n"
            f"The router correctly identified this as a **{specialist}** query.\n"
            f"Full generation requires downloading the LoRA checkpoint from RunPod.\n\n"
            f"**What would happen:** SmolLM3-3B + {specialist} LoRA adapter generates a response "
            f"using the query{' and retrieved context' if context else ''} as input."
        )
        pipeline_steps.append(f"**Generation:** {specialist} specialist (not loaded)")
    else:
        # Slow fallback: HuggingFace transformers on CPU
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "HuggingFaceTB/SmolLM3-3B"
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            if specialist == 'general':
                import torch
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.float32, device_map="cpu")

                prompt = f"{query}" if not context else f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=200,
                        temperature=0.7, do_sample=True,
                        pad_token_id=tokenizer.eos_token_id)

                answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                t_gen_ms = (time.time() - t_gen) * 1000
                pipeline_steps.append(f"**Generation:** {specialist} specialist via HF ({t_gen_ms:.0f}ms, convert to GGUF for 3-5x speedup)")
                del model
            else:
                answer = f"*[{specialist} specialist: convert to GGUF for fast CPU inference]*\n\nRun: `python olympus/convert_gguf.py --specialist {specialist}`"
                pipeline_steps.append(f"**Generation:** skipped (run convert_gguf.py first)")

        except Exception as e:
            answer = f"*[Generation error: {str(e)[:100]}]*"
            pipeline_steps.append(f"**Generation:** error")

    t_total = (time.time() - t_start) * 1000
    pipeline_steps.append(f"**Total:** {t_total:.0f}ms")

    pipeline_info = "### Pipeline\n" + "\n\n".join(pipeline_steps)
    history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
    return history, pipeline_info


# Gradio UI
DESCRIPTION = """
# ◇ Lattice

**H4 Polytopic Attention · E8 Retrieval · Geometric Routing**

A multi-specialist AI system running locally. No cloud. No API. No monthly cost.

| Component | Status |
|-----------|--------|
| Router | 100% accuracy, <1ms |
| transformer-vm | Any C program, exact, 30K tok/s |
| Compiled Arithmetic | Fallback, 30/30 exact, zero deps |
| E8 Retrieval | R@5=100%, 20ms |
| MiniLM Reranking | R@1=98.5% |

**Try:** `What is 15 * 23?` · `(3+5)*(7-2)` · `99*99+1` · `500*500`
"""

with gr.Blocks(
    title="Lattice — H4 Polytopic Attention",
) as app:

    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Lattice",
                height=450,
                show_label=False,
            )

            with gr.Row():
                query_input = gr.Textbox(
                    placeholder="Ask anything...",
                    show_label=False,
                    scale=5,
                    container=False,
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")

        with gr.Column(scale=1):
            pipeline_display = gr.Markdown(
                value="*Pipeline details appear here after each query.*",
                label="Pipeline",
            )

            with gr.Accordion("System Status", open=False):
                status_lines = []
                for name, status in SPECIALIST_STATUS.items():
                    icon = "✓" if 'ready' in status else "○"
                    status_lines.append(f"- {icon} **{name}:** {status}")
                gr.Markdown("\n".join(status_lines))

    submit_btn.click(
        process_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, pipeline_display],
    ).then(lambda: "", outputs=[query_input])

    query_input.submit(
        process_query,
        inputs=[query_input, chatbot],
        outputs=[chatbot, pipeline_display],
    ).then(lambda: "", outputs=[query_input])

    gr.Examples(
        examples=[
            "What is 15 * 23?",
            "99 * 99 + 1",
            "fib 10",
            "prime 97",
            "gcd 24 36",
            "collatz 27",
            "Write a binary search in Python",
            "When was the Eiffel Tower built?",
        ],
        inputs=query_input,
    )


if __name__ == '__main__':
    print("Starting Lattice app...")
    print("Compiled arithmetic: 30/30 exact")
    print("Router: 100% on test set")
    print(f"Specialists: {SPECIALIST_STATUS}")
    gguf_status = gguf_engine.status()
    gguf_ready = [k for k, v in gguf_status.items() if v != "not converted"]
    print(f"GGUF models: {gguf_ready if gguf_ready else 'none (run convert_gguf.py after checkpoint download)'}")
    print()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
