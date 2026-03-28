#!/usr/bin/env python3
"""
RunPod orchestrator for ARC-AGI brute-force solving.

Launches 8 GPU pods, distributes 400 tasks across them, monitors progress,
downloads results, and merges into the local tool registry.

Usage:
    # Set API key first:
    export RUNPOD_API_KEY=rp_xxxxx

    # Launch all pods:
    python -m olympus.arc.launch_pods launch

    # Check status:
    python -m olympus.arc.launch_pods status

    # Download results and stop pods:
    python -m olympus.arc.launch_pods collect

    # Emergency stop all pods:
    python -m olympus.arc.launch_pods stop
"""

import argparse
import json
import os
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────

NUM_PODS = 8
TASKS_PER_POD = 50
GPU_TYPE = "NVIDIA RTX 4080 SUPER"
CONTAINER_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04"
VOLUME_SIZE_GB = 50  # Persistent volume for model cache + results
CLOUD_TYPE = "COMMUNITY"  # or "SECURE" for dedicated

MODEL_REPO = "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-coder-32b-instruct-q4_k_m.gguf"
MODEL_PATH = f"/runpod-volume/models/{MODEL_FILE}"

TASKS_DIR = "/runpod-volume/arc_tasks"
RESULTS_DIR = "/runpod-volume/arc_results"
CODE_DIR = "/runpod-volume/arc_code"

LOCAL_TASKS = Path(__file__).parent.parent.parent / "data" / "arc1"
LOCAL_RESULTS = Path(__file__).parent.parent.parent / "data" / "arc_results"
POD_SOLVER_PATH = Path(__file__).parent / "pod_solver.py"

STATE_FILE = Path(__file__).parent.parent.parent / "data" / "pod_state.json"


def _load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"pods": [], "status": "idle"}


def _save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Pod launch ───────────────────────────────────────────────────

def _get_startup_command(batch_start, batch_size, attempts=50):
    """Generate the command that runs on pod startup."""
    return f"""
# Install deps
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 2>&1 | tail -3
pip install huggingface-hub arc-agi 2>&1 | tail -1

# Download pod_solver.py if not present
mkdir -p {CODE_DIR}
if [ ! -f "{CODE_DIR}/pod_solver.py" ]; then
    echo "ERROR: pod_solver.py not found at {CODE_DIR}/pod_solver.py"
    echo "Upload it with: runpodctl send olympus/arc/pod_solver.py"
    echo "Or copy via the web terminal."
    exit 1
fi

# Download model if not cached
mkdir -p /runpod-volume/models
if [ ! -f "{MODEL_PATH}" ]; then
    echo "Downloading {MODEL_FILE}..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='{MODEL_REPO}', filename='{MODEL_FILE}', local_dir='/runpod-volume/models/')
"
fi

# Download tasks if not cached
if [ ! -d "{TASKS_DIR}" ] || [ $(ls {TASKS_DIR}/*.json 2>/dev/null | wc -l) -lt 100 ]; then
    echo "Downloading ARC tasks..."
    PYTHONIOENCODING=utf-8 python -c "
import arc_agi
t = arc_agi.ARC1Training()
t.download('{TASKS_DIR}')
" 2>&1 || echo "arc-agi download failed, tasks should be pre-uploaded"
fi

# Run solver
cd {CODE_DIR}
python pod_solver.py \
    --tasks-dir {TASKS_DIR} \
    --output-dir {RESULTS_DIR} \
    --model-path {MODEL_PATH} \
    --batch-start {batch_start} \
    --batch-size {batch_size} \
    --attempts {attempts} \
    --n-gpu-layers -1

echo "DONE: batch {batch_start}"
"""


def launch(attempts=50):
    """Launch NUM_PODS pods, each solving a batch of tasks."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        return

    runpod.api_key = api_key

    # Check task files exist
    task_files = sorted(LOCAL_TASKS.glob("*.json"))
    if not task_files:
        print(f"ERROR: No task JSONs found in {LOCAL_TASKS}")
        return
    print(f"Found {len(task_files)} task files")

    state = {"pods": [], "status": "launching", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for i in range(NUM_PODS):
        batch_start = i * TASKS_PER_POD
        batch_size = min(TASKS_PER_POD, len(task_files) - batch_start)
        if batch_size <= 0:
            break

        name = f"arc-solver-{i}"
        print(f"Launching pod {name}: tasks {batch_start}-{batch_start + batch_size - 1}")

        try:
            pod = runpod.create_pod(
                name=name,
                image_name=CONTAINER_IMAGE,
                gpu_type_id="NVIDIA RTX 4080 SUPER",
                volume_in_gb=VOLUME_SIZE_GB,
                container_disk_in_gb=20,
                volume_mount_path="/runpod-volume",
                cloud_type=CLOUD_TYPE,
                docker_args=f"bash -c '{_get_startup_command(batch_start, batch_size, attempts)}'",
            )

            state["pods"].append({
                "id": pod["id"],
                "name": name,
                "batch_start": batch_start,
                "batch_size": batch_size,
                "status": "running",
            })
            print(f"  Pod {pod['id']} launched")

        except Exception as e:
            print(f"  ERROR launching pod {name}: {e}")
            state["pods"].append({
                "name": name,
                "batch_start": batch_start,
                "batch_size": batch_size,
                "status": "failed",
                "error": str(e),
            })

    state["status"] = "running"
    _save_state(state)
    print(f"\n{len(state['pods'])} pods launched. State saved to {STATE_FILE}")
    print("Run 'python -m olympus.arc.launch_pods status' to check progress")
    print("Run 'python -m olympus.arc.launch_pods collect' when done")


# ── Upload tasks and solver code to volume ───────────────────────

def upload():
    """Upload task files and solver code to the first pod's volume.

    This should be run BEFORE launch, or the first pod should handle it.
    Actually, since all pods share the same volume, uploading once suffices.

    For simplicity, we include the upload in the startup command or
    use runpod's file upload API.
    """
    print("Note: Tasks and solver code should be uploaded to /runpod-volume/")
    print(f"  Tasks: {LOCAL_TASKS} -> {TASKS_DIR}")
    print(f"  Solver: {POD_SOLVER_PATH} -> {CODE_DIR}/pod_solver.py")
    print()
    print("Use `runpodctl send` or SCP to upload files.")
    print("Or: include task JSONs in the Docker image / download from HuggingFace.")
    print()
    print("Quick approach: upload a tar.gz to the volume via the RunPod web console,")
    print("or use this command on any running pod:")
    print(f"  pip install arc-agi && python -c \"import arc_agi; arc_agi.ARC1Training().download('{TASKS_DIR}')\"")


# ── Status check ─────────────────────────────────────────────────

def status():
    """Check status of all pods."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY")
        return

    runpod.api_key = api_key
    state = _load_state()

    if not state["pods"]:
        print("No pods launched. Run 'launch' first.")
        return

    print(f"Status: {state['status']} (started: {state.get('started_at', '?')})")
    print(f"{'Pod ID':<25} {'Name':<20} {'Batch':<12} {'Status'}")
    print("-" * 70)

    for pod_info in state["pods"]:
        pod_id = pod_info.get("id", "N/A")
        try:
            pod = runpod.get_pod(pod_id)
            runtime_status = pod.get("runtime", {}).get("status", "unknown")
        except Exception:
            runtime_status = "error"

        batch = f"{pod_info['batch_start']}-{pod_info['batch_start'] + pod_info['batch_size'] - 1}"
        print(f"{pod_id:<25} {pod_info['name']:<20} {batch:<12} {runtime_status}")


# ── Collect results ──────────────────────────────────────────────

def collect():
    """Download results from pods, merge into local tool registry, stop pods."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY")
        return

    runpod.api_key = api_key
    state = _load_state()

    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)

    print("Collecting results...")
    print("NOTE: Use `runpodctl receive` or SCP to download result files from pods.")
    print(f"  Remote: {RESULTS_DIR}/batch_*.json")
    print(f"  Local:  {LOCAL_RESULTS}/")
    print()

    # Merge any downloaded results
    all_results = []
    for result_file in sorted(LOCAL_RESULTS.glob("batch_*.json")):
        with open(result_file) as f:
            batch_results = json.load(f)
        all_results.extend(batch_results)
        solved = sum(1 for r in batch_results if r["solved"])
        print(f"  {result_file.name}: {solved}/{len(batch_results)} solved")

    if all_results:
        total_solved = sum(1 for r in all_results if r["solved"])
        print(f"\nTotal: {total_solved}/{len(all_results)} solved")

        # Save merged results
        merged_path = LOCAL_RESULTS / "all_results.json"
        with open(merged_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Merged results: {merged_path}")

        # List solved tasks
        print("\nSolved tasks:")
        for r in all_results:
            if r["solved"]:
                desc = r.get("description", "")[:50]
                print(f"  {r['task_id']}: {desc} (attempt {r['attempts']}, temp={r.get('temperature')})")


def stop():
    """Stop all pods."""
    import runpod

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY")
        return

    runpod.api_key = api_key
    state = _load_state()

    for pod_info in state["pods"]:
        pod_id = pod_info.get("id")
        if pod_id:
            try:
                runpod.stop_pod(pod_id)
                print(f"Stopped {pod_info['name']} ({pod_id})")
                pod_info["status"] = "stopped"
            except Exception as e:
                print(f"Error stopping {pod_info['name']}: {e}")

    state["status"] = "stopped"
    _save_state(state)
    print("All pods stopped.")


# ── Merge results into tool registry ─────────────────────────────

def merge_to_registry():
    """Merge downloaded pod results into the local tool registry."""
    results_file = LOCAL_RESULTS / "all_results.json"
    if not results_file.exists():
        print(f"No results file at {results_file}. Run 'collect' first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    registry_path = Path(__file__).parent.parent / "tool_registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"tools": [], "failures_logged": []}

    existing_names = {t["name"] for t in registry["tools"]}
    added = 0

    for r in results:
        if not r["solved"]:
            continue

        tool_name = f"arc_{r['task_id']}_llm"
        if tool_name in existing_names:
            continue

        # Save Python code
        code_dir = Path(__file__).parent.parent / "wasm_tools" / "arc" / "solved"
        code_dir.mkdir(parents=True, exist_ok=True)

        py_path = code_dir / f"{r['task_id']}_llm.py"
        py_path.write_text(r["python_code"])

        if r.get("c_code"):
            c_path = code_dir / f"{r['task_id']}_llm.c"
            c_path.write_text(r["c_code"])

        registry["tools"].append({
            "name": tool_name,
            "c_source": str(code_dir / f"{r['task_id']}_llm.c") if r.get("c_code") else None,
            "python_source": str(py_path),
            "query_patterns": [f"arc.*{r['task_id']}"],
            "description": f"ARC {r['task_id']}: {r.get('description', 'LLM-generated')}",
            "compiled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "llm_32b",
            "attempts_needed": r["attempts"],
            "uses": 0,
        })
        existing_names.add(tool_name)
        added += 1

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Added {added} new tools to registry ({len(registry['tools'])} total)")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-AGI RunPod Orchestrator")
    parser.add_argument("command", choices=["launch", "status", "collect", "stop",
                                            "upload", "merge"],
                        help="Command to run")
    parser.add_argument("--attempts", type=int, default=50, help="Attempts per task")
    args = parser.parse_args()

    if args.command == "launch":
        launch(attempts=args.attempts)
    elif args.command == "status":
        status()
    elif args.command == "collect":
        collect()
    elif args.command == "stop":
        stop()
    elif args.command == "upload":
        upload()
    elif args.command == "merge":
        merge_to_registry()
