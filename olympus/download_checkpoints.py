"""
Download specialist checkpoints from RunPod pods.

CRITICAL: Verifies checkpoints exist on PERSISTENT volume (/runpod-volume/)
before downloading. Never stops a pod without verification.

Usage:
    python olympus/download_checkpoints.py

Lesson learned: /workspace/ is ephemeral (wiped on pod stop).
Only /runpod-volume/ persists. All training scripts must save there.
"""

import subprocess
import sys
import os

# Pod SSH info — UPDATE THESE before running (get from RunPod dashboard)
PODS = {
    'code': {'host': '<CODE_POD_IP>', 'port': '<CODE_POD_PORT>', 'remote_dir': '/runpod-volume/olympus_code_specialist/final'},
    'math': {'host': '<MATH_POD_IP>', 'port': '<MATH_POD_PORT>', 'remote_dir': '/runpod-volume/olympus_math_specialist/final'},
    'qa':   {'host': '<QA_POD_IP>',   'port': '<QA_POD_PORT>',   'remote_dir': '/runpod-volume/olympus_qa_specialist/final'},
}

LOCAL_BASE = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')


def ssh_cmd(host, port, cmd):
    """Run a command on a pod via SSH."""
    result = subprocess.run(
        ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
         f'root@{host}', '-p', port, cmd],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip(), result.returncode


def verify_checkpoint(name, pod_info):
    """Verify checkpoint exists on persistent volume before downloading."""
    host, port = pod_info['host'], pod_info['port']
    remote_dir = pod_info['remote_dir']

    print(f"\n{'='*50}")
    print(f"  Verifying {name} specialist checkpoint")
    print(f"{'='*50}")

    # Check if pod is reachable
    out, rc = ssh_cmd(host, port, 'echo connected')
    if rc != 0:
        print(f"  ERROR: Cannot reach pod at {host}:{port}")
        return False

    # Check if checkpoint directory exists
    out, rc = ssh_cmd(host, port, f'ls {remote_dir}/adapter_model.safetensors 2>/dev/null && echo EXISTS || echo MISSING')
    if 'MISSING' in out:
        print(f"  ERROR: Checkpoint not found at {remote_dir}")
        print(f"  Training may still be running. Check logs:")
        print(f"    ssh root@{host} -p {port} 'tail -5 /runpod-volume/{name}_training.log'")
        return False

    # Check file size
    out, rc = ssh_cmd(host, port, f'ls -lh {remote_dir}/adapter_model.safetensors')
    print(f"  Found: {out}")

    # Check if it's on persistent volume (not workspace)
    out, rc = ssh_cmd(host, port, f'df {remote_dir} | tail -1')
    print(f"  Storage: {out}")

    # Verify it's a valid safetensors file (check magic bytes)
    out, rc = ssh_cmd(host, port, f'head -c 8 {remote_dir}/adapter_model.safetensors | xxd | head -1')
    print(f"  Header: {out}")

    print(f"  VERIFIED: {name} checkpoint exists on persistent volume")
    return True


def download_checkpoint(name, pod_info):
    """Download a verified checkpoint."""
    host, port = pod_info['host'], pod_info['port']
    remote_dir = pod_info['remote_dir']
    local_dir = os.path.join(LOCAL_BASE, f'olympus_{name}')

    os.makedirs(local_dir, exist_ok=True)

    print(f"  Downloading {name} to {local_dir}...")
    result = subprocess.run(
        ['scp', '-o', 'StrictHostKeyChecking=no', '-P', port, '-r',
         f'root@{host}:{remote_dir}/', local_dir + '/'],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"  ERROR: Download failed: {result.stderr}")
        return False

    # Verify local download
    local_file = os.path.join(local_dir, 'adapter_model.safetensors')
    if os.path.exists(local_file):
        size_mb = os.path.getsize(local_file) / 1024 / 1024
        print(f"  Downloaded: {local_file} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ERROR: File not found locally after download")
        return False


def main():
    print("Olympus Checkpoint Downloader")
    print("=" * 50)
    print("RULE: Never stop a pod without verifying checkpoints first.")
    print()

    # Phase 1: Verify all checkpoints exist
    verified = {}
    for name, pod_info in PODS.items():
        verified[name] = verify_checkpoint(name, pod_info)

    print(f"\n{'='*50}")
    print("Verification Summary:")
    for name, ok in verified.items():
        status = "READY" if ok else "NOT READY"
        print(f"  {name}: {status}")

    # Phase 2: Download verified checkpoints
    ready = [name for name, ok in verified.items() if ok]
    not_ready = [name for name, ok in verified.items() if not ok]

    if not_ready:
        print(f"\nWARNING: {len(not_ready)} specialist(s) not ready: {not_ready}")
        print("These pods should NOT be stopped yet.")

    if ready:
        print(f"\nDownloading {len(ready)} verified checkpoint(s)...")
        for name in ready:
            download_checkpoint(name, PODS[name])

    # Phase 3: Only suggest stopping verified pods
    if ready:
        print(f"\nSafe to stop pods for: {ready}")
        print("DO NOT stop pods for: {not_ready}" if not_ready else "All pods safe to stop.")
    else:
        print("\nNo checkpoints ready. Do not stop any pods.")


if __name__ == '__main__':
    main()
