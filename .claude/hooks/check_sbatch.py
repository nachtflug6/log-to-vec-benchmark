#!/usr/bin/env python3
"""PreToolUse hook: resource check before any sbatch submission.

Parses the SLURM script referenced in the bash command and injects a
resource summary into Claude's context. Warns when an expensive GPU type
(A40, A100) is requested for a task where T4 would suffice.
"""
import json, re, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

GPU_TIERS = {
    "T4":      ("cheap",          "✓ good choice"),
    "V100":    ("medium",         "~ ok for memory-heavy inference"),
    "A40":     ("expensive",      "⚠ consider T4 for inference-only"),
    "A100":    ("expensive",      "⚠ consider T4 for inference-only"),
    "A100fat": ("very expensive", "✗ only if model needs >40 GB VRAM"),
}


def parse_slurm(path: Path) -> dict:
    text = path.read_text()
    def get(pattern):
        m = re.search(pattern, text, re.MULTILINE)
        return m.group(1) if m else None

    gpu_raw  = get(r"^#SBATCH\s+--gpus-per-node=(\S+)")
    gpu_type = gpu_raw.split(":")[0] if gpu_raw else None
    n_gpu    = gpu_raw.split(":")[-1] if gpu_raw and ":" in gpu_raw else "1"

    return {
        "job_name":  get(r"^#SBATCH\s+--job-name=(\S+)") or path.name,
        "gpu_raw":   gpu_raw or "none",
        "gpu_type":  gpu_type,
        "n_gpu":     n_gpu,
        "time":      get(r"^#SBATCH\s+--time=(\S+)") or "?",
        "cpus":      get(r"^#SBATCH\s+--cpus-per-task=(\d+)") or "?",
        "nodes":     get(r"^#SBATCH\s+--nodes=(\d+)") or "1",
        "account":   get(r"^#SBATCH\s+--account=(\S+)") or "?",
    }


def main():
    data = json.load(sys.stdin)
    cmd  = data.get("tool_input", {}).get("command", "")

    if "sbatch" not in cmd:
        sys.exit(0)

    # Extract the .slurm file path from the command
    m = re.search(r"sbatch\s+(?:[A-Z_]+=\S+\s+)*(\S+\.slurm)", cmd)
    if not m:
        sys.exit(0)

    script_arg = m.group(1)
    script     = Path(script_arg)
    if not script.is_absolute():
        script = REPO_ROOT / script_arg
    if not script.exists():
        # last-ditch: try relative to cwd
        script = Path.cwd() / script_arg
    if not script.exists():
        out = {"systemMessage": f"sbatch resource check: script not found ({script_arg})"}
        print(json.dumps(out))
        sys.exit(0)

    r = parse_slurm(script)
    tier, advice = GPU_TIERS.get(r["gpu_type"], ("unknown", "? unrecognised GPU type"))

    lines = [
        f"SLURM resource check — {r['job_name']}",
        f"  GPU:     {r['gpu_raw']}  [{tier}]  {advice}",
        f"  CPUs:    {r['cpus']}  |  Nodes: {r['nodes']}",
        f"  Time:    {r['time']}",
        f"  Account: {r['account']}",
    ]

    if tier in ("expensive", "very expensive"):
        lines.append("")
        vram_floor = {"A40": "16", "A100": "16", "A100fat": "40"}.get(r["gpu_type"], "?")
        lines.append(
            f"  Recommendation: switch to T4:1 unless the model needs >{vram_floor} GB VRAM."
        )

    print(json.dumps({"systemMessage": "\n".join(lines)}))


if __name__ == "__main__":
    main()
