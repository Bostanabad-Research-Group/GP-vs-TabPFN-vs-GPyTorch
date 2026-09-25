"""Run one experiment in its own process and keep going if it fails.

A configuration that raises, or that runs longer than the timeout, is recorded
under ``results/failures.jsonl`` and in ``FAILED.txt`` beside that problem.
The caller then starts the next configuration.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def result_already_saved(
    save_dir: Path,
    train_size: int | None,
    noise: float | None,
    dims: int | None = None,
) -> bool:
    """True when this folder already has a finished metrics file for this setup.

    Ackley 20D and 40D share one folder. A file counts for this job only when
    its dimension token matches ``dims``. Files with no dimension token, such
    as wing, still match on training size and noise alone.
    """
    if not save_dir.is_dir():
        return False
    size_re = None
    if train_size is not None:
        size_re = re.compile(rf"(?<![0-9]){int(train_size)}d(?:n|_)", re.IGNORECASE)
    dim_re = None
    any_dim_re = None
    if dims is not None:
        dim_re = re.compile(rf"(?<![0-9]){int(dims)}(?:dx|xdim)", re.IGNORECASE)
        any_dim_re = re.compile(r"(?<![0-9])\d+(?:dx|xdim)", re.IGNORECASE)
    noise_token = None if noise is None else f"noisetest{noise}_".lower()
    for path in save_dir.glob("*.json"):
        name = path.name.lower()
        if "trainer" in name or name.startswith("failed"):
            continue
        if size_re is not None and size_re.search(name) is None:
            continue
        if noise_token is not None and noise_token not in name:
            continue
        if dim_re is not None and any_dim_re.search(name) and dim_re.search(name) is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _has_metrics(data):
            return True
    return False


def _has_metrics(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    for key in ("gp_data", "tabpfn_data", "BO_metrics"):
        block = data.get(key)
        if isinstance(block, dict):
            metrics = block.get("metrics")
            if isinstance(metrics, list) and metrics:
                return True
        if key == "BO_metrics" and isinstance(block, list) and block:
            return True
    summary = data.get("BO_summary")
    return isinstance(summary, dict) and bool(summary)


def run_isolated(
    script: Path,
    job: dict,
    *,
    timeout_s: float,
    failures_path: Path,
) -> str:
    """Run ``script --job-file`` and return ok, skipped, error, or timeout."""
    label = str(job["label"])
    save_dir = Path(job["save_path"])
    if job.get("skip_existing", True) and result_already_saved(
        save_dir, job.get("train_size"), job.get("noise"), job.get("dims")
    ):
        print(f"SKIP existing {label}", flush=True)
        return "skipped"

    job_dir = failures_path.parent / "_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)[:180]
    job_path = job_dir / f"{safe}.json"
    trace_path = job_dir / f"{safe}.trace.txt"
    log_path = job_dir / f"{safe}.log"
    payload = dict(job)
    payload["trace_path"] = str(trace_path)
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    print(f"\n=== {label}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(script), "--job-file", str(job_path)],
            cwd=str(script.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            message = f"Stopped after {timeout_s / 3600:.1f} hours."
            _record(failures_path, payload, "timeout", message)
            print(f"TIMEOUT {label}: {message}", flush=True)
            return "timeout"
    if code == 0:
        return "ok"
    if trace_path.exists():
        message = trace_path.read_text(encoding="utf-8")[-4000:]
    elif log_path.exists():
        message = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    else:
        message = f"Process exited with code {code}."
    _record(failures_path, payload, "error", message)
    print(f"FAILED {label}", flush=True)
    return "error"


def _record(failures_path: Path, job: dict, status: str, message: str) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "label": job.get("label"),
        "status": status,
        "problem": job.get("problem"),
        "model": job.get("model"),
        "save_path": job.get("save_path"),
        "message": message,
    }
    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    save_dir = Path(job["save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)
    note = save_dir / "FAILED.txt"
    with note.open("a", encoding="utf-8") as handle:
        handle.write(f"{status}: {job.get('label')}\n{message}\n\n")


def write_failure_report(failures_path: Path) -> None:
    """Rewrite ``failures.md`` next to the jsonl log."""
    if not failures_path.exists():
        return
    rows = []
    for line in failures_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    lines = [
        "# Experiments that did not finish",
        "",
        "These configurations were skipped after an error or a timeout. The rest of the suite continued.",
        "",
    ]
    if not rows:
        lines.append("None.")
    for row in rows:
        lines.append(f"- **{row.get('status')}** `{row.get('label')}`")
        message = str(row.get("message") or "").strip().splitlines()
        if message:
            lines.append(f"  {message[-1][:300]}")
    report = failures_path.with_suffix(".md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recorded {len(rows)} failed experiments in {report}", flush=True)


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        subprocess.run(["kill", "-9", str(pid)], capture_output=True, check=False)
