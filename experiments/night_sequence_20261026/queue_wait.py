"""Process identities and one-shot execution for the fixed October 26 sequence."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import select
import subprocess
import time


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_start(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def exit_descriptor(pid: int) -> int:
    # The confirmed bitvae Python lacks os.pidfd_open. Linux x86_64 uses syscall434.
    require(platform.system() == "Linux" and platform.machine() == "x86_64",
            "This finite sequence requires the confirmed Linux x86_64 host.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(descriptor)


def wait_for_process(pid: int, expected_start: int) -> None:
    """Attach to one confirmed process and wait for its kernel exit event."""
    actual = process_start(pid)
    if actual is None:
        return
    require(actual == expected_start, f"PID {pid} was reused; refusing a different predecessor.")
    try:
        descriptor = exit_descriptor(pid)
    except ProcessLookupError:
        return
    try:
        require(process_start(pid) in (None, expected_start),
                f"PID {pid} identity changed while opening its exit descriptor.")
        select.select([descriptor], [], [])
    finally:
        os.close(descriptor)


def free_gpu(gpu: int) -> dict:
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    devices = {int(row.split(",")[0]): row.split(",")[1].strip() for row in inventory}
    require(gpu in devices, f"Assigned GPU {gpu} is absent.")
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    blockers = [row for row in active if row.split(",")[0].strip() == devices[gpu]]
    require(not blockers, f"Assigned GPU {gpu} is occupied; no waiting or retry: {blockers}")
    return {"gpu": gpu, "uuid": devices[gpu], "checked_unix": time.time(), "compute_processes": blockers}


def run_stage(command: list[str], *, cwd: Path, environment: dict, log: Path,
              report: dict, report_path: Path, name: str) -> None:
    """Run exactly once, record the actual exit code, and stop dependent execution on failure."""
    row = {"command": command, "cwd": str(cwd), "log": str(log),
           "started_unix": time.time(), "status": "running"}
    report.setdefault("stages", {})[name] = row
    write_json(report_path, report)
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
        row.update(pid=process.pid, start_ticks=process_start(process.pid))
        write_json(report_path, report)
        code = process.wait()
    row.update(exit_code=code, finished_unix=time.time(), status="complete" if code == 0 else "failed")
    write_json(report_path, report)
    require(code == 0, f"{name} failed with exit {code}; dependent tasks were not started. See {log}")
