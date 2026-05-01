import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_pid(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_pid_group(pid: Optional[int], *, label: str, grace_sec: float = 10.0) -> None:
    if not _pid_alive(pid):
        return
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = None

    targets = []
    if pgid is not None:
        targets.append(("pgid", pgid))
    targets.append(("pid", pid))

    for _, target in targets:
        try:
            if target == pid:
                os.kill(target, signal.SIGTERM)
            else:
                os.killpg(target, signal.SIGTERM)
            break
        except OSError:
            continue

    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(1.0)

    for _, target in targets:
        try:
            if target == pid:
                os.kill(target, signal.SIGKILL)
            else:
                os.killpg(target, signal.SIGKILL)
            break
        except OSError:
            continue


def _healthcheck(base_url: str, timeout_sec: float = 10.0) -> Tuple[bool, str]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"{base_url.rstrip('/')}/health", timeout=timeout_sec)
        if response.ok:
            return True, response.text.strip()
        return False, f"{response.status_code} {response.text.strip()}"
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def _log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[watch] {_now()} {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _dataset_paths(workspace_root: Path, dataset_key: str, run_name: str) -> Dict[str, Path]:
    base = workspace_root / "outputs" / dataset_key / run_name
    return {
        "dir": base,
        "results": base / "results.jsonl",
        "summary": base / "summary.json",
        "errors": base / "errors.jsonl",
    }


def _aggregate_scores(workspace_root: Path, run_name: str, dataset_keys: List[str], output_path: Path) -> Path:
    per_dataset: Dict[str, Any] = {}
    total = correct = wrong = invalid = 0

    for dataset_key in dataset_keys:
        summary_path = _dataset_paths(workspace_root, dataset_key, run_name)["summary"]
        summary = _read_json(summary_path)
        if summary is None:
            raise RuntimeError(f"Missing summary for {dataset_key}: {summary_path}")
        per_dataset[dataset_key] = summary
        total += int(summary.get("total", 0))
        correct += int(summary.get("correct", 0))
        wrong += int(summary.get("wrong", 0))
        invalid += int(summary.get("invalid", 0))

    aggregate = {
        "run_name": run_name,
        "completed_at": _now(),
        "datasets": per_dataset,
        "overall": {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "invalid": invalid,
            "accuracy": (correct / total) if total else 0.0,
            "invalid_rate": (invalid / total) if total else 0.0,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _start_service(repo_root: Path, model_config: Path, service_pid_file: Path, service_log: Path) -> int:
    service_log.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"nohup tri-agent-serve --model-config {str(model_config)!r} "
        f"> {str(service_log)!r} 2>&1 < /dev/null & echo $! > {str(service_pid_file)!r}"
    )
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=repo_root,
        env=_base_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    del completed
    pid = _read_pid(service_pid_file)
    if pid is None:
        raise RuntimeError(f"Failed to read restarted service pid from {service_pid_file}")
    return pid


def _start_batch(
    *,
    repo_root: Path,
    model_config: Path,
    dataset_keys: List[str],
    run_name: str,
    batch_log: Path,
    batch_pid_file: Path,
) -> int:
    batch_log.parent.mkdir(parents=True, exist_ok=True)
    datasets_literal = " ".join(dataset_keys)
    script = f"""
set -euo pipefail
cd {str(repo_root)!r}
for ds in {datasets_literal}; do
  echo "[batch] starting ${{ds}} $(date --iso-8601=seconds)"
  tri-agent eval --model-config {str(model_config)!r} --dataset-config "configs/datasets/${{ds}}.yaml" --run-name {run_name!r} --resume
  status=$?
  echo "[batch] finished ${{ds}} status=${{status}} $(date --iso-8601=seconds)"
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
done
echo "[batch] all done $(date --iso-8601=seconds)"
"""
    handle = batch_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", script],
        cwd=repo_root,
        env=_base_env(),
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    batch_pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    return process.pid


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch and auto-resume sequential Qwen3-Omni evaluation runs.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--service-url", type=str, default="http://127.0.0.1:18080")
    parser.add_argument("--datasets", nargs="+", default=["daily_omni", "worldsense", "omnibench"])
    parser.add_argument("--batch-log", type=Path, required=True)
    parser.add_argument("--batch-pid-file", type=Path, required=True)
    parser.add_argument("--service-pid-file", type=Path, default=Path("/tmp/tri_agent_serve.pid"))
    parser.add_argument("--service-log", type=Path, default=Path("/tmp/tri_agent_serve.log"))
    parser.add_argument("--monitor-log", type=Path, required=True)
    parser.add_argument("--aggregate-summary", type=Path, required=True)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--stale-sec", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dataset_keys = list(args.datasets)
    last_progress_marker = ""

    _log_line(args.monitor_log, f"watchdog_started run_name={args.run_name} datasets={dataset_keys}")

    while True:
        summaries_present: Dict[str, bool] = {}
        results_rows: Dict[str, int] = {}
        newest_progress_mtime = 0.0

        for dataset_key in dataset_keys:
            paths = _dataset_paths(args.workspace_root, dataset_key, args.run_name)
            summaries_present[dataset_key] = paths["summary"].exists()
            results_rows[dataset_key] = _count_jsonl_rows(paths["results"])
            for key in ("results", "summary", "errors"):
                if paths[key].exists():
                    newest_progress_mtime = max(newest_progress_mtime, paths[key].stat().st_mtime)

        if all(summaries_present.values()):
            aggregate_path = _aggregate_scores(
                args.workspace_root,
                args.run_name,
                dataset_keys,
                args.aggregate_summary,
            )
            _log_line(args.monitor_log, f"all_datasets_finished aggregate={aggregate_path}")
            return 0

        pending = [dataset for dataset in dataset_keys if not summaries_present[dataset]]
        batch_pid = _read_pid(args.batch_pid_file)
        batch_alive = _pid_alive(batch_pid)
        service_pid = _read_pid(args.service_pid_file)
        service_alive = _pid_alive(service_pid)
        health_ok, health_detail = _healthcheck(args.service_url)

        progress_marker = ",".join(f"{name}:{results_rows[name]}" for name in dataset_keys)
        if progress_marker != last_progress_marker:
            last_progress_marker = progress_marker
            _log_line(
                args.monitor_log,
                f"progress batch_pid={batch_pid} batch_alive={batch_alive} "
                f"service_pid={service_pid} service_alive={service_alive} "
                f"health_ok={health_ok} pending={pending} rows={progress_marker}",
            )

        reference_mtime = newest_progress_mtime
        if args.batch_log.exists():
            reference_mtime = max(reference_mtime, args.batch_log.stat().st_mtime)
        stale_for_sec = time.time() - reference_mtime if reference_mtime else 0.0

        if not health_ok:
            _log_line(args.monitor_log, f"healthcheck_failed detail={health_detail}")

        if batch_alive:
            if not health_ok and stale_for_sec >= args.stale_sec:
                _log_line(
                    args.monitor_log,
                    f"stale_and_unhealthy stale_for_sec={int(stale_for_sec)} action=restart_batch_and_service",
                )
                _kill_pid_group(batch_pid, label="batch")
                _kill_pid_group(service_pid, label="service")
                restarted_service_pid = _start_service(
                    args.repo_root,
                    args.model_config,
                    args.service_pid_file,
                    args.service_log,
                )
                _log_line(args.monitor_log, f"service_restarted pid={restarted_service_pid}")
                restarted_batch_pid = _start_batch(
                    repo_root=args.repo_root,
                    model_config=args.model_config,
                    dataset_keys=pending,
                    run_name=args.run_name,
                    batch_log=args.batch_log,
                    batch_pid_file=args.batch_pid_file,
                )
                _log_line(args.monitor_log, f"batch_restarted pid={restarted_batch_pid} pending={pending}")
        else:
            if service_alive:
                _kill_pid_group(service_pid, label="service")
            restarted_service_pid = _start_service(
                args.repo_root,
                args.model_config,
                args.service_pid_file,
                args.service_log,
            )
            _log_line(args.monitor_log, f"service_restarted pid={restarted_service_pid}")
            restarted_batch_pid = _start_batch(
                repo_root=args.repo_root,
                model_config=args.model_config,
                dataset_keys=pending,
                run_name=args.run_name,
                batch_log=args.batch_log,
                batch_pid_file=args.batch_pid_file,
            )
            _log_line(args.monitor_log, f"batch_started pid={restarted_batch_pid} pending={pending}")

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    sys.exit(main())
