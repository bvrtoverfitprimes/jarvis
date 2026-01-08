import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _taskkill_tree(pid: int) -> None:
    if os.name != "nt":
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _stop_process(proc: subprocess.Popen, timeout_s: float = 4.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout_s)
            return
        except Exception:
            pass

        try:
            proc.kill()
        except Exception:
            pass

        try:
            proc.wait(timeout=timeout_s)
        except Exception:
            _taskkill_tree(proc.pid)
        return

    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout_s)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def main() -> int:
    root = Path(__file__).resolve().parent

    jarvis_path = root / "jarvis.py"
    mouse_path = root / "mouse_control.py"

    if not jarvis_path.exists() or not mouse_path.exists():
        print("Missing jarvis.py or mouse_control.py in the same folder.")
        return 2

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    jarvis_proc = subprocess.Popen(
        [sys.executable, str(jarvis_path)],
        cwd=str(root),
        creationflags=creationflags,
    )

    mouse_proc = subprocess.Popen(
        [sys.executable, str(mouse_path), "--parent-pid", str(os.getpid())],
        cwd=str(root),
        creationflags=creationflags,
    )

    stopping = False

    def _shutdown(_signum=None, _frame=None) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        _stop_process(jarvis_proc)
        _stop_process(mouse_proc)

    try:
        signal.signal(signal.SIGINT, _shutdown)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _shutdown)
    except Exception:
        pass

    exit_code = 0
    try:
        while True:
            j = jarvis_proc.poll()
            m = mouse_proc.poll()

            if j is not None:
                exit_code = int(j)
                _stop_process(mouse_proc)
                break

            if m is not None:
                exit_code = int(m)
                _stop_process(jarvis_proc)
                break

            time.sleep(0.2)
    finally:
        _shutdown()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
