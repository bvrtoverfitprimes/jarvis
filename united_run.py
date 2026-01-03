import os
import sys
import time
import tempfile
import subprocess


def _tail_text_file(path: str, max_lines: int = 60) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)
        lines = text.splitlines()
        tail = lines[-max_lines:]
        return "\n".join(tail)
    except Exception:
        return ""


def _safe_unlink(p: str) -> None:
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except Exception:
        pass


def _touch(p: str, text: str = "") -> None:
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def _wait_for_file(p: str, proc: subprocess.Popen | None, timeout_s: float) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            if p and os.path.exists(p):
                return True
        except Exception:
            return True
        time.sleep(0.05)
    return False


def _speak_ready_phrase(text: str) -> None:
    try:
        import win32com.client 

        v = win32com.client.Dispatch("SAPI.SpVoice")
        v.Speak(text)
        return
    except Exception:
        pass

    if os.name == "nt":
        safe = text.replace("'", "''").replace('"', '\\"')
        ps = (
            "Add-Type -AssemblyName System.speech;"
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.Rate=0;"
            f"$s.Speak('{safe}')"
        )
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            creationflags = 0
        try:
            subprocess.run(
                ["powershell", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
                check=False,
            )
        except Exception:
            pass


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    jarvis_script = os.path.join(root, "jarvis_model.py")
    mouse_script = os.path.join(root, "mouse_control.py")

    if not os.path.exists(jarvis_script):
        print("Missing jarvis_model.py")
        return 1
    if not os.path.exists(mouse_script):
        print("Missing mouse_control.py")
        return 1

    tmp = tempfile.gettempdir()
    stamp = f"{os.getpid()}_{int(time.time())}"

    logs_dir = os.path.join(root, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        logs_dir = tmp
    mouse_log_path = os.path.join(logs_dir, f"mouse_control_{stamp}.log")
    jarvis_log_path = os.path.join(logs_dir, f"jarvis_{stamp}.log")

    mouse_ready = os.path.join(tmp, f"jarvis_mouse_ready_{stamp}.txt")
    mouse_shutdown = os.path.join(tmp, f"jarvis_mouse_shutdown_{stamp}.txt")
    jarvis_model_ready = os.path.join(tmp, f"jarvis_model_ready_{stamp}.txt")
    jarvis_start = os.path.join(tmp, f"jarvis_start_{stamp}.txt")
    jarvis_shutdown = os.path.join(tmp, f"jarvis_shutdown_{stamp}.txt")

    for p in [mouse_ready, mouse_shutdown, jarvis_model_ready, jarvis_start, jarvis_shutdown]:
        _safe_unlink(p)

    creationflags = 0
    if os.name == "nt":
        try:
            creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            creationflags = 0

    print("Starting mouse control...")
    mouse_log = open(mouse_log_path, "ab", buffering=0)
    mouse_proc = subprocess.Popen(
        [
            sys.executable,
            mouse_script,
            "--no-display",
            "--ready-file",
            mouse_ready,
            "--shutdown-file",
            mouse_shutdown,
            "--parent-pid",
            str(os.getpid()),
        ],
        stdout=mouse_log,
        stderr=mouse_log,
        creationflags=creationflags,
    )

    print("Starting Jarvis...")
    env = os.environ.copy()
    env["JARVIS_ORCHESTRATED"] = "1"
    env["JARVIS_MODEL_READY_FILE"] = jarvis_model_ready
    env["JARVIS_START_FILE"] = jarvis_start
    env["JARVIS_SHUTDOWN_FILE"] = jarvis_shutdown

    jarvis_log = open(jarvis_log_path, "ab", buffering=0)
    jarvis_proc = subprocess.Popen([sys.executable, jarvis_script], env=env, stderr=jarvis_log)

    try:
        if not _wait_for_file(jarvis_model_ready, jarvis_proc, timeout_s=300.0):
            print("Jarvis failed to become ready.")
            try:
                print(f"Jarvis exit code: {jarvis_proc.poll()}")
            except Exception:
                pass
            tail = _tail_text_file(jarvis_log_path)
            if tail:
                print("--- Jarvis log (tail) ---")
                print(tail)
            return 1
        if not _wait_for_file(mouse_ready, mouse_proc, timeout_s=300.0):
            print("Mouse control failed to become ready.")
            try:
                print(f"Mouse exit code: {mouse_proc.poll()}")
            except Exception:
                pass
            tail = _tail_text_file(mouse_log_path)
            if tail:
                print("--- Mouse log (tail) ---")
                print(tail)
            return 1

        _speak_ready_phrase("Everything is ready to run")
        _touch(jarvis_start, "start\n")

        # Supervisor loop
        while True:
            if jarvis_proc.poll() is not None:
                break
            if mouse_proc.poll() is not None:
                _touch(jarvis_shutdown, "shutdown\n")
                break
            if os.path.exists(mouse_shutdown):
                _touch(jarvis_shutdown, "shutdown\n")
                break
            time.sleep(0.05)

    except KeyboardInterrupt:
        _touch(jarvis_shutdown, "shutdown\n")
    finally:
        try:
            if jarvis_proc.poll() is None:
                jarvis_proc.terminate()
        except Exception:
            pass
        try:
            if mouse_proc.poll() is None:
                mouse_proc.terminate()
        except Exception:
            pass

        try:
            mouse_log.close()
        except Exception:
            pass
        try:
            jarvis_log.close()
        except Exception:
            pass

        try:
            jarvis_proc.wait(timeout=3.0)
        except Exception:
            try:
                jarvis_proc.kill()
            except Exception:
                pass

        try:
            mouse_proc.wait(timeout=3.0)
        except Exception:
            try:
                mouse_proc.kill()
            except Exception:
                pass

        for p in [mouse_ready, mouse_shutdown, jarvis_model_ready, jarvis_start, jarvis_shutdown]:
            _safe_unlink(p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
