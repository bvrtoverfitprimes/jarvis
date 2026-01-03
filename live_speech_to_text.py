from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
import urllib.request
import zipfile

def _print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _pip_install(packages: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    _print_err("Installing: " + " ".join(packages))
    subprocess.check_call(cmd)

def _ensure_deps() -> None:
    missing: list[str] = []
    try:
        import vosk  
    except Exception:
        missing.append("vosk==0.3.45")

    try:
        import sounddevice  
    except Exception:
        missing.append("sounddevice>=0.5,<0.6")

    if not missing:
        return

    try:
        _pip_install(missing)
    except Exception as exc:
        raise RuntimeError(f"Auto-install failed: {exc}")


def _download_with_progress(url: str, out_path: str) -> None:
    tmp_path = out_path + ".download"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as f:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else 0
        downloaded = 0
        chunk_size = 256 * 1024
        last = time.time()

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last >= 0.15:
                last = now
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100.0
                    _print_err(f"Downloading model: {pct:5.1f}% ({downloaded/1e6:.1f}MB/{total_bytes/1e6:.1f}MB)")
                else:
                    _print_err(f"Downloading model: {downloaded/1e6:.1f}MB")

    os.replace(tmp_path, out_path)


def _ensure_vosk_model(model_dir: str | None) -> str:
    if model_dir:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        return model_dir

    models_root = os.path.join(os.path.dirname(__file__), "models")
    model_name = "vosk-model-small-en-us-0.15"
    model_path = os.path.join(models_root, model_name)

    if os.path.isdir(model_path):
        return model_path

    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = os.path.join(models_root, f"{model_name}.zip")

    _print_err("Downloading speech model...")
    _download_with_progress(url, zip_path)

    _print_err("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(models_root)

    try:
        os.remove(zip_path)
    except OSError:
        pass

    if not os.path.isdir(model_path):
        raise RuntimeError("Model download/extract finished, but model folder was not found.")

    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Live speech-to-text (offline, real-time)")
    parser.add_argument("--model-dir", default=None, help="Path to Vosk model directory")
    parser.add_argument("--device", type=int, default=None, help="Input device index (optional)")
    parser.add_argument("--samplerate", type=int, default=None, help="Sample rate (optional)")
    parser.add_argument("--blocksize", type=int, default=8000, help="Audio blocksize (frames)")
    args = parser.parse_args()

    try:
        _ensure_deps()
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
    except Exception as exc:
        _print_err(f"Import error: {exc}")
        return 2

    model_path = _ensure_vosk_model(args.model_dir)

    device_info = None
    if args.device is not None:
        try:
            device_info = sd.query_devices(args.device, "input")
        except Exception:
            device_info = None

    if args.samplerate is not None:
        samplerate = float(args.samplerate)
    else:
        try:
            if device_info is None:
                device_info = sd.query_devices(None, "input")
            samplerate = float(device_info["default_samplerate"])  
        except Exception:
            samplerate = 16000.0

    _print_err(f"Loading model: {model_path}")
    model = Model(model_path)
    rec = KaldiRecognizer(model, samplerate)
    rec.SetWords(True)

    audio_q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            _print_err(str(status))
        audio_q.put(bytes(indata))

    _print_err("Listening... (Ctrl+C to stop)")

    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=args.blocksize,
            dtype="int16",
            channels=1,
            callback=callback,
            device=args.device,
        ):
            last_partial = ""
            while True:
                data = audio_q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result() or "{}")
                    text = (result.get("text") or "").strip()
                    if text:
                        print(text)
                    last_partial = ""
                else:
                    partial = json.loads(rec.PartialResult() or "{}").get("partial") or ""
                    partial = partial.strip()
                    if partial and partial != last_partial:
                        print(partial + " " * 10, end="\r", flush=True)
                        last_partial = partial
    except KeyboardInterrupt:
        _print_err("\nStopped.")
        return 0
    except Exception as exc:
        _print_err(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
