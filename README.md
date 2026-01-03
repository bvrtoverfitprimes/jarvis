# JARVIS

Offline voice assistant with optional camera-based mouse control.

This project runs a local language model for responses, uses offline speech to text for commands, and speaks replies using Windows SAPI. It can also run a MediaPipe based hand and face tracker to control the mouse and trigger a few gestures.

## What it does

- Offline speech to text using Vosk
- Text to speech using Windows SAPI (smooth, interruptible)
- Local LLM inference using Transformers (Qwen2.5 0.5B Instruct by default)
- Mouse control using MediaPipe Tasks (hand and face landmarks)
- Unified launcher that waits for everything to load, then starts listening

## Quick start (Windows)

### 1) Create and activate a venv

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

### 3) Download the offline speech model (Vosk)

The assistant expects the Vosk model at:

- `models/vosk-model-small-en-us-0.15/`

You can download it automatically by running the helper once:

```powershell
python live_speech_to_text.py
```

Let it finish downloading and extracting, then press Ctrl+C.

### 4) Run the unified launcher

```powershell
python united_run.py
```

When both the LLM and mouse control are ready, it will say:

- Everything is ready to run

After that, Jarvis starts listening.

## How to use

### Wake word

Say:

- Jarvis
- Hey Jarvis

Then speak your command or question.

### Voice controls

- `stop` stops speaking and cancels the current generation
- `exit` or `quit` terminates everything
- `type ... enter` types into the currently focused window, then presses Enter

Tip: for typing, click the target app first (Notepad, browser address bar, chat box). Keyboard injection can be blocked if the target app runs elevated while Jarvis does not.

### Mouse control gestures

Mouse control runs in the background when using `united_run.py`.

- Open palm (high five) triggers a full shutdown
- Fist triggers Alt+Tab

## What gets downloaded locally

This repo intentionally does not commit large model assets.

- Vosk model goes under `models/`
- The LLM weights download into `qwen-model/` on first run

Both are ignored by `.gitignore` so you do not accidentally commit large files.

### MediaPipe task files

The MediaPipe `.task` files are small and are committed to the repo.

- Runtime expectation: `mouse_control.py` loads `hand_landmarker.task` and `face_landmarker.task` from the same folder as the script (repo root in the default layout).
- Organization: copies are also stored under `assets/mediapipe/` for clarity.

## Troubleshooting

- Camera not found: ensure a camera is connected and not locked by another app. Some systems expose cameras on a different index; the mouse process scans several indices.
- Microphone issues: check Windows input device settings and permissions.
- Typing or hotkeys not working: try running the terminal as Administrator if the target app is elevated.

## Repo layout

- `united_run.py`: starts mouse control and Jarvis, waits for readiness, then starts listening
- `jarvis_model.py`: main assistant loop (STT, LLM, TTS, stop/exit, typing)
- `mouse_control.py`: camera based mouse control and gestures
- `live_speech_to_text.py`: standalone offline STT utility (also useful for downloading the Vosk model)
- `face_landmarker.task`, `hand_landmarker.task`: MediaPipe task files

Supporting files:

- `assets/mediapipe/`: organized copies of the MediaPipe `.task` files
- `pyproject.toml`: minimal project metadata/config (does not change runtime)
- `.editorconfig`: consistent formatting rules across editors

