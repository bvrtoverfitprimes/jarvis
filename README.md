# Mouse Control

Camera-based mouse control and gestures using MediaPipe Tasks.

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

### 3) Run

```powershell
python mouse_control.py
```

### Mouse control gestures
- Open palm (high five) triggers a full shutdown
- Fist triggers Alt+Tab

## Repo layout

- `mouse_control.py`: camera based mouse control and gestures
- `face_landmarker.task`, `hand_landmarker.task`: MediaPipe task files

Supporting files:

- `assets/mediapipe/`: organized copies of the MediaPipe `.task` files
- `pyproject.toml`: minimal project metadata/config (does not change runtime)
- `.editorconfig`: consistent formatting rules across editors
