# JARVIS

A unified AI assistant combining voice interaction via Qwen2.5 LLM with gesture-based mouse control. JARVIS integrates intelligent conversation, text-to-speech synthesis, and camera-based hand gesture recognition to create a seamless multimodal interface.

## Overview

JARVIS consists of two integrated components:

- **jarvis.py**: Voice-activated AI assistant powered by Qwen2.5-0.5B-Instruct with real-time speech recognition (Vosk) and text-to-speech synthesis. Activate with "Jarvis" spoken aloud, provide commands, and receive intelligent responses.
- **mouse_control.py**: Camera-based gesture interface using MediaPipe hand and face landmark detection. Control your cursor through hand position and trigger actions via specific gestures (open palm for shutdown, fist for Alt+Tab).

Both run simultaneously in a unified session managed by **runcentral.py**, which ensures that exiting either component gracefully terminates the entire application.

## How It Works

1. **Voice Component (jarvis.py)**
   - Listens for the wake word "Jarvis" via real-time speech recognition
   - Upon detection, enters active mode and processes your spoken command
   - Generates a response using the Qwen2.5 LLM
   - Streams the response as speech via system text-to-speech
   - Say "exit" to terminate both this component and mouse control

2. **Gesture Component (mouse_control.py)**
   - Continuously monitors hand landmarks via webcam
   - Tracks index finger position to control cursor movement with smoothing and gain adjustments
   - Detects hand poses: open palm (8+ consecutive frames) triggers shutdown, fist (6+ frames) triggers Alt+Tab
   - Requires both eyes open for cursor control
   - Open palm gesture terminates both this component and the voice assistant

3. **Central Manager (runcentral.py)**
   - Launches both jarvis.py and mouse_control.py in parallel
   - Monitors both processes and terminates the other when either exits
   - Ensures unified shutdown across all components

## Getting Started

### Prerequisites
- Python 3.8 or later
- Webcam (for gesture control)
- Microphone (for voice input)
- Windows environment (though core components may work on macOS/Linux with adjustments)

### Installation

#### 1) Clone the Repository

```powershell
git clone https://github.com/bvrtoverfitprimes/jarvis.git
cd jarvis
```

#### 2) Create and Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3) Install Dependencies

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

### Running JARVIS

To launch the complete system with both voice and gesture control:

```powershell
python runcentral.py
```

The system will start both components:
- Camera window opens showing hand detection overlay
- Voice assistant begins listening for the "Jarvis" wake word

### Individual Components

If you need to run components separately:

**Voice Assistant Only:**
```powershell
python jarvis.py
```

**Mouse Control Only:**
```powershell
python mouse_control.py
```

## Controls & Gestures

**Voice Commands:**
- Say "Jarvis" to activate the assistant
- Speak your command or question
- Say "exit" to shutdown the entire application
- Say "stop" to interrupt the current response

**Hand Gestures:**
- **Open Palm (High Five):** Triggers full application shutdown
- **Fist:** Simulates Alt+Tab window switching
- **Index Finger:** Controls cursor position (requires eyes open)
- **Pinch Gesture:** Left mouse button click/drag

## Project Structure

- `runcentral.py`: Central orchestrator that launches and manages both components
- `jarvis.py`: Voice assistant with LLM and speech synthesis
- `mouse_control.py`: Gesture recognition and cursor control engine
- `model.py`: Model loading and configuration utilities
- `assets/mediapipe/`: MediaPipe task model files (hand and face landmarks)
- `qwen-model/`: Cached Qwen2.5 LLM model
- `vosk-model-small-en-us-0.15/`: Voice recognition model
- `requirements.txt`: Python dependencies
- `pyproject.toml`: Project metadata

## Troubleshooting

- **Camera not opening:** Ensure no other application is using the camera. Try different camera indices with `--cam-id`.
- **Microphone not detected:** Check system audio settings and ensure microphone permissions are granted.
- **Models downloading slowly:** Large models (Qwen2.5 LLM, voice recognition) may take several minutes on first run.
- **Low gesture recognition:** Ensure adequate lighting and maintain hand visibility in camera frame.
