import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "1")

import cv2
import ctypes
import math
import time
import urllib.request
import threading
import queue
import numpy as np
import warnings
import argparse

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

warnings.filterwarnings(
    "ignore",
    message="Revert to STA COM threading mode",
    category=UserWarning,
)

try:
    from pywinauto import Desktop

    _HAS_UIA = True
except Exception:
    Desktop = None
    _HAS_UIA = False

try:
    import win32api
    import win32con
    import win32gui

    _HAS_WIN32 = True
except Exception:
    win32api = None
    win32con = None
    win32gui = None
    _HAS_WIN32 = False

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass

CAM_ID = 0
FLIP = True
MAX_HANDS = 1
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5

EXTENDED_THRESHOLD = 0.38
PINCH_DOWN_THRESHOLD = 0.30
PINCH_UP_THRESHOLD = 0.38

CURSOR_SMOOTHING = 0.6
CURSOR_GAIN = 1.5

SNAP_RADIUS_PX = 30
UI_SCAN_EVERY_N_FRAMES = 8

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_FILENAME = "face_landmarker.task"

# Require both eyes open to control mouse
BLINK_SCORE_THRESHOLD = 0.45
EAR_THRESHOLD = 0.22

STARTUP_WINDOW_NAME = "Starting..." 

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera-based mouse control")
    p.add_argument("--no-display", action="store_true", help="Run without showing any camera/progress windows")
    p.add_argument("--ready-file", default=None, help="If set, write this file once initialization is complete")
    p.add_argument("--shutdown-file", default=None, help="If set, write this file when an exit gesture is detected")
    p.add_argument("--parent-pid", type=int, default=None, help="If set, exit when this PID is no longer running")
    p.add_argument("--cam-id", type=int, default=CAM_ID, help="Camera index to use (default 0). If it fails, other indices will be tried.")
    return p.parse_args()


def _try_open_camera(index: int):
    try:
        cap = cv2.VideoCapture(int(index))
        if cap is not None and cap.isOpened():
            return cap, "DEFAULT"
        try:
            cap.release()
        except Exception:
            pass
    except Exception:
        pass

    if os.name == "nt":
        for backend, name in (
            (getattr(cv2, "CAP_DSHOW", None), "DSHOW"),
            (getattr(cv2, "CAP_MSMF", None), "MSMF"),
        ):
            if backend is None:
                continue
            try:
                cap = cv2.VideoCapture(int(index), int(backend))
                if cap is not None and cap.isOpened():
                    return cap, name
                try:
                    cap.release()
                except Exception:
                    pass
            except Exception:
                continue

    return None, ""


def _open_any_camera(preferred_index: int, scan_max: int = 6):
    cap, backend = _try_open_camera(preferred_index)
    if cap is not None:
        return cap, int(preferred_index), backend

    for idx in range(0, int(scan_max)):
        if idx == int(preferred_index):
            continue
        cap, backend = _try_open_camera(idx)
        if cap is not None:
            return cap, idx, backend

    return None, -1, ""


def _parent_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return True
    if os.name != "nt":
        return True
    try:
        SYNCHRONIZE = 0x00100000
        kernel32 = ctypes.windll.kernel32
        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        OpenProcess.restype = ctypes.c_void_p
        WaitForSingleObject = kernel32.WaitForSingleObject
        WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        WaitForSingleObject.restype = ctypes.c_uint32
        CloseHandle = kernel32.CloseHandle

        h = OpenProcess(SYNCHRONIZE, 0, int(pid))
        if not h:
            return False
        try:
            WAIT_TIMEOUT = 0x00000102
            rc = WaitForSingleObject(h, 0)
            return rc == WAIT_TIMEOUT
        finally:
            try:
                CloseHandle(h)
            except Exception:
                pass
    except Exception:
        return True


def _is_open_palm(lm: list[tuple[int, int]]) -> bool:
    try:
        wrist = lm[0]
        mid_mcp = lm[9]
        hand_size = dist(wrist, mid_mcp)
        if hand_size < 1:
            return False

        def extended(tip_i: int, pip_i: int, extra: float) -> bool:
            return dist(wrist, lm[tip_i]) > (dist(wrist, lm[pip_i]) + (extra * hand_size))

        idx_ok = extended(8, 6, 0.08)
        mid_ok = extended(12, 10, 0.08)
        ring_ok = extended(16, 14, 0.08)
        pink_ok = extended(20, 18, 0.08)
        thumb_ok = extended(4, 3, 0.06)

        if not (idx_ok and mid_ok and ring_ok and pink_ok and thumb_ok):
            return False

        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        min_adj = min(dist(tips[i], tips[i + 1]) for i in range(4))
        return min_adj > (0.18 * hand_size)
    except Exception:
        return False


def _is_fist(lm: list[tuple[int, int]]) -> bool:
    try:
        wrist = lm[0]
        mid_mcp = lm[9]
        hand_size = dist(wrist, mid_mcp)
        if hand_size < 1:
            return False

        def curled(tip_i: int, pip_i: int) -> bool:
            return dist(wrist, lm[tip_i]) < dist(wrist, lm[pip_i])

        idx_curled = curled(8, 6)
        mid_curled = curled(12, 10)
        ring_curled = curled(16, 14)
        pink_curled = curled(20, 18)

        thumb_curled = dist(wrist, lm[4]) < dist(wrist, lm[3]) + (0.1 * hand_size)

        return idx_curled and mid_curled and ring_curled and pink_curled and thumb_curled
    except Exception:
        return False


def _press_alt_tab() -> None:
    """Simulate Alt+Tab key press."""
    try:
        VK_MENU = 0x12  
        VK_TAB = 0x09
        KEYEVENTF_KEYUP = 0x0002

        user32.keybd_event(VK_MENU, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_TAB, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
    except Exception:
        pass


def _ensure_model_file(model_path: str, url: str, on_progress=None) -> None:
    if os.path.exists(model_path):
        if on_progress is not None:
            on_progress(1.0, "Model ready")
        return

    print(f"Model file not found: {model_path}")
    print("Downloading MediaPipe HandLandmarker model...")

    tmp_path = model_path + ".download"
    try:
        with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as f:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else 0
            downloaded = 0
            chunk_size = 256 * 1024

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if on_progress is not None:
                    if total_bytes > 0:
                        frac = max(0.0, min(1.0, downloaded / total_bytes))
                        on_progress(frac, "Downloading model")
                    else:
                        on_progress(0.0, f"Downloading model ({downloaded / (1024*1024):.1f} MB)")

        os.replace(tmp_path, model_path)
        if on_progress is not None:
            on_progress(1.0, "Model downloaded")
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise FileNotFoundError(
            "Could not download model automatically. Download it manually from: "
            f"{url} "
            "and place it next to main.py"
        ) from exc


def _eye_aspect_ratio(pts: list[tuple[int, int]]) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    num = dist(p2, p6) + dist(p3, p5)
    den = 2.0 * dist(p1, p4)
    if den <= 1e-6:
        return 0.0
    return num / den


def _both_eyes_open(face_result, img_w: int, img_h: int) -> bool:
    try:
        if getattr(face_result, "face_blendshapes", None):
            cats = face_result.face_blendshapes[0]
            scores = {c.category_name: c.score for c in cats}
            blink_l = float(scores.get("eyeBlinkLeft", 1.0))
            blink_r = float(scores.get("eyeBlinkRight", 1.0))
            return (blink_l < BLINK_SCORE_THRESHOLD) and (blink_r < BLINK_SCORE_THRESHOLD)
    except Exception:
        pass

    try:
        if getattr(face_result, "face_landmarks", None):
            lms = face_result.face_landmarks[0]

            def xy(i: int) -> tuple[int, int]:
                p = lms[i]
                return int(p.x * img_w), int(p.y * img_h)

            left_eye = [xy(33), xy(160), xy(158), xy(133), xy(153), xy(144)]
            right_eye = [xy(362), xy(385), xy(387), xy(263), xy(373), xy(380)]

            ear_l = _eye_aspect_ratio(left_eye)
            ear_r = _eye_aspect_ratio(right_eye)
            return (ear_l > EAR_THRESHOLD) and (ear_r > EAR_THRESHOLD)
    except Exception:
        pass

    return False

def _render_progress_window(progress: float, message: str, *, enabled: bool = True) -> bool:
    return True

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

user32 = ctypes.windll.user32
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

def _screen_size() -> tuple[int, int]:
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))

def _set_cursor_pos(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))

def _mouse_left_down() -> None:
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def _mouse_left_up() -> None:
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def _rect_center(rect) -> tuple[int, int]:
    return int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)

def _distance_sq(a: tuple[int, int], b: tuple[int, int]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _find_nearby_button(cx: int, cy: int, radius_px: int):
    if not _HAS_UIA:
        return None

    radius_sq = radius_px * radius_px
    offsets = [
        (0, 0),
        (-radius_px // 2, 0),
        (radius_px // 2, 0),
        (0, -radius_px // 2),
        (0, radius_px // 2),
        (-radius_px // 2, -radius_px // 2),
        (radius_px // 2, -radius_px // 2),
        (-radius_px // 2, radius_px // 2),
        (radius_px // 2, radius_px // 2),
    ]

    best = None
    best_d2 = None
    desktop = Desktop(backend="uia")

    for ox, oy in offsets:
        px = int(cx + ox)
        py = int(cy + oy)
        try:
            el = desktop.from_point(px, py)
        except Exception:
            continue

        cur = el
        for _ in range(8):
            try:
                ct = getattr(cur.element_info, "control_type", None)
                if ct == "Button":
                    rect = cur.rectangle()
                    bx, by = _rect_center(rect)
                    d2 = _distance_sq((cx, cy), (bx, by))
                    if d2 <= radius_sq and (best_d2 is None or d2 < best_d2):
                        best = cur
                        best_d2 = d2
                    break
                cur = cur.parent()
                if cur is None:
                    break
            except Exception:
                break

    return best


class _CursorOverlay:
    def __init__(self, radius_px: int):
        self.radius = int(radius_px)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = (0, 0, False, False)
        self._hwnd = None
        self._thread = None

        if _HAS_WIN32:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def update(self, x: int, y: int, active: bool, has_target: bool) -> None:
        if not _HAS_WIN32:
            return
        with self._lock:
            self._state = (int(x), int(y), bool(active), bool(has_target))
        hwnd = self._hwnd
        if hwnd:
            try:
                win32gui.PostMessage(hwnd, win32con.WM_APP + 1, 0, 0)
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        hwnd = self._hwnd
        if hwnd:
            try:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            except Exception:
                pass

    def _run(self) -> None:
        r = self.radius
        size = (2 * r) + 4
        class_name = "JARVIS_CURSOR_OVERLAY"

        transparent_key = win32api.RGB(255, 0, 255) 

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == win32con.WM_ERASEBKGND:
                try:
                    hdc = wparam
                    rc = win32gui.GetClientRect(hwnd)
                    brush = win32gui.CreateSolidBrush(transparent_key)
                    try:
                        win32gui.FillRect(hdc, rc, brush)
                    finally:
                        win32gui.DeleteObject(brush)
                    return 1
                except Exception:
                    return 1

            if msg == win32con.WM_PAINT:
                hdc, ps = win32gui.BeginPaint(hwnd)
                try:
                    try:
                        rc = win32gui.GetClientRect(hwnd)
                        brush = win32gui.CreateSolidBrush(transparent_key)
                        try:
                            win32gui.FillRect(hdc, rc, brush)
                        finally:
                            win32gui.DeleteObject(brush)
                    except Exception:
                        pass

                    with self._lock:
                        _, _, active, has_target = self._state
                    if not active:
                        color = win32api.RGB(120, 120, 120)
                    else:
                        color = win32api.RGB(0, 204, 102) if not has_target else win32api.RGB(255, 204, 0)

                    pen = win32gui.CreatePen(win32con.PS_SOLID, 3, color)
                    old_pen = win32gui.SelectObject(hdc, pen)
                    old_brush = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.HOLLOW_BRUSH))
                    win32gui.Ellipse(hdc, 2, 2, size - 2, size - 2)
                    win32gui.SelectObject(hdc, old_brush)
                    win32gui.SelectObject(hdc, old_pen)
                    win32gui.DeleteObject(pen)
                finally:
                    win32gui.EndPaint(hwnd, ps)
                return 0

            if msg == win32con.WM_APP + 1:
                with self._lock:
                    x, y, _, _ = self._state
                try:
                    win32gui.SetWindowPos(
                        hwnd,
                        win32con.HWND_TOPMOST,
                        x - r,
                        y - r,
                        size,
                        size,
                        win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW,
                    )
                    win32gui.InvalidateRect(hwnd, None, True)
                except Exception:
                    pass
                return 0

            if msg == win32con.WM_CLOSE:
                try:
                    win32gui.DestroyWindow(hwnd)
                except Exception:
                    pass
                return 0

            if msg == win32con.WM_DESTROY:
                try:
                    win32gui.PostQuitMessage(0)
                except Exception:
                    pass
                return 0

            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = wndproc
        wc.lpszClassName = class_name
        wc.hInstance = win32api.GetModuleHandle(None)
        try:
            win32gui.RegisterClass(wc)
        except Exception:
            pass

        ex_style = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
        style = win32con.WS_POPUP
        hwnd = win32gui.CreateWindowEx(
            ex_style,
            class_name,
            "",
            style,
            0,
            0,
            size,
            size,
            0,
            0,
            wc.hInstance,
            None,
        )
        self._hwnd = hwnd
        try:
            win32gui.SetLayeredWindowAttributes(hwnd, transparent_key, 0, win32con.LWA_COLORKEY)
        except Exception:
            pass

        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)
        win32gui.UpdateWindow(hwnd)

        win32gui.PumpMessages()


def main() -> None:
    args = _parse_args()
    display = not bool(getattr(args, "no_display", False))

    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    face_model_path = os.path.join(os.path.dirname(__file__), FACE_MODEL_FILENAME)

    def on_download_progress(frac: float, msg: str) -> None:
        return

    _ensure_model_file(model_path, MODEL_URL, on_progress=on_download_progress)

    def on_face_download_progress(frac: float, msg: str) -> None:
        return

    _ensure_model_file(face_model_path, FACE_MODEL_URL, on_progress=on_face_download_progress)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=MIN_DET_CONF,
        min_hand_presence_confidence=MIN_TRACK_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
        running_mode=vision.RunningMode.IMAGE,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=face_model_path),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.IMAGE,
    )

    q: queue.Queue = queue.Queue()

    def _load_model_worker() -> None:
        try:
            hand = vision.HandLandmarker.create_from_options(options)
            face = vision.FaceLandmarker.create_from_options(face_options)
            q.put((hand, face))
        except Exception as exc:
            q.put(exc)

    t = threading.Thread(target=_load_model_worker, daemon=True)
    t.start()

    start = time.time()
    hand_landmarker = None
    face_landmarker = None
    while True:
        try:
            item = q.get_nowait()
            if isinstance(item, Exception):
                raise item
            hand_landmarker, face_landmarker = item
            break
        except queue.Empty:
            pass

        elapsed = time.time() - start
        phase = (elapsed % 1.4) / 1.4
        if phase < 0.5:
            p = 0.60 + (0.90 - 0.60) * (phase / 0.5)
        else:
            p = 0.90 - (0.90 - 0.60) * ((phase - 0.5) / 0.5)

    cap, used_cam, used_backend = _open_any_camera(getattr(args, "cam_id", CAM_ID))
    if cap is None or not cap.isOpened():
        hand_landmarker.close()
        face_landmarker.close()
        raise SystemExit(
            "Cannot open camera. Check camera id or that it's not used by another app."
        )

    if not display:
        print(f"Camera opened: id={used_cam} backend={used_backend}")

    if getattr(args, "ready_file", None):
        try:
            with open(args.ready_file, "w", encoding="utf-8") as f:
                f.write("ready\n")
        except Exception:
            pass

    prev_time = 0.0
    screen_w, screen_h = _screen_size()
    cursor_x = screen_w / 2
    cursor_y = screen_h / 2
    mouse_down = False
    pinch_consumed = False
    frame_idx = 0
    cached_button = None
    cached_button_has_target = False
    overlay = _CursorOverlay(SNAP_RADIUS_PX)

    window_name = "Index Finger Detector"
    if display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Starting Index Finger Detector...")
        print("Press 'q' or Esc to quit")
    else:
        print("Starting mouse control (no display)...")

    try:
        open_palm_frames = 0
        fist_frames = 0
        fist_cooldown = 0  
        while True:
            if not _parent_is_alive(getattr(args, "parent_pid", None)):
                break
            frame_idx += 1
            if display and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            if FLIP:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face_landmarker.detect(mp_image)
            eyes_open = _both_eyes_open(face_result, w, h)
            result = hand_landmarker.detect(mp_image)

            status_text = "No hand"
            pinch_ratio = None
            control_active = False

            if not eyes_open:
                status_text = "Eyes closed (no control)"
                if mouse_down:
                    _mouse_left_up()
                    mouse_down = False
                pinch_consumed = False
                cached_button = None
                cached_button_has_target = False
                open_palm_frames = 0
                fist_frames = 0

            elif result.hand_landmarks:
                control_active = True
                hand_landmarks = result.hand_landmarks[0]

                lm = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]

                try:
                    if _is_open_palm(lm):
                        open_palm_frames += 1
                    else:
                        open_palm_frames = 0
                except Exception:
                    open_palm_frames = 0
                if open_palm_frames >= 8:
                    if getattr(args, "shutdown_file", None):
                        try:
                            with open(args.shutdown_file, "w", encoding="utf-8") as f:
                                f.write("shutdown\n")
                        except Exception:
                            pass
                    break

                if fist_cooldown > 0:
                    fist_cooldown -= 1
                try:
                    if _is_fist(lm):
                        fist_frames += 1
                    else:
                        fist_frames = 0
                except Exception:
                    fist_frames = 0
                if fist_frames >= 6 and fist_cooldown == 0:
                    _press_alt_tab()
                    fist_frames = 0
                    fist_cooldown = 30 

                wrist = lm[0]
                idx_pip = lm[6]
                idx_tip = lm[8]
                thumb_tip = lm[4]

                hand_size = dist(wrist, lm[9])  
                if hand_size < 1:
                    hand_size = 1.0

                tip_pip = dist(idx_tip, idx_pip)
                ratio = tip_pip / hand_size
                index_extended = ratio > EXTENDED_THRESHOLD

                pinch = dist(thumb_tip, idx_tip)
                pinch_ratio = pinch / hand_size
                if mouse_down:
                    pinched = pinch_ratio < PINCH_UP_THRESHOLD
                else:
                    pinched = pinch_ratio < PINCH_DOWN_THRESHOLD

                status_text = f"INDEX: {'UP' if index_extended else 'DOWN'}"

                mid_x = (thumb_tip[0] + idx_tip[0]) / 2.0
                mid_y = (thumb_tip[1] + idx_tip[1]) / 2.0

                raw_x = (mid_x / w) * screen_w
                raw_y = (mid_y / h) * screen_h
                target_x = (screen_w / 2) + (CURSOR_GAIN * (raw_x - (screen_w / 2)))
                target_y = (screen_h / 2) + (CURSOR_GAIN * (raw_y - (screen_h / 2)))

                if target_x < 0:
                    target_x = 0
                elif target_x > screen_w - 1:
                    target_x = screen_w - 1
                if target_y < 0:
                    target_y = 0
                elif target_y > screen_h - 1:
                    target_y = screen_h - 1

                cursor_x = (CURSOR_SMOOTHING * cursor_x) + ((1 - CURSOR_SMOOTHING) * target_x)
                cursor_y = (CURSOR_SMOOTHING * cursor_y) + ((1 - CURSOR_SMOOTHING) * target_y)
                _set_cursor_pos(int(cursor_x), int(cursor_y))

                if _HAS_UIA and (frame_idx % UI_SCAN_EVERY_N_FRAMES == 0):
                    try:
                        cached_button = _find_nearby_button(int(cursor_x), int(cursor_y), SNAP_RADIUS_PX)
                        cached_button_has_target = cached_button is not None
                    except Exception:
                        cached_button = None
                        cached_button_has_target = False

                if pinched and not mouse_down:
                    if _HAS_UIA and (not pinch_consumed):
                        btn = cached_button
                        if btn is None:
                            btn = _find_nearby_button(int(cursor_x), int(cursor_y), SNAP_RADIUS_PX)
                        if btn is not None:
                            try:
                                btn.invoke()
                            except Exception:
                                try:
                                    rect = btn.rectangle()
                                    bx, by = _rect_center(rect)
                                    _set_cursor_pos(bx, by)
                                    _mouse_left_down()
                                    _mouse_left_up()
                                except Exception:
                                    pass
                            pinch_consumed = True
                        else:
                            _mouse_left_down()
                            mouse_down = True
                    else:
                        _mouse_left_down()
                        mouse_down = True
                elif (not pinched) and mouse_down:
                    _mouse_left_up()
                    mouse_down = False

                if not pinched:
                    pinch_consumed = False

                if display:
                    cv2.circle(frame, idx_tip, 10, (0, 255, 0), -1)
                    cv2.line(frame, idx_pip, idx_tip, (255, 0, 0), 2)
                    cv2.circle(frame, thumb_tip, 10, (0, 200, 255), -1)
                    cv2.line(frame, thumb_tip, idx_tip, (0, 200, 255), 2)
                    cv2.putText(
                        frame,
                        f"({idx_tip[0]}, {idx_tip[1]})",
                        (idx_tip[0] + 12, idx_tip[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"ratio {ratio:.2f}",
                        (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    if pinch_ratio is not None:
                        cv2.putText(
                            frame,
                            f"pinch {pinch_ratio:.2f}",
                            (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 200, 255),
                            2,
                            cv2.LINE_AA,
                        )
            else:
                if mouse_down:
                    _mouse_left_up()
                    mouse_down = False
                pinch_consumed = False
                cached_button = None
                cached_button_has_target = False
                open_palm_frames = 0
                fist_frames = 0

            try:
                overlay.update(int(cursor_x), int(cursor_y), control_active, cached_button_has_target)
            except Exception:
                pass

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            if display:
                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if "UP" in status_text else (0, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if mouse_down:
            _mouse_left_up()
        try:
            overlay.close()
        except Exception:
            pass
        cap.release()
        if display:
            cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
