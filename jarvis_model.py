import sys,subprocess,threading,platform,shutil,queue,time,re,textwrap,os,json,tempfile,ctypes,traceback
from pathlib import Path
from array import array
print("Process starting please wait. This might take a while.")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import TextIteratorStreamer
    except Exception:
        TextIteratorStreamer=None

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        StoppingCriteria=None
        StoppingCriteriaList=None
except Exception:
    subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","transformers","accelerate","sentencepiece","safetensors","huggingface-hub"])
    try:
        import torch
    except Exception:
        subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","torch"])
        import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    try:
        from transformers import TextIteratorStreamer
    except Exception:
        TextIteratorStreamer=None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        StoppingCriteria=None
        StoppingCriteriaList=None

GEN_STOP_EVENT=threading.Event()
EXIT_EVENT=threading.Event()
ORCHESTRATED = os.environ.get("JARVIS_ORCHESTRATED", "0") == "1"

def _type_text_windows(text: str) -> bool:
    """Type unicode text into the focused window (Windows only) using clipboard paste."""
    if os.name != "nt":
        return False
    try:
        import subprocess
        
        escaped = text.replace("'", "''")
        ps_cmd = f"Set-Clipboard -Value '{escaped}'"
        creationflags = 0
        try:
            creationflags = subprocess.CREATE_NO_WINDOW
        except Exception:
            pass
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            check=False,
            timeout=5.0,
        )
        
        time.sleep(0.05)
        
        user32 = ctypes.windll.user32
        VK_CONTROL = 0x11
        VK_V = 0x56
        KEYEVENTF_KEYUP = 0x0002

        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_V, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)

        return True
    except Exception as e:
        print(f"[Typing error] {e}")
        return False


def _type_text_windows_sendinput(text: str) -> bool:
    """Type unicode text using SendInput (backup, often blocked by UAC)."""
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32

        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002
        KEYEVENTF_UNICODE = 0x0004

        ULONG_PTR = ctypes.c_size_t

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class INPUT_I(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("ii", INPUT_I)]

        user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(INPUT), ctypes.c_int)
        user32.SendInput.restype = ctypes.c_uint

        def send_unicode_char(ch: str) -> None:
            code = ord(ch)
            down = INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, 0)))
            up = INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)))
            arr = (INPUT * 2)(down, up)
            user32.SendInput(2, arr, ctypes.sizeof(INPUT))

        for ch in text:
            if ch == "\n":
                _press_enter_windows()
            else:
                send_unicode_char(ch)
        return True
    except Exception:
        return False


def _press_enter_windows() -> bool:
    """Press Enter key using keybd_event (reliable)."""
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        VK_RETURN = 0x0D
        KEYEVENTF_KEYUP = 0x0002

        user32.keybd_event(VK_RETURN, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception as e:
        print(f"[Enter key error] {e}")
        return False


if 'StoppingCriteria' in globals() and StoppingCriteria is not None:
    class _StopOnEvent(StoppingCriteria):
        def __init__(self, evt):
            self.evt=evt
        def __call__(self, input_ids, scores, **kwargs):
            try:
                return bool(self.evt.is_set())
            except Exception:
                return False


def _make_stopping_criteria(evt):
    if ('StoppingCriteriaList' not in globals()) or (StoppingCriteriaList is None):
        return None
    try:
        return StoppingCriteriaList([_StopOnEvent(evt)])
    except Exception:
        return None

class SpeechManager:
    def __init__(self):
        self.sentence_q=queue.Queue()
        self.text_buf=""
        self.lock=threading.Lock()
        self.current_proc=None
        self.active=True
        self.is_speaking=False
        self.min_chars=25
        self.max_chars=120
        self._interrupt_evt=threading.Event()
        self._muted=False
        self.worker=threading.Thread(target=self._worker,daemon=True);self.worker.start()

    def begin_utterance(self):
        """Allow speech output for a new assistant response."""
        self._muted=False
        self._interrupt_evt.clear()

    def interrupt(self):
        """Stop speaking immediately and drop queued phrases."""
        self._interrupt_evt.set()
        self._muted=True
        if self.current_proc:
            try: self.current_proc.terminate()
            except: pass
        self.text_buf=""
        try:
            while True:
                self.sentence_q.get_nowait(); self.sentence_q.task_done()
        except queue.Empty:
            pass
    def _speak_blocking(self,t):
        try:
            osys=platform.system()
            if osys=="Darwin":
                p=subprocess.Popen(["say","-r","165",t],stdout=subprocess.PIPE,stderr=subprocess.PIPE);self.current_proc=p; p.wait()
            elif osys=="Windows":
                safe=t.replace("'", "''").replace('"','\\"')
                ps=f"Add-Type -AssemblyName System.speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;$s.Rate=0;$s.Speak('{safe}')"
                p=subprocess.Popen(["powershell","-Command",ps],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW);self.current_proc=p; p.wait()
            else:
                if shutil.which("espeak"):
                    p=subprocess.Popen(["espeak","-s","145",t],stdout=subprocess.PIPE,stderr=subprocess.PIPE);self.current_proc=p; p.wait()
                elif shutil.which("spd-say"):
                    p=subprocess.Popen(["spd-say","-r","0","-w",t],stdout=subprocess.PIPE,stderr=subprocess.PIPE);self.current_proc=p; p.wait()
                else:
                    import pyttsx3
                    e=pyttsx3.init(); 
                    try: e.setProperty("rate",165); e.setProperty("volume",1.0)
                    except: pass
                    e.say(t); e.runAndWait()
        except Exception as e:
            print("Speech error:",e)
        finally:
            self.current_proc=None
    def _worker(self):
        osys = platform.system()
        sapi_voice = None
        
        if osys == "Windows":
            try:
                import win32com.client
                import pythoncom
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
            except:
                sapi_voice = None
        
        while self.active:
            try:
                s=self.sentence_q.get(timeout=0.05)
                if s is None: break
                if s.strip():
                    with self.lock: self.is_speaking=True
                    if sapi_voice:
                        try:
                            self._interrupt_evt.clear()
                            sapi_voice.Speak(s, 1) 
                            while True:
                                if self._interrupt_evt.is_set():
                                    try:
                                        sapi_voice.Speak("", 3) 
                                    except Exception:
                                        pass
                                    break
                                try:
                                    done = sapi_voice.WaitUntilDone(50)
                                except Exception:
                                    done = True
                                if done:
                                    break
                        except Exception as e:
                            # Fallback to subprocess if SAPI fails
                            self._speak_blocking(s)
                    else:
                        self._speak_blocking(s)
                    with self.lock: self.is_speaking=False
                self.sentence_q.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print("Speech worker error:",e)
                with self.lock: self.is_speaking=False
        
        if osys == "Windows" and sapi_voice:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except:
                pass
    def add_text(self,t):
        if self._muted: return
        if not t or not self.active: return
        t=re.sub(r'\s+',' ',t).strip()
        if self.text_buf and not self.text_buf.endswith(' '): self.text_buf+=' '
        self.text_buf+=t
        self._process()
    def _process(self):
        if not self.text_buf: return
        pattern=r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        parts=re.split(pattern,self.text_buf)
        if len(parts)>1:
            for s in parts[:-1]:
                s=s.strip()
                if s and len(s)>3: self.sentence_q.put(s)
            self.text_buf=parts[-1].strip()
            if len(self.text_buf)>self.max_chars:
                last_space=self.text_buf.rfind(' ')
                if last_space>20:
                    to_speak=self.text_buf[:last_space].strip(); rem=self.text_buf[last_space:].strip()
                    if to_speak: self.sentence_q.put(to_speak)
                    self.text_buf=rem
        else:
            if len(self.text_buf)>self.max_chars:
                for p in ['. ', '! ', '? ', ', ', '; ', ': ']:
                    pos=self.text_buf.rfind(p)
                    if pos>20:
                        to_speak=self.text_buf[:pos+2].strip(); rem=self.text_buf[pos+2:].strip()
                        if to_speak: self.sentence_q.put(to_speak)
                        self.text_buf=rem
                        break
                else:
                    last_space=self.text_buf.rfind(' ')
                    if last_space>20:
                        to_speak=self.text_buf[:last_space].strip(); rem=self.text_buf[last_space:].strip()
                        if to_speak: self.sentence_q.put(to_speak)
                        self.text_buf=rem
    def flush(self):
        if self._muted:
            self.text_buf=""
            return True
        if self.text_buf.strip() and len(self.text_buf.strip())>3:
            self.sentence_q.put(self.text_buf.strip())
        self.text_buf=""
        return self.sentence_q.empty()
    def is_busy(self):
        with self.lock:
            return self.is_speaking or not self.sentence_q.empty()
    def wait_until_done(self,timeout=10.0):
        start=time.time()
        while self.is_busy() and (time.time()-start)<timeout: time.sleep(0.05)
        return not self.is_busy()
    def stop(self):
        self.active=False
        self.sentence_q.put(None)
        if self.current_proc:
            try: self.current_proc.terminate()
            except: pass

speech_manager=SpeechManager()


_mouse_proc=None
_mouse_ready_file=None
_mouse_shutdown_file=None


def _start_mouse_control():
    """Run mouse control alongside Jarvis without showing camera windows."""
    global _mouse_proc,_mouse_ready_file,_mouse_shutdown_file
    if _mouse_proc is not None:
        return
    script=os.path.join(os.path.dirname(__file__),"mouse_control.py")
    if not os.path.exists(script):
        return
    try:
        _mouse_ready_file=os.path.join(tempfile.gettempdir(),f"jarvis_mouse_ready_{os.getpid()}.txt")
        _mouse_shutdown_file=os.path.join(tempfile.gettempdir(),f"jarvis_mouse_shutdown_{os.getpid()}.txt")
        try:
            if os.path.exists(_mouse_ready_file):
                os.remove(_mouse_ready_file)
        except Exception:
            pass
        try:
            if os.path.exists(_mouse_shutdown_file):
                os.remove(_mouse_shutdown_file)
        except Exception:
            pass
    except Exception:
        _mouse_ready_file=None
        _mouse_shutdown_file=None
    creationflags=0
    if platform.system()=="Windows":
        try:
            creationflags=subprocess.CREATE_NO_WINDOW
        except Exception:
            creationflags=0
    try:
        _mouse_proc=subprocess.Popen(
            [
                sys.executable,
                script,
                "--no-display",
                "--ready-file",
                str(_mouse_ready_file or ""),
                "--shutdown-file",
                str(_mouse_shutdown_file or ""),
                "--parent-pid",
                str(os.getpid()),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        _mouse_proc=None


def _wait_for_mouse_ready(timeout=120.0):
    """Wait until mouse_control reports ready (or until timeout)."""
    if _mouse_proc is None:
        return True
    if not _mouse_ready_file:
        return True
    deadline=time.time()+float(timeout)
    while time.time()<deadline:
        try:
            if _mouse_proc.poll() is not None:
                return False
        except Exception:
            pass
        try:
            if os.path.exists(_mouse_ready_file):
                return True
        except Exception:
            return True
        time.sleep(0.05)
    return False


def _stop_mouse_control():
    global _mouse_proc,_mouse_ready_file,_mouse_shutdown_file
    if _mouse_proc is None:
        return
    try:
        _mouse_proc.terminate()
    except Exception:
        pass
    try:
        _mouse_proc.wait(timeout=2.0)
    except Exception:
        try:
            _mouse_proc.kill()
        except Exception:
            pass
    _mouse_proc=None
    _mouse_ready_file=None
    _mouse_shutdown_file=None


def _start_mouse_shutdown_watcher():
    def _watch():
        while True:
            if EXIT_EVENT.is_set():
                return
            if _mouse_proc is not None:
                try:
                    if _mouse_proc.poll() is not None:
                        EXIT_EVENT.set();
                        return
                except Exception:
                    pass
            if _mouse_shutdown_file:
                try:
                    if os.path.exists(_mouse_shutdown_file):
                        EXIT_EVENT.set();
                        return
                except Exception:
                    pass
            time.sleep(0.05)

    threading.Thread(target=_watch, daemon=True).start()


def _pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def _ensure_stt_deps():
    missing=[]
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
    print("Installing speech dependencies...", " ".join(missing))
    _pip_install(missing)


def _ensure_vosk_model(model_dir=None):
    if model_dir and os.path.isdir(model_dir):
        return model_dir
    models_root=os.path.join(os.path.dirname(__file__),"models")
    model_name="vosk-model-small-en-us-0.15"
    model_path=os.path.join(models_root,model_name)
    if os.path.isdir(model_path):
        return model_path
    raise FileNotFoundError(
        f"Vosk model not found at: {model_path}. "
    )


class LiveSpeechToText:
    def __init__(self, model_dir=None, device=None, samplerate=None, blocksize=2048):
        self.model_dir=model_dir
        self.device=device
        self.samplerate=samplerate
        self.blocksize=blocksize
        self.events=queue.Queue()
        self._stop=threading.Event()
        self._paused=threading.Event()
        self._thread=threading.Thread(target=self._run,daemon=True)
        self.last_voice_time=0.0
        self._noise_floor=0.0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def _run(self):
        try:
            _ensure_stt_deps()
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except Exception as e:
            print("Speech-to-text disabled:",e)
            return

        try:
            model_path=_ensure_vosk_model(self.model_dir)
        except Exception as e:
            print("Speech-to-text disabled:",e)
            return

        try:
            model=Model(model_path)
        except Exception as e:
            print("Speech-to-text disabled (model load failed):",e)
            return

        try:
            device_info=None
            if self.device is not None:
                try: device_info=sd.query_devices(self.device,"input")
                except Exception: device_info=None
            if self.samplerate is not None:
                samplerate=float(self.samplerate)
            else:
                try:
                    if device_info is None:
                        device_info=sd.query_devices(None,"input")
                    samplerate=float(device_info["default_samplerate"])
                except Exception:
                    samplerate=16000.0
        except Exception:
            samplerate=16000.0

        audio_q=queue.Queue()

        def callback(indata, frames, time_info, status):
            if self._stop.is_set():
                return
            try:
                b=bytes(indata)
                try:
                    a=array('h')
                    a.frombytes(b)
                    if a:
                        s=0
                        for x in a:
                            s += x if x >= 0 else -x
                        mean_abs = s / float(len(a))

                        if self._noise_floor <= 0:
                            self._noise_floor=float(mean_abs)
                        else:
                            self._noise_floor=(0.995*self._noise_floor)+(0.005*float(mean_abs))

                        thr=max(120.0, self._noise_floor*3.0)
                        if (not self._paused.is_set()) and mean_abs > thr:
                            self.last_voice_time=time.time()
                except Exception:
                    pass

                audio_q.put(b)
            except Exception:
                pass

        def new_recognizer():
            r=KaldiRecognizer(model,samplerate)
            try: r.SetWords(False)
            except Exception: pass
            return r

        rec=new_recognizer()
        last_partial=""

        try:
            with sd.RawInputStream(
                samplerate=samplerate,
                blocksize=self.blocksize,
                dtype="int16",
                channels=1,
                callback=callback,
                device=self.device,
            ):
                while not self._stop.is_set():
                    if self._paused.is_set():
                        try:
                            while True:
                                audio_q.get_nowait()
                        except queue.Empty:
                            pass
                        rec=new_recognizer()
                        last_partial=""
                        time.sleep(0.05)
                        continue

                    try:
                        data=audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if rec.AcceptWaveform(data):
                        try:
                            result=json.loads(rec.Result() or "{}")
                        except Exception:
                            result={}
                        text=(result.get("text") or "").strip()
                        if text:
                            self.events.put(("final",text))
                        last_partial=""
                    else:
                        try:
                            partial=json.loads(rec.PartialResult() or "{}").get("partial") or ""
                        except Exception:
                            partial=""
                        partial=partial.strip()
                        if partial and partial!=last_partial:
                            self.events.put(("partial",partial))
                            last_partial=partial
        except Exception as e:
            print("Speech-to-text error:",e)
            return

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
OUTDIR=Path("qwen-model")
HISTORY_LEN=3
MAX_TOKENS=256
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.1
system_prompt="Instructions: You are Jarvis, an assistant created by SFHS. Keep replies short. Be friendly and concise. Address the user by 'sir' in every sentence."
history=[]
loading_complete=threading.Event()

def progress_cb_print(m):
    try: sys.stdout.write(str(m)+"\n"); sys.stdout.flush()
    except: print(m)

class ModelManager:
    def __init__(self,name,outdir): self.name=name; self.outdir=outdir; self.model=None; self.tokenizer=None
    def model_exists(self):
        p=self.outdir
        if not p.is_dir(): return False
        if not (p/"config.json").exists(): return False
        weights=list(p.glob("pytorch_model*.bin"))+list(p.glob("*.safetensors"))+list(p.glob("pytorch_model-*.bin"))
        return len(weights)>0
    def download_model(self,progress_cb=None):
        """Download model artifacts into outdir.

        Uses huggingface_hub snapshot download for resumable, on-disk downloads.
        """
        self.outdir.mkdir(parents=True,exist_ok=True)
        if progress_cb: progress_cb("Downloading model snapshot (this may take a while)...")
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            # Fallback to Transformers download+save path
            if progress_cb: progress_cb("huggingface_hub not available; falling back to Transformers download...")
            tok=AutoTokenizer.from_pretrained(self.name,use_fast=True)
            model=AutoModelForCausalLM.from_pretrained(self.name,low_cpu_mem_usage=True)
            tok.save_pretrained(self.outdir); model.save_pretrained(self.outdir)
            if progress_cb: progress_cb("Download complete")
            return self.outdir

        # Download only the typical files we need.
        allow_patterns=[
            "*.safetensors",
            "*.bin",
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
            "chat_template.jinja",
            "*.model",
        ]
        snapshot_download(
            repo_id=self.name,
            local_dir=str(self.outdir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
        if progress_cb: progress_cb("Download complete")
        return self.outdir
    def load_model(self,progress_cb=None):
        device="cuda" if torch.cuda.is_available() else "cpu"
        if progress_cb: progress_cb(f"Loading tokenizer to {device}...")
        # Some tokenizers (e.g., Mistral) may require fix_mistral_regex=True.
        try:
            tokenizer=AutoTokenizer.from_pretrained(self.outdir,use_fast=True,fix_mistral_regex=True)
        except TypeError:
            tokenizer=AutoTokenizer.from_pretrained(self.outdir,use_fast=True)
        if device=="cuda":
            try:
                if progress_cb: progress_cb("Loading model with device_map=auto...")
                model=AutoModelForCausalLM.from_pretrained(self.outdir,device_map="auto",torch_dtype=torch.float16)
            except:
                if progress_cb: progress_cb("Fallback: loading model and moving to cuda...")
                model=AutoModelForCausalLM.from_pretrained(self.outdir); model.to("cuda")
        else:
            if progress_cb: progress_cb("Loading model to cpu...")
            model=AutoModelForCausalLM.from_pretrained(self.outdir); model.to("cpu")
        model.eval(); self.model=model; self.tokenizer=tokenizer
        if progress_cb: progress_cb("Model ready")
        return model,tokenizer

manager=ModelManager(MODEL_NAME,OUTDIR)

def build_prompt(system,history,user_msg):
    try:
        messages=[{"role":"system","content":system}]
        for u,a in history[-HISTORY_LEN:]:
            messages.append({"role":"user","content":u}); messages.append({"role":"assistant","content":a})
        messages.append({"role":"user","content":user_msg})
        return manager.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    except:
        parts=[f"SYSTEM: {system}"]
        for u,a in history[-HISTORY_LEN:]:
            parts.append(f"USER: {u}\nASSISTANT: {a}")
        parts.append(f"USER: {user_msg}\nASSISTANT:")
        return "\n".join(parts)

def safe_decode(gen,inputs_len):
    try: return manager.tokenizer.decode(gen,skip_special_tokens=True).strip()
    except: return ""

def lazy_load_model():
    try:
        manager.load_model(progress_cb=progress_cb_print)
        loading_complete.set()
        return
    except Exception as e:
        print("Failed to load model:",e)
        try:
            traceback.print_exc()
        except Exception:
            pass

    # If the local directory exists but is corrupt/partial, delete and re-download once.
    err_txt=str(e).lower() if 'e' in locals() else ""
    looks_corrupt=(
        "incomplete metadata" in err_txt
        or "file not fully covered" in err_txt
        or "error while deserializing header" in err_txt
        or "safetensors" in err_txt
        or "unexpected eof" in err_txt
    )
    if looks_corrupt and manager.outdir.is_dir():
        print("Local model appears corrupted/incomplete. Deleting and re-downloading...")
        try:
            shutil.rmtree(manager.outdir)
        except Exception as e2:
            print("Failed to delete model directory:",e2)
        try:
            manager.download_model(progress_cb=progress_cb_print)
            manager.load_model(progress_cb=progress_cb_print)
            loading_complete.set()
            return
        except Exception as e3:
            print("Re-download/load failed:",e3)
            try:
                traceback.print_exc()
            except Exception:
                pass

    loading_complete.set()

def download_and_load():
    try:
        manager.download_model(progress_cb=progress_cb_print)
        manager.load_model(progress_cb=progress_cb_print)
        loading_complete.set()
    except Exception as e:
        print("Download/load failed:",e)
        try:
            traceback.print_exc()
        except Exception:
            pass
        loading_complete.set()

def delete_model():
    try:
        if manager.model_exists():
            print("Deleting model files...")
            shutil.rmtree(manager.outdir)
            manager.model=None; manager.tokenizer=None
            print("Model deleted successfully. Type '/add' to download it again.")
        else:
            print("No model found to delete.")
    except Exception as e:
        print("Delete failed:",e)

def generate_and_stream(prompt,user_text):
    try:
        inputs=manager.tokenizer(prompt,return_tensors="pt")
        inputs={k:v.to(manager.model.device) for k,v in inputs.items()}
        streamer=None
        if TextIteratorStreamer is not None:
            try: streamer=TextIteratorStreamer(manager.tokenizer,skip_prompt=True,skip_special_tokens=True,timeout=60.0)
            except: streamer=None
        stopping=_make_stopping_criteria(GEN_STOP_EVENT)
        gen_kwargs=dict(max_new_tokens=MAX_TOKENS,do_sample=True,temperature=TEMPERATURE,top_p=TOP_P,top_k=TOP_K,repetition_penalty=REPETITION_PENALTY,pad_token_id=getattr(manager.tokenizer,"eos_token_id",None) or manager.tokenizer.pad_token_id)
        if stopping is not None:
            gen_kwargs["stopping_criteria"]=stopping
        reply=""
        if streamer:
            gen_thread=threading.Thread(target=manager.model.generate,kwargs={**inputs,**gen_kwargs,"streamer":streamer},daemon=True); gen_thread.start()
            speech_manager.text_buf=""
            collected=""
            print("\nJarvis: ",end="",flush=True); line_len=8
            buffer_for_next=""
            for new_text in streamer:
                if GEN_STOP_EVENT.is_set():
                    break
                if not new_text: continue
                collected+=new_text
                for ch in new_text:
                    print(ch,end="",flush=True); line_len+=1
                    if line_len>=80 and ch==' ':
                        print(); line_len=0
                    elif line_len>=100:
                        print(); line_len=0
                speech_manager.add_text(new_text)
                buffer_for_next+=new_text
                if len(buffer_for_next)>40 and not speech_manager.is_busy():
                    speech_manager.add_text("")  
                    buffer_for_next=""
            print()
            reply=collected.strip()
            speech_manager.flush()
            speech_manager.wait_until_done(timeout=120.0)
            gen_thread.join(timeout=2.0)
        else:
            out=manager.model.generate(**inputs,**gen_kwargs)
            gen=out[0][inputs["input_ids"].shape[1]:]
            reply=safe_decode(gen,inputs["input_ids"].shape[1])
            print("\nJarvis: ",end="")
            print(textwrap.fill(reply,width=80,subsequent_indent="        "))
            speech_manager.add_text(reply); speech_manager.flush(); speech_manager.wait_until_done(timeout=120.0)
        if reply.startswith(user_text):
            reply=reply[len(user_text):].lstrip(" :\n")
        history.append((user_text,reply))
        if len(history)>HISTORY_LEN: history.pop(0)
    except Exception as e:
        print("Error:",e)

if manager.model_exists():
    threading.Thread(target=lazy_load_model,daemon=True).start()
else:
    print("Model not found locally. Downloading and loading it now...")
    threading.Thread(target=download_and_load,daemon=True).start()


def _strip_wake_words(text: str) -> str:
    t=text.strip()
    t=re.sub(r"\bhey\s+jarvis\b","",t,flags=re.IGNORECASE).strip()
    t=re.sub(r"\bjarvis\b","",t,flags=re.IGNORECASE).strip()
    return re.sub(r"\s+"," ",t).strip()


def voice_loop():
    try:
        EXIT_EVENT.clear()
    except Exception:
        pass
    loading_complete.wait()
    if manager.model is None or manager.tokenizer is None:
        print("Model not loaded. Use /add in the code or ensure qwen-model exists.")
        return 1

    model_ready_file=os.environ.get("JARVIS_MODEL_READY_FILE") if ORCHESTRATED else None
    if model_ready_file:
        try:
            with open(model_ready_file,"w",encoding="utf-8") as f:
                f.write("ready\n")
        except Exception:
            pass

    shutdown_file=os.environ.get("JARVIS_SHUTDOWN_FILE") if ORCHESTRATED else None
    if shutdown_file:
        def _watch_shutdown():
            while True:
                if EXIT_EVENT.is_set():
                    return
                try:
                    if os.path.exists(shutdown_file):
                        EXIT_EVENT.set()
                        return
                except Exception:
                    return
                time.sleep(0.05)
        threading.Thread(target=_watch_shutdown,daemon=True).start()

    start_file=os.environ.get("JARVIS_START_FILE") if ORCHESTRATED else None
    if start_file:
        print("Waiting for orchestrator start signal...")
        while True:
            if EXIT_EVENT.is_set():
                return
            try:
                if os.path.exists(start_file):
                    break
            except Exception:
                break
            time.sleep(0.05)

    if not ORCHESTRATED:
        _start_mouse_control()
        _start_mouse_shutdown_watcher()

        if not _wait_for_mouse_ready(timeout=180.0):
            print("Mouse control failed to start.")
            _stop_mouse_control()
            return

        try:
            speech_manager.begin_utterance()
            speech_manager.add_text("Everything is ready to run")
            speech_manager.flush()
            speech_manager.wait_until_done(timeout=20.0)
        except Exception:
            pass

    stt=LiveSpeechToText(blocksize=2048)
    stt.start()

    WAKE_WORDS=("jarvis",)
    COMMAND_SILENCE_SEC=1.8
    COMMAND_MAX_SEC=12.0

    state="wake"
    cmd_final=""
    cmd_partial=""
    last_activity=0.0
    started_at=0.0
    last_text_time=0.0

    def _typing_process_phrase(phrase: str) -> bool:
        """Type phrase; if it contains the word 'enter', press Enter and finish."""
        p=re.sub(r"\s+"," ",phrase).strip()
        if not p:
            return False
        words=p.split()
        low_words=[w.lower().strip(".,!?") for w in words]
        if "enter" in low_words:
            i=low_words.index("enter")
            before=" ".join(words[:i]).strip()
            if before:
                print(f"[Typing] {before}")
                _type_text_windows(before)
            print("[Typing mode terminated]")
            _press_enter_windows()
            return True
        print(f"[Typing] {p}")
        _type_text_windows(p+" ")
        return False

    print("\nListening for wake word: 'Hey Jarvis' or 'Jarvis'...")

    def shutdown_now():
        try:
            GEN_STOP_EVENT.set()
        except Exception:
            pass
        try:
            speech_manager.interrupt()
        except Exception:
            pass
        try:
            stt.stop()
        except Exception:
            pass
        try:
            speech_manager.stop()
        except Exception:
            pass
        try:
            _stop_mouse_control()
        except Exception:
            pass
        raise SystemExit(0)

    try:
        while True:
            now=time.time()

            if EXIT_EVENT.is_set():
                shutdown_now()

            if state in ("command","typing"):
                last_voice=max(float(stt.last_voice_time or 0.0), float(last_text_time or 0.0), float(last_activity or 0.0))
                if (cmd_final or cmd_partial) and (now-last_voice) >= COMMAND_SILENCE_SEC:
                    user_text=(cmd_final or cmd_partial).strip()
                    cmd_final=""
                    cmd_partial=""
                    last_text_time=0.0

                    if state=="typing":
                        finished=_typing_process_phrase(user_text)
                        if finished:
                            state="wake"
                            print("\nListening for wake word: 'Hey Jarvis' or 'Jarvis'...")
                        continue

                    low_user=user_text.lower().strip()
                    if low_user=="type" or low_user.startswith("type "):
                        print("\n[Typing mode activated]")
                        remainder=user_text[4:].strip()
                        if remainder:
                            finished=_typing_process_phrase(remainder)
                            if finished:
                                state="wake"
                                print("\nListening for wake word: 'Hey Jarvis' or 'Jarvis'...")
                            else:
                                state="typing"
                        else:
                            state="typing"
                        continue

                    state="wake"
                    print(f"\nYou (speech): {user_text}")

                    speech_manager.begin_utterance()
                    GEN_STOP_EVENT.clear()
                    prompt=build_prompt(system_prompt,history,user_text)
                    gen_thread=threading.Thread(target=generate_and_stream,args=(prompt,user_text),daemon=True)
                    gen_thread.start()

                    stop_deadline=time.time()+180.0
                    while (gen_thread.is_alive() or speech_manager.is_busy()) and time.time() < stop_deadline:
                        if EXIT_EVENT.is_set():
                            shutdown_now()
                        try:
                            k2,t2=stt.events.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if not t2:
                            continue
                        low2=t2.lower()
                        if ("stop" in low2.split()) or ("stop jarvis" in low2) or ("jarvis stop" in low2):
                            print("\n[Stop detected]")
                            GEN_STOP_EVENT.set()
                            speech_manager.interrupt()
                            break
                        if ("exit" in low2.split()) or ("quit" in low2.split()):
                            print("\n[Exit detected]")
                            GEN_STOP_EVENT.set()
                            speech_manager.interrupt()
                            shutdown_now()

                    try: gen_thread.join(timeout=0.5)
                    except Exception: pass
                    speech_manager.flush()
                    speech_manager.wait_until_done(timeout=2.0)
                    print("\nListening for wake word: 'Hey Jarvis' or 'Jarvis'...")

                elif state=="command" and (now-started_at) >= COMMAND_MAX_SEC:
                    cmd_final=""
                    cmd_partial=""
                    state="wake"
                    print("\nListening for wake word: 'Hey Jarvis' or 'Jarvis'...")

            try:
                kind,text=stt.events.get(timeout=0.05)
            except queue.Empty:
                continue

            if not text:
                continue
            low=text.lower()

            # Global exit commands
            if ("exit" in low.split()) or ("quit" in low.split()):
                if state=="wake":
                    print("\n[Exit detected]")
                    shutdown_now()
                else:
                    cleaned=_strip_wake_words(text).lower().strip()
                    if cleaned in ("exit","quit"):
                        print("\n[Exit detected]")
                        shutdown_now()

            if state=="wake":
                if any(w in low.split() for w in WAKE_WORDS) or ("jarvis" in low):
                    remainder=_strip_wake_words(text)
                    state="command"
                    started_at=time.time()
                    last_activity=time.time()
                    last_text_time=time.time()
                    cmd_final=""
                    cmd_partial=remainder
                    print("\n[Wake word detected]")
            else:
                last_activity=time.time()
                if state=="typing":
                    cleaned=re.sub(r"\s+"," ",text).strip()
                else:
                    cleaned=_strip_wake_words(text)
                if kind=="partial":
                    if cleaned:
                        cmd_partial=cleaned
                        last_text_time=time.time()
                elif kind=="final":
                    if cleaned:
                        cmd_final=(cmd_final+" "+cleaned).strip() if cmd_final else cleaned
                        last_text_time=time.time()
                    cmd_partial=""
    except KeyboardInterrupt:
        print("\nShutting down...")
        shutdown_now()
    finally:
        try: stt.stop()
        except Exception: pass
        try: speech_manager.stop()
        except Exception: pass
        try: _stop_mouse_control()
        except Exception: pass

    return 0


if __name__=="__main__":
    raise SystemExit(int(voice_loop() or 0))
