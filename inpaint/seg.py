"""The models of Image Studio and the segmentation worker, on the board's server's side (owner 2026-10-10: «давай качай конечно»).

SAM 3 and BiRefNet need mlx and torch, which the board's Python has not: they run in segworker.py under their own Python
(~/Library/Caches/Hyimg/venvs/seg, HYIMG_SEG_PYTHON), started on the first request and kept warm; it leaves by itself after 10 min
without requests or when this server goes. One request at a time; the pictures go through a temporary folder as PNG.

  call(req, timeout)      the worker's answer (a dict); RuntimeError with the worker's words
  missing(*ids)           the first model or the runtime not installed, with the words to say (README «Models»), else None
  status()                every model: installed, size, where, licence, what uses it; the runtime too (Settings › Image Studio)
  remove(id)              a model's folder to nothing (asked first in Settings); the worker stops first

HYIMG_MODELS moves the models' folder (tests), HYIMG_SEG_STUB=1 makes the worker answer without models (segworker.py, tests).
"""
import json
import os
import shutil
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/Library/Caches/Hyimg")
HINT = "README «Models» in hyimg-image-studio: the download is asked first"
MODELS = {   # what Settings shows and Remove may delete: only these folders, only under the models' folder
    "lama": {"name": "LaMa", "for": "Content-Aware Fill, Spot Healing", "file": "lama_fp32.onnx", "licence": "Apache-2.0",
             "source": "https://huggingface.co/Carve/LaMa-ONNX"},
    "sam3-4bit": {"name": "SAM 3 (MLX, 4-bit)", "for": "Select by Text, Smart Select", "file": "model.safetensors",
                  "licence": "SAM License (Meta), LICENSE-SAM.txt beside the weights", "source": "https://huggingface.co/mlx-community/sam3-4bit"},
    "birefnet-lite": {"name": "BiRefNet lite", "for": "Refine Edge, Remove Background", "file": "model.safetensors", "licence": "MIT",
                      "source": "https://huggingface.co/ZhengPeng7/BiRefNet_lite"},
}


def models_dir():
    return os.path.expanduser(os.environ.get("HYIMG_MODELS") or os.path.join(CACHE, "models"))


def python():
    return os.path.expanduser(os.environ.get("HYIMG_SEG_PYTHON") or os.path.join(CACHE, "venvs", "seg", "bin", "python"))


def stub():
    return os.environ.get("HYIMG_SEG_STUB") == "1"


def installed(mid):
    return os.path.isfile(os.path.join(models_dir(), mid, MODELS[mid]["file"]))


def missing(*ids):
    if stub():
        return None
    for mid in ids:
        if not installed(mid):
            return {"error": f"{MODELS[mid]['name']} is not installed ({os.path.join(models_dir(), mid)}): {HINT}", "missing": mid}
    if not os.path.isfile(python()):
        return {"error": f"The segmentation runtime is not installed ({python()}): {HINT}", "missing": "runtime"}
    return None


def _size(path):
    n = 0
    for d, _, files in os.walk(path):
        for f in files:
            try: n += os.lstat(os.path.join(d, f)).st_size
            except OSError: pass
    return n


def status():
    out = []
    for mid, m in MODELS.items():
        p = os.path.join(models_dir(), mid)
        out.append({"id": mid, "name": m["name"], "for": m["for"], "licence": m["licence"], "source": m["source"], "path": p,
                    "installed": installed(mid), "bytes": _size(p) if os.path.isdir(p) else 0})
    venv = os.path.dirname(os.path.dirname(python()))
    return {"models": out, "runtime": {"path": venv, "installed": os.path.isfile(python()), "bytes": _size(venv) if os.path.isdir(venv) else 0},
            "worker": WORKER.alive(), "hint": HINT}


def remove(mid):
    if mid not in MODELS:
        raise ValueError(f"not a model of Image Studio: {mid!r}")
    root = os.path.realpath(models_dir()); p = os.path.realpath(os.path.join(root, mid))
    if not p.startswith(root + os.sep) or os.path.islink(os.path.join(root, mid)):
        raise ValueError(f"not inside the models' folder: {p}")
    WORKER.stop()
    freed = _size(p) if os.path.isdir(p) else 0
    if os.path.isdir(p):
        shutil.rmtree(p)
    if mid == "lama":   # the LaMa session the server keeps (routes.py loads lama.py once)
        lama = sys.modules.get("hyimg_frames_lama")
        if lama is not None: lama._sess = None
    return {"removed": mid, "freed": freed}


class Worker:
    def __init__(self):
        self.p, self.lock = None, threading.Lock()

    def alive(self):
        return bool(self.p and self.p.poll() is None)

    def stop(self):
        with self.lock:
            if self.alive():
                try: self.p.stdin.close(); self.p.wait(5)
                except (OSError, subprocess.TimeoutExpired): self.p.kill()
            self.p = None

    def _start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1", HYIMG_MODELS=models_dir())
        log = open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "hyimg-segworker.log"), "ab")
        # niced: the board stays smooth while a model works; -I: none of the server's paths or user packages leak in
        self.p = subprocess.Popen(["nice", "-n", "5", python(), "-I", os.path.join(HERE, "segworker.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=log, env=env, text=True, bufsize=1)

    def call(self, req, timeout=300):
        with self.lock:
            for attempt in (0, 1):   # a worker that left (idle, or crashed) is started again once
                if not self.alive():
                    self._start()
                try:
                    self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
                except (BrokenPipeError, OSError):
                    self.p = None
                    continue
                line = _readline(self.p, timeout)
                if line:
                    res = json.loads(line)
                    if res.get("error"): raise RuntimeError(res["error"])
                    return res
                self.p.kill(); self.p = None
                if attempt: break
            raise RuntimeError("the segmentation worker gave no answer (its log: $TMPDIR/hyimg-segworker.log)")


def _readline(p, timeout):
    import select
    r, _, _ = select.select([p.stdout], [], [], timeout)
    return p.stdout.readline() if r else ""


WORKER = Worker()


def call(req, timeout=300):
    return WORKER.call(req, timeout)
