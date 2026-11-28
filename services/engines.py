"""Bounded engine subprocess transport. Backend imports no Qt or model libraries."""

from pathlib import Path
import json
import os
import subprocess
import threading
from app.config import ROOT, interpreter, engine_environment
from core.store import uid


class EngineProcess:
    def __init__(self, controller, activity):
        self.controller = controller
        self.session_id = activity["session_id"]
        self.process = None
        self.write_lock = threading.Lock()
        self.retiring = False
        self.closed = False
        self.activity = activity

    def send(self, message):
        if self.process is None:
            engine = self.activity["engine"]
            mode = "releasing" if engine == "releasing" else "packet"
            env = engine_environment(mode)
            env.update(
                OMP_NUM_THREADS=str(self.controller.settings.cpu_threads),
                MKL_NUM_THREADS=str(self.controller.settings.cpu_threads),
                PYTHONUNBUFFERED="1",
            )
            env["EMBSYS_APP_MODE"] = self.controller.settings.mode
            if self.controller.settings.mode == "production":
                env.pop("DISPLAY", None)
                env.pop("WAYLAND_DISPLAY", None)
                env.pop("QT_QPA_PLATFORM", None)
            env["EMBSYS_RUNTIME"] = str(self.controller.settings.runtime)
            env["EMBSYS_REPLAY_SOURCE"] = self.controller.settings.replay_source
            self.process = subprocess.Popen(
                [
                    str(interpreter(mode)),
                    str(ROOT / "services/engine_host.py"),
                    "--family",
                    "releasing" if engine == "releasing" else "pledge",
                    "--mode",
                    self.controller.settings.mode,
                ],
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            threading.Thread(
                target=self._read, name="engine-" + self.session_id, daemon=True
            ).start()
        with self.write_lock:
            if self.process.poll() is not None:
                raise RuntimeError("Engine host is not running")
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def _read(self):
        path = (
            self.controller.settings.runtime
            / "logs"
            / (self.session_id + "-" + uid("host") + ".log")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            for line in self.process.stdout:
                stream.write(line)
                stream.flush()
                if "@SOP@" in line:
                    try:
                        message = json.loads(line.split("@SOP@", 1)[1])
                    except ValueError:
                        continue
                    self.controller.receive(self, message)
        self.process.wait()
        self.controller.receive(
            self, dict(event="host_exit", exit_code=self.process.returncode)
        )

    def close(self):
        self.closed = True
        if self.process and self.process.poll() is None:
            try:
                self.send(dict(command="EXIT"))
            except (OSError, RuntimeError):
                pass
