"""Original shared/camera/acquisition.py definitions.

Loaded into tracker globals by the explicit temporary compatibility binder.
The source bodies are unchanged; see docs/IMPORT_PATH_MIGRATION.md.
"""

def dahua_url(cam):
    """RTSP URL of a Dahua camera (subtype 0 = main stream, 1 = sub stream)."""
    auth = f"{quote(cam['user'], safe='')}:{quote(cam['password'], safe='')}@" if cam.get("user") else ""
    return (f"rtsp://{auth}{cam['ip']}:{RTSP_PORT}/cam/realmonitor"
            f"?channel={int(cam.get('channel', 1))}&subtype={int(cam.get('subtype', 0))}")


def local_ipv4s():
    try:
        return {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        return set()


def unreachable_hint(host, port):
    """None if host:port accepts a TCP connection, else a message on what to check."""
    try:
        socket.create_connection((host, port), timeout=2).close()
        return None
    except OSError as e:
        reason = str(e)
    subnet = ".".join(host.split(".")[:3])
    if host.count(".") == 3 and not any(ip.startswith(subnet + ".") for ip in local_ipv4s()):
        return (f"Camera {host} not reachable: this PC has no address in {subnet}.x. Plug the cable into the "
                f"Ethernet port, and give that port a static IP like {subnet}.10 / 255.255.255.0 "
                f"(or change the camera IP to this PC's subnet).")
    others = sorted(ip for ip in local_ipv4s() if not ip.startswith(subnet + ".") and not ip.startswith("127."))
    direct = (f" If it is cabled straight into this PC: Windows sends {subnet}.x through the adapter that already "
              f"has a {subnet}.x address (Wi-Fi), not the cable; this PC's other address(es): {', '.join(others)}. "
              f"Give the camera an address in that subnet, or the Ethernet port a {subnet}.x address."
              if others else "")
    return (f"Camera {host}:{port} not reachable ({reason}). Check the cable, the camera IP and that it is "
            f"powered.{direct}")


def af_flicker_note(denom, mains_hz):
    """(exposure ms, flicker-safe?): safe when the shutter is ~a whole multiple of the light's half-period."""
    ms = 1000.0 / denom
    half = 1000.0 / (2 * mains_hz)
    frac = (ms % half) / half
    return ms, ms >= 0.8 * half and (frac < 0.2 or frac > 0.8)  # shorter than a half-period catches part of it


def apply_usb_antiflicker(cap, af, state, index=0):
    """Push the anti-flicker setting onto an open USB camera (DirectShow). Returns a read-back message.
    `state` keeps the camera's own brightness so switching off restores it."""
    if sys.platform.startswith("linux"):
        return apply_linux_antiflicker(cap, af, state, index)
    if not af.get("enabled"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto exposure (DSHOW/UVC)
        if state.get("brightness") is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, state["brightness"])
        return "Anti-flicker off: camera back on auto exposure"
    if state.get("brightness") is None:
        state["brightness"] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    else:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, state["brightness"])  # start from the camera's own brightness
    denom = max(AF_DENOM_RANGE[0], min(AF_DENOM_RANGE[1], int(af.get("shutter_denom", 100))))
    hz = int(af.get("mains_hz", 50))

    def settle():  # several UVC drivers ignore controls until samples flow; a read-back right after set() is stale
        for _ in range(5):
            cap.grab()
        time.sleep(0.12)

    def manual_mode():
        for val in (0.25, 1.0, 0.0, 3.0):  # the manual-exposure code differs between drivers
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            time.sleep(0.12)
            rb = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            if rb != -1 and abs(rb - val) < 0.5:
                return True
        return False

    settle()
    manual = manual_mode()
    want = math.log2(1.0 / denom)
    ok = cap.set(cv2.CAP_PROP_EXPOSURE, want)
    time.sleep(0.12)
    rb = cap.get(cv2.CAP_PROP_EXPOSURE)
    scale = "log2"
    if ok and rb != -1 and abs(rb - want) >= 0.3 and abs(rb - round(rb)) < 0.05:
        # the camera only takes whole stops (asked 1/100 s = -6.64, it used -6 = 15.6 ms): use the whole stop
        # closest to the asked shutter that is flicker-safe at this mains frequency instead
        safe = [st for st in range(-13, 0) if af_flicker_note(2.0 ** -st, hz)[1]]
        if safe:
            st = min(safe, key=lambda v: abs(v - want))
            ok2 = cap.set(cv2.CAP_PROP_EXPOSURE, float(st))
            time.sleep(0.12)
            rb2 = cap.get(cv2.CAP_PROP_EXPOSURE)
            if ok2 and rb2 != -1:
                want, ok, rb, scale = float(st), ok2, rb2, "nearest flicker-safe whole stop"
    if not ok or rb == -1 or abs(rb - want) > 1.5:  # absolute 0.1 ms units (MSMF-style drivers)
        manual = manual_mode() or manual
        absolute = max(1, int(round(10000 / denom)))
        ok3 = cap.set(cv2.CAP_PROP_EXPOSURE, float(absolute))
        time.sleep(0.12)
        rb3 = cap.get(cv2.CAP_PROP_EXPOSURE)
        if ok3 and rb3 != -1 and abs(rb3 - absolute) <= max(2.0, absolute * 0.5):
            want, ok, rb, scale = float(absolute), ok3, rb3, "absolute"
    settle()
    took = bool(ok) and rb != -1
    # the exposure the camera really uses (read back), not the one asked for
    real_ms = 1000.0 * 2.0 ** rb if took and scale.startswith(("log2", "nearest")) else (
        rb / 10.0 if took else 1000.0 / denom)
    safe = af_flicker_note(1000.0 / real_ms, hz)[1] if real_ms > 0 else False
    # keep the white bench from burning out: lower the brightness until little of the frame is clipped
    clipped, bright = clipped_share(cap), cap.get(cv2.CAP_PROP_BRIGHTNESS)
    for b in AF_BRIGHTNESS_STEPS:
        if clipped <= AF_MAX_CLIPPED or b >= bright:
            if clipped <= AF_MAX_CLIPPED:
                break
            continue
        cap.set(cv2.CAP_PROP_BRIGHTNESS, b)
        clipped, bright = clipped_share(cap), cap.get(cv2.CAP_PROP_BRIGHTNESS)
    return (f"Anti-flicker {'applied' if took else 'NOT accepted by the camera'}: asked 1/{denom} s, camera uses "
            f"{real_ms:.1f} ms ({'flicker-safe' if safe else 'may band'} at {hz} Hz; {scale} {want:.2f}, read-back "
            f"{rb:.2f}) | brightness {bright:.0f}, {clipped:.0%} of the frame burnt white"
            f"{'' if manual else ' | manual mode unconfirmed'}")


def clipped_share(cap):
    """Share of pure-white pixels once the camera has settled after a change."""
    t = time.time()
    while time.time() - t < AF_SETTLE_S:
        cap.grab()
    shares = []
    for _ in range(3):
        ok, f = cap.read()
        if ok:
            g = cv2.cvtColor(f[::4, ::4], cv2.COLOR_BGR2GRAY)
            shares.append(float(np.mean(g >= 250)))
    return float(np.median(shares)) if shares else 0.0


class FlickerMeter:
    """Measures flicker on the live feed (~1 ms per frame): banding = rows brighter/darker than their running
    average (rolling shutter under AC light), pulsing = whole-frame brightness changing between frames."""

    def __init__(self, n=30):
        self.n = n
        self.reset()

    def reset(self):
        self.ref = None
        self.band = deque(maxlen=self.n)
        self.level = deque(maxlen=self.n)

    def add(self, frame):
        g = cv2.cvtColor(frame[::4, ::16], cv2.COLOR_BGR2GRAY).astype(np.float32)
        prof = np.median(g, axis=1) + 1.0
        self.level.append(float(g.mean()))
        if self.ref is not None and self.ref.shape == prof.shape:
            self.band.append(float(np.std(prof / self.ref)))
            self.ref = 0.9 * self.ref + 0.1 * prof
        else:
            self.ref = prof

    def value(self):
        """(banding %, pulsing %, brightness) over the last frames, or None while too few."""
        if len(self.band) < 10:
            return None
        lv = np.array(self.level)
        return float(np.median(self.band)) * 100, float(lv.std() / max(lv.mean(), 1e-6)) * 100, float(lv.mean())


def open_usb_camera(index, resolution=None):
    """A USB camera at the asked resolution: DirectShow with MJPG (Windows otherwise opens it at a low default
    resolution, or uncompressed at a few fps)."""
    linux = sys.platform.startswith("linux")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if linux else cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if resolution:
        w, h = (int(v) for v in str(resolution).lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if linux:
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


class RTSPGrabber(threading.Thread):
    """Always keeps only the newest frame, so processing never lags behind the stream."""

    def __init__(self, source, camera=None):
        super().__init__(daemon=True)
        self.source = source      # RTSP URL, or the device number of a USB camera
        self.camera = camera or {}
        self.af = self.camera.get("antiflicker", dict(AF_DEFAULT) if sys.platform.startswith("linux") else None)
        # USB anti-flicker is applied on every (re)connect, before publishing frames.
        self.af_dirty = True
        self.af_state = {}        # the camera's own brightness, restored when anti-flicker is switched off
        self.af_busy_until = 0.0  # applying anti-flicker pauses the frames for a few seconds: not a lost stream
        self.on_af = None         # called with the read-back message after applying
        self.on_frame = None      # non-blocking evidence recorder, independent of inference
        self.frame, self.ts = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.problem = ""  # why the camera can't be reached, shown in the status line

    def run(self):
        while self.running:
            if not isinstance(self.source, int):
                # quick TCP check first: opening an unreachable RTSP URL can hang for ~30 s
                url = urlparse(self.source)
                if url.hostname:
                    self.problem = unreachable_hint(url.hostname, url.port or RTSP_PORT) or ""
                    if self.problem:
                        time.sleep(2)
                        continue
            if isinstance(self.source, int):
                cap = open_usb_camera(self.source, self.camera.get("resolution"))
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                self.problem = (f"USB camera {self.source} not found: check the cable or try another camera number"
                                if isinstance(self.source, int) else
                                "Camera reachable but the stream did not open: check user, password and channel")
                time.sleep(2)
                continue
            self.connected, self.problem = True, ""
            self.af_dirty = True
            while self.running:
                if self.af_dirty and isinstance(self.source, int) and self.af is not None:
                    self.af_dirty = False  # on this thread: it owns the camera handle
                    self.af_busy_until = time.time() + 30
                    try:
                        msg = apply_usb_antiflicker(cap, self.af, self.af_state, self.source)
                    except Exception as e:
                        msg = f"Anti-flicker could not be applied: {e}"
                    self.af_busy_until = time.time() + 2.0
                    if self.on_af:
                        self.on_af(msg)
                ok, f = cap.read()
                if not ok:
                    break
                captured_at = time.time()
                if self.on_frame is not None:
                    self.on_frame(f, captured_at)
                with self.lock:
                    self.frame, self.ts = f, captured_at
            cap.release()
            self.connected = False
            time.sleep(1)  # then reconnect

    def latest(self, after_ts=None, copy_frame=True):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            if self.ts == after_ts:
                return None, self.ts
            return (self.frame.copy() if copy_frame else self.frame), self.ts

    def request_antiflicker(self, af):
        self.af, self.af_dirty = af, True

    def stop(self):
        self.running = False

