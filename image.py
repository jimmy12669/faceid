"""Camera capture and the on-screen prompts people actually read.

This is the file the prototype started as. It still opens the webcam and still
answers to the same keys -- 'c' to enrol, 'v' to verify, 'q' to quit -- but the
frame no longer flows straight into a comparison. It flows into the capture
pipeline in auth.py, and this module's job is narrowed to two things: handing
over frames, and telling the person in front of the lens what to do next.

That second job matters more than it looks. A liveness challenge nobody
understands is just a system that rejects honest users, and a security control
people route around is worse than one you never shipped.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

import config
from liveness import LivenessState

WINDOW = "faceid"

# BGR. Deliberately muted -- a full-saturation red rectangle over someone's
# face reads as an alarm even when nothing is wrong.
INK = (245, 245, 245)
MUTED = (170, 170, 170)
GOOD = (120, 200, 120)
WARN = (90, 180, 235)
BAD = (95, 95, 225)
PANEL = (32, 30, 28)


class CameraError(Exception):
    pass


class Camera:
    """A webcam as a frame source. Use it as a context manager."""

    def __init__(self, index: int = 0, *, width: int = 1280, height: int = 720,
                 warmup_frames: int = 5):
        self.index = index
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            raise CameraError(
                f"could not open camera {self.index}. "
                "Is another application using it, or is the device permission denied?"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        # The first frames off most webcams are dark while exposure settles,
        # and dark frames fail the quality gate for reasons that aren't the
        # user's fault.
        for _ in range(self.warmup_frames):
            capture.read()
        return self

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise CameraError("camera is not open")
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


class ReplaySource:
    """Frames from a list. Handy for demos and for testing without a camera."""

    def __init__(self, frames: list[np.ndarray], *, loop: bool = True):
        if not frames:
            raise ValueError("need at least one frame")
        self.frames = frames
        self.loop = loop
        self._i = 0

    def read(self) -> np.ndarray | None:
        if self._i >= len(self.frames):
            if not self.loop:
                return None
            self._i = 0
        frame = self.frames[self._i]
        self._i += 1
        return frame.copy()


# -- drawing ---------------------------------------------------------------


def _text(frame, text, origin, *, colour=INK, scale=0.6, weight=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight,
                cv2.LINE_AA)


def _panel(frame, top_left, bottom_right, *, alpha=0.62):
    """Translucent backing so text stays readable over a bright room."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, PANEL, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _corner_box(frame, box, colour, *, thickness=2, arm=28):
    """Corner brackets rather than a full rectangle -- less visual noise."""
    x, y, w, h = box
    for cx, sx in ((x, 1), (x + w, -1)):
        for cy, sy in ((y, 1), (y + h, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * arm, cy), colour, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * arm), colour, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, update, *, title: str = "") -> np.ndarray:
    """Render one frame of feedback. Returns a new image; never mutates input."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    progress = update.progress

    colour = MUTED
    if progress is not None:
        colour = {
            LivenessState.CALIBRATING: WARN,
            LivenessState.RUNNING: WARN,
            LivenessState.PASSED: GOOD,
            LivenessState.FAILED: BAD,
        }[progress.state]
    if update.hint:
        colour = WARN

    if update.face is not None:
        _corner_box(canvas, update.face.box, colour)
        for px, py in update.face.landmarks.astype(int):
            cv2.circle(canvas, (int(px), int(py)), 2, colour, -1, cv2.LINE_AA)

    # Top bar: what we're doing.
    _panel(canvas, (0, 0), (w, 46))
    _text(canvas, title or "faceid", (16, 30), scale=0.7, weight=2)
    if update.needed > 1:
        _text(canvas, "samples", (w - 250, 30), colour=MUTED, scale=0.55)
        for i in range(update.needed):
            cx = w - 170 + i * 22
            if i < update.samples:
                cv2.circle(canvas, (cx, 24), 7, GOOD, cv2.FILLED, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (cx, 24), 7, MUTED, 1, cv2.LINE_AA)

    # Bottom bar: what the person should do.
    _panel(canvas, (0, h - 96), (w, h))
    if update.hint:
        _text(canvas, update.hint, (16, h - 60), colour=WARN, scale=0.66, weight=2)
    elif progress is not None and progress.prompt:
        _text(canvas, progress.prompt, (16, h - 60), colour=INK, scale=0.72, weight=2)
    elif progress is not None and progress.message:
        _text(canvas, progress.message, (16, h - 60), colour=colour, scale=0.7, weight=2)

    if progress is not None:
        step = ""
        if progress.total:
            step = f"step {min(progress.index, progress.total)}/{progress.total}"
        line = "  ".join(x for x in (step, progress.message if progress.prompt else "") if x)
        _text(canvas, line, (16, h - 30), colour=MUTED, scale=0.55)
        if progress.pad is not None:
            _meter(canvas, progress.pad.score, (w - 190, h - 44))

    _text(canvas, "q quit", (w - 90, h - 12), colour=MUTED, scale=0.45)
    return canvas


def _meter(frame, value: float, origin: tuple[int, int], *, width: int = 150, height: int = 8):
    """A small bar for the liveness confidence, so it isn't a black box."""
    x, y = origin
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 70), cv2.FILLED)
    filled = int(width * max(0.0, min(1.0, value)))
    colour = GOOD if value > 0.7 else WARN if value > 0.45 else BAD
    cv2.rectangle(frame, (x, y), (x + filled, y + height), colour, cv2.FILLED)
    _text(frame, "live", (x - 42, y + height), colour=MUTED, scale=0.45)


# -- front ends ------------------------------------------------------------


class WindowUI:
    """Draws to an OpenCV window. Needs a desktop session."""

    def __init__(self, *, mirror: bool = True, title: str = "faceid"):
        self.mirror = mirror
        self.title = title
        self.last_key = -1
        _require_gui()

    def __call__(self, update) -> None:
        canvas = draw_overlay(update.frame, update, title=self.title)
        if self.mirror:
            # Only the preview is flipped. Liveness runs on the true frame, so
            # 'turn left' still means the user's left.
            canvas = cv2.flip(canvas, 1)
        cv2.imshow(WINDOW, canvas)
        self.last_key = cv2.waitKey(1) & 0xFF

    @staticmethod
    def close() -> None:
        cv2.destroyAllWindows()


class ConsoleUI:
    """Prints prompts to the terminal. For headless boxes and SSH sessions."""

    def __init__(self, stream=sys.stdout):
        self.stream = stream
        self._last = ""

    def __call__(self, update) -> None:
        progress = update.progress
        line = update.hint or (progress.prompt if progress else "") or (
            progress.message if progress else ""
        )
        if update.needed > 1 and line:
            line = f"[{update.samples}/{update.needed}] {line}"
        if line and line != self._last:
            print(f"  {line}", file=self.stream, flush=True)
            self._last = line

    @staticmethod
    def close() -> None:
        return None


def _require_gui() -> None:
    """Headless wheels still *have* imshow -- it just throws. So try it."""
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow(WINDOW)
    except cv2.error as exc:
        raise CameraError(
            "this OpenCV build has no GUI support (it is the headless wheel, or "
            "there is no display attached).\n"
            "Install opencv-contrib-python instead of the -headless variant, "
            "or run with --headless for terminal prompts."
        ) from exc


def select_ui(headless: bool = False, *, mirror: bool = True, title: str = "faceid"):
    if headless:
        return ConsoleUI()
    try:
        return WindowUI(mirror=mirror, title=title)
    except CameraError as exc:
        print(f"note: {exc}\nfalling back to terminal prompts.\n", file=sys.stderr)
        return ConsoleUI()


def main(argv: list[str] | None = None) -> int:
    """The original interactive loop, wired to the hardened pipeline.

    Kept because muscle memory is real: same window, same keys. The camera is
    opened once here and handed to the authenticator -- the earlier version of
    this function shelled out to the CLI, which then tried to open the same
    device a second time and failed.
    """
    import auth
    import cli

    settings = config.load()
    try:
        store = cli._open_store(settings)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        engine = cli._engine(settings)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        store.close()
        return 1

    authenticator = auth.Authenticator(store, engine, settings, actor="image.py")
    ui = select_ui(headless=False)
    print("faceid -- 'c' enrol, 'v' verify, 'q' quit")

    try:
        with Camera(0) as camera:
            while True:
                frame = camera.read()
                if frame is None:
                    print("camera stopped returning frames", file=sys.stderr)
                    return 1
                cv2.imshow(WINDOW, cv2.flip(frame, 1))
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return 0
                if key in (ord("c"), ord("e")):
                    name = input("name to enrol: ").strip()
                    if not name:
                        continue
                    result = authenticator.enroll(name, camera, on_progress=ui)
                    print(result.message)
                    if result.ok and result.detail.get("totp_secret"):
                        print(f"  TOTP secret : {result.detail['totp_secret']}")
                        print("  recovery    : "
                              + " ".join(result.detail["recovery_codes"][:3]) + " ...")
                        print("  (run cli.py enroll for the full list, shown once)")
                elif key == ord("v"):
                    result = authenticator.authenticate(
                        camera, on_progress=ui,
                        second_factor=lambda prompt: input(f"{prompt}: ").strip(),
                    )
                    print(result.message)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        ui.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
