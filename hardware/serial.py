"""Future serial transport boundary. No unvalidated hardware protocol is simulated."""

from hardware.interface import HardwareAdapter


class SerialAdapter(HardwareAdapter):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Physical serial adapter is not commissioned. Use --hardware virtual; no serial hardware has been validated."
        )
