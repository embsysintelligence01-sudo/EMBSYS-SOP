import json
from urllib.request import Request, urlopen
from hardware.interface import HardwareAdapter


class VirtualAdapter(HardwareAdapter):
    def __init__(self, url="http://127.0.0.1:8765", token=""):
        self.url = url.rstrip("/")
        self.token = token

    def request(self, path, payload=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(
            self.url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers=headers,
        )
        with urlopen(request, timeout=3) as response:
            return json.load(response)

    def snapshot(self):
        return self.request("/api/hardware")

    def submit(self, command):
        return self.request("/api/commands", command.payload())
