"""Transport-independent physical/virtual button protocol."""

from dataclasses import dataclass
import uuid


@dataclass(frozen=True)
class HardwareCommand:
    name: str
    confirmed: bool = False
    command_id: str = ""

    def payload(self):
        return dict(
            name=self.name,
            confirmed=self.confirmed,
            command_id=self.command_id or uuid.uuid4().hex,
        )


class HardwareAdapter:
    def snapshot(self):
        raise NotImplementedError

    def submit(self, command):
        raise NotImplementedError
