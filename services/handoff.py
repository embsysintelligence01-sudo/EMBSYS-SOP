"""UI-independent two-START adapter for the existing PacketStage handoff rules."""

import copy
import threading
from pathlib import Path
from services.evidence import publish


class SurveillanceHandoff:
    def __init__(self, worker, directory):
        from engines.purity.packet_handoff import camera_released, testing_snapshot

        if not camera_released(worker):
            raise RuntimeError("Purity camera-release barrier has not completed")
        self.worker = worker
        self.lock = threading.RLock()
        self.ready = threading.Event()
        monitor = getattr(worker, "continuous_monitor", None)
        source = getattr(monitor, "directory", None) or directory
        self.testing = testing_snapshot(worker, source)
        worker.packet_identity_statuses = {
            j["id"]: j["identity_status"] for j in self.testing["jewels"]
        }
        self.identity_at_handoff = dict(worker.packet_identity_statuses)
        self.directory = Path(directory)
        self.persist()

    def persist(self):
        # Keep evidence plus identity features/positions separately; no executable pickle.
        from services.identity import identity_snapshot

        value = dict(testing=self.testing, identity=identity_snapshot(self.worker))
        publish(
            self.directory
            / (
                "identity-final.json"
                if self.ready.is_set()
                else "identity-handoff.json"
            ),
            value,
        )
        publish(self.directory / "identity.json", value)

    def finish(self):
        from engines.purity.packet_handoff import testing_snapshot

        with self.lock:
            value = testing_snapshot(self.worker, self.testing["directory"], final=True)
            for jewel in value["jewels"]:
                jewel["identity_status"] = self.identity_at_handoff.get(
                    jewel["id"], "UNASSIGNED"
                )
            value["identity_status"] = self.testing["identity_status"]
            self.testing.update(value)
            from services.identity import identity_snapshot

            final = dict(testing=self.testing, identity=identity_snapshot(self.worker))
            publish(self.directory / "identity-final.json", final)
            publish(self.directory / "identity.json", final)
        self.ready.set()

    def payload(self):
        return dict(
            worker=self.worker,
            testing=self.testing,
            testing_ready=self.ready,
            testing_lock=self.lock,
        )


def recovered_handoff(path, source, camera, roi):
    """Retain historical IDs as uncertain; a restart cannot restore live ownership."""
    import json
    from types import SimpleNamespace

    if path is not None and Path(path).is_file():
        data = json.loads(Path(path).read_text())
        testing = copy.deepcopy(data["testing"])
    else:
        testing = dict(
            directory=str(Path(path).parent) if path else "",
            jewels=[],
            evidence_status="INCOMPLETE",
            identity_status="UNASSIGNED",
            workflow_stage="PACKET_SEALING",
        )
    testing["analysis_status"] = "INTERRUPTED"
    testing["evidence_status"] = "REVIEW_REQUIRED"
    testing["identity_status"] = (
        "REVIEW_REQUIRED" if testing.get("jewels") else "UNASSIGNED"
    )
    for jewel in testing.get("jewels", []):
        jewel["identity_status"] = "UNCERTAIN"
    testing["recovery_note"] = (
        "Previous live identity owner was lost. Available historical identities retained; new recording remains unassigned."
    )
    worker = SimpleNamespace(
        tracks=[],
        id_model=None,
        identity_extractor=None,
        frame_wh=None,
        model=None,
        camera=camera,
        source=source,
        roi=roi,
        mode="LOCKED",
        packet_identity_statuses={},
    )
    ready = threading.Event()
    ready.set()
    return dict(
        worker=worker,
        testing=testing,
        testing_ready=ready,
        testing_lock=threading.RLock(),
    )
