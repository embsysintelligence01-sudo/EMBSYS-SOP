"""Real existing Qt workers behind one UI-independent lifecycle. No alternate detectors."""

import json
import os
from pathlib import Path
import time
from app.config import ROOT, JEWEL, PACKET, RELEASING
from services.evidence import publish
from services.handoff import SurveillanceHandoff, recovered_handoff
from services.identity import plain


class PledgeAdapter:
    def __init__(self, emit, debug=False):
        import jewel_tracker_5 as tracker
        from engines.sealing.workflow import PacketWorker

        self.tracker = tracker
        self.PacketWorker = PacketWorker
        self.emit = emit
        self.debug = debug
        self.worker = None
        self.current = None
        self.context = None
        self.purity = None
        self.purity_message = None
        self.results = {}
        self.errors = {}
        self.released = set()
        self.finished = set()
        self.thread_done = set()
        self.running = set()
        self.rub_models = None
        self.acid_models = None
        self.status = "IDLE"
        self.table = []
        self.sealing_state = {}
        self.identity_key = None
        self.identity_version = 0
        self.diagnostics = None

    def source_settings(self):
        from engines.sealing.settings import tracker_camera, tracker_source, tracker_roi

        text = os.environ.get("EMBSYS_REPLAY_SOURCE") or tracker_source()
        source = int(text) if str(text).isdigit() else text
        camera = tracker_camera(source)
        return source, camera, tracker_roi()

    def start(self, message):
        if self.worker is not None and self.worker.isRunning():
            from engines.purity.packet_handoff import camera_released

            if self.context is None or not camera_released(self.purity):
                raise RuntimeError("Previous live worker has not released acquisition")
        self.current = message
        activity = message["activity"]
        aid = message["attempt_id"]
        directory = Path(message["directory"])
        source, camera, roi = self.source_settings()
        if os.environ.get("EMBSYS_REPLAY_SOURCE"):
            from engines.sealing.camera import VideoFileGrabber

            self.tracker.RTSPGrabber = VideoFileGrabber
        if activity["engine"] == "purity":
            zones = json.loads(Path(self.tracker.ZONES_FILE).read_text())
            self.rub_models = self.tracker.RubPhotoModels()
            self.acid_models = self.tracker.AcidModelCache()
            self.tracker.CAPTURE_DIR = str(directory / "purity")
            worker = self.tracker.Worker(
                source,
                roi,
                zones.get("stone"),
                camera=camera,
                continuous=True,
                test_pts=zones.get("test"),
                rub_models=self.rub_models,
                acid_models=self.acid_models,
                visualization=self.debug,
            )
            worker.set_recording_settings(
                dict(
                    margin_percent=zones.get("stone_grip_margin_percent", 20),
                    hand=zones.get("pinch_hand", "Any"),
                    pinch_px=zones.get("pinch_px", 460),
                    gold_fallback=zones.get("box_gold_fallback", False),
                )
            )
            worker.preserve_gold_model = True
            if self.debug:
                from services.diagnostics import PurityDiagnostics

                self.diagnostics = PurityDiagnostics()
                worker.workflow.connect(self.diagnostics.event)
                worker.event.connect(self.diagnostics.log)
                worker.jewel_table.connect(lambda rows: setattr(self, "table", rows))
            worker.event.connect(lambda text, level: self.notice(aid, text))
            worker.status.connect(lambda text: self.notice(aid, text))
            self.purity = worker
            self.purity_message = message
        else:
            handoff = None
            if activity["linked"]:
                if self.context is not None:
                    handoff = self.context.payload()
                else:
                    handoff = recovered_handoff(
                        message.get("history"), source, camera, roi
                    )
            worker = self.PacketWorker(
                source,
                roi,
                handoff=handoff,
                output_root=directory / "sealing",
                visualization=self.debug,
            )
            worker.notice.connect(lambda text: self.notice(aid, text))
            worker.failed.connect(lambda text: self.failed(aid, text))
            worker.exported.connect(lambda result: self.results.update({aid: result}))
            if self.debug:
                worker.state.connect(
                    lambda value: setattr(self, "sealing_state", value)
                )
        self.worker = worker
        worker.finished.connect(lambda: self.thread_done.add(aid))
        worker.start()

    def notice(self, aid, text):
        self.status = str(text)
        if "Model load failed" in text:
            self.failed(aid, text)

    def failed(self, aid, text):
        if aid not in self.errors:
            self.errors[aid] = str(text)
            self.emit("error", aid, error=str(text))

    def stop(self):
        if self.worker:
            self.worker.stop()
            if not self.worker.isRunning():
                grabber = getattr(self.worker, "grabber", None)
                if grabber is not None and grabber.is_alive():
                    grabber.stop()

    def tick(self):
        if not self.current:
            return
        message = self.current
        aid = message["attempt_id"]
        worker = self.worker
        grabber = getattr(worker, "grabber", None)
        if aid in self.thread_done and aid not in self.released:
            self.stop()
        if (
            aid not in self.running
            and grabber is not None
            and getattr(grabber, "connected", False)
            and getattr(grabber, "frame", None) is not None
        ):
            self.running.add(aid)
            self.emit("running", aid)
        if message["activity"]["engine"] == "purity":
            from engines.purity.packet_handoff import camera_released, testing_snapshot

            if worker.running and getattr(worker, "frame_wh", None):
                key = (
                    tuple(getattr(t, "id", None) for t in worker.tracks),
                    worker.mode,
                    bool(getattr(worker.id_model, "ready", False)),
                )
                if key != self.identity_key:
                    from services.identity import identity_snapshot

                    value = dict(
                        testing=testing_snapshot(worker, Path(message["directory"])),
                        identity=identity_snapshot(worker),
                    )
                    publish(
                        Path(message["directory"])
                        / ("identity-" + str(self.identity_version) + ".json"),
                        value,
                    )
                    publish(Path(message["directory"]) / "identity.json", value)
                    self.identity_key = key
                    self.identity_version += 1
            if (
                (not worker.running or aid in self.thread_done)
                and camera_released(worker)
                and aid not in self.released
            ):
                try:
                    self.context = SurveillanceHandoff(
                        worker, Path(message["directory"])
                    )
                except Exception as exc:
                    self.failed(aid, "Identity snapshot failed: " + str(exc))
                self.released.add(aid)
                self.emit("released", aid)
        elif (
            (
                (
                    getattr(worker, "capture_stopped", None) is not None
                    and worker.capture_stopped.is_set()
                )
                or aid in self.thread_done
            )
            and (grabber is None or not grabber.is_alive())
            and aid not in self.released
        ):
            self.released.add(aid)
            self.emit("released", aid)
        # The original purity worker drains its original ContinuousMonitor before finished.
        if self.purity_message:
            previous_id = self.purity_message["attempt_id"]
            if previous_id in self.thread_done and previous_id not in self.finished:
                if self.context is not None:
                    self.context.finish()
                    result = dict(self.context.testing)
                    report = Path(result["directory"]) / "report.pdf"
                    if report.is_file():
                        result["report"] = str(report)
                else:
                    result = dict(
                        evidence_status="INCOMPLETE", error=self.errors.get(previous_id)
                    )
                self.publish_finished(
                    self.purity_message,
                    result,
                    result.get("evidence_status", "INCOMPLETE"),
                )
                if self.rub_models:
                    self.rub_models.close()
                if self.acid_models:
                    self.acid_models.close()
        if (
            message["activity"]["engine"] == "packet"
            and aid in self.thread_done
            and aid not in self.finished
        ):
            result = self.results.get(aid, {})
            status = result.get("evidence_statuses", {}).get(
                "packet_sealing", "INCOMPLETE"
            )
            self.publish_finished(message, result, status)

    def publish_finished(self, message, result, status):
        aid = message["attempt_id"]
        if aid not in self.released:
            return  # Never replace the acquisition barrier with thread completion.
        value = dict(
            event="finished",
            attempt_id=aid,
            evidence_status="REVIEW_REQUIRED" if aid in self.errors else status,
            reports=plain(result),
            error=self.errors.get(aid),
        )
        publish(Path(message["directory"]) / "engine_result.json", value)
        self.finished.add(aid)
        self.emit(
            "finished",
            aid,
            **{k: v for k, v in value.items() if k not in ("event", "attempt_id")},
        )

    def busy(self):
        workers = [w for w in (self.worker, self.purity) if w is not None]
        return any(
            w.isRunning()
            or any(
                getattr(w, name, None) is not None and getattr(w, name).is_alive()
                for name in ("grabber", "stone", "id_training_thread")
            )
            for w in workers
        )

    def close(self):
        if self.busy():
            self.stop()
            return False
        model = getattr(self.purity, "model", None)
        if model is not None:
            model.close()
        if self.rub_models:
            self.rub_models.close()
        if self.acid_models:
            self.acid_models.close()
        return True

    def debug_frame(self):
        worker = self.worker
        grabber = getattr(worker, "grabber", None)
        if grabber is None:
            return None
        frame, stamp = grabber.latest()
        if frame is None:
            return None
        if self.current["activity"]["engine"] == "purity":
            frame = worker.preview.render(frame, stamp)
            frame = self.tracker.draw_hand_preview(
                frame, stamp, getattr(worker, "recording_hand_preview", None)
            )
        else:
            worker.live_overlay.render(frame, stamp)
            worker.gold_overlay.render(frame, stamp)
        return frame


class ReleaseAdapter:
    def __init__(self, emit, debug=False):
        self.emit = emit
        self.debug = debug
        self.worker = None
        self.current = None
        self.result = None
        self.paths = {}
        self.released = False
        self.done = False
        self.running = False
        self.errors = []
        self.status = "IDLE"
        self.live = {}
        self.system = {}
        self.compliance = {}
        self.manager = None

    def start(self, message):
        self.result = None
        self.paths = {}
        self.released = False
        self.done = False
        self.running = False
        self.errors = []
        self.live = {}
        self.system = {}
        self.compliance = {}
        from packet_cut.desktop.worker import PacketWorker
        from packet_cut.desktop.automation import AutoPipelineConfig
        from workflow_monitor.config import WorkflowMonitorConfig
        from workflow_monitor.models import UnifiedModelManager

        cfg = WorkflowMonitorConfig.load()
        if self.manager is None:
            self.manager = UnifiedModelManager(cfg)
            self.manager.load_all_models(
                keys=tuple(k for k in self.manager.ALL_KEYS if k != "sam")
            )
        source = os.environ.get("EMBSYS_REPLAY_SOURCE") or 0
        if str(source).isdigit():
            source = int(source)
        candidates = [
            ROOT / "configs/releasing-camera.local.json",
            RELEASING / "packet_cut_outputs/testbed_roi.json",
            RELEASING / "automated_app/packet_cut_outputs/testbed_roi.json",
        ]
        roi_file = next((p for p in candidates if p.is_file()), None)
        if roi_file is None:
            raise RuntimeError(
                "Release ROI configuration is missing; configure it with the preserved development UI"
            )
        roi_data = json.loads(roi_file.read_text())
        roi = roi_data.get("roi")
        if not roi or len(roi) != 4:
            raise RuntimeError(
                "Configure the release camera ROI using its preserved development interface first"
            )
        pipe = AutoPipelineConfig(
            output_dir=str(Path(message["directory"]) / "release")
        )
        self.current = message
        aid = message["attempt_id"]
        self.worker = PacketWorker(
            source,
            device="cpu",
            model_manager=self.manager,
            pipeline_config=pipe,
            manual_roi=tuple(roi),
            visualization=self.debug,
        )
        self.worker.acquisition_stopped.connect(self.release)
        self.worker.error.connect(self.error)
        self.worker.final_report_ready.connect(self.report)
        self.worker.finished.connect(self.finish)
        self.worker.live_monitoring_updated.connect(
            lambda value: setattr(self, "live", value)
        )
        self.worker.compliance_updated.connect(
            lambda value: setattr(self, "compliance", value)
        )
        self.worker.system_monitoring_updated.connect(self.monitor)
        self.worker.start()

    def monitor(self, value):
        self.system = value
        self.status = str(value.get("camera_status", "ACTIVE"))

    def error(self, text):
        self.errors.append(str(text))
        self.emit("error", self.current["attempt_id"], error=str(text))

    def release(self):
        if not self.released:
            self.released = True
            self.emit("released", self.current["attempt_id"])

    def report(self, result, paths):
        self.result = result
        self.paths = paths

    def finish(self):
        if self.done:
            return
        self.release()
        self.done = True
        # The real release report supplies individual check results. A completed session alone is not verification.
        checks = (self.result or {}).get("release_checks", [])
        status = (
            "VERIFIED"
            if checks and all(c.get("status") == "VERIFIED" for c in checks)
            else "INCOMPLETE"
        )
        value = dict(
            evidence_status="REVIEW_REQUIRED" if self.errors else status,
            reports=self.paths,
            check_statuses={c["key"]: c["status"] for c in checks},
            errors=self.errors,
            historical_pledge_id=self.current["activity"].get("pledge_id"),
            linkage=self.current["activity"].get("linkage"),
        )
        publish(
            Path(self.current["directory"]) / "engine_result.json",
            dict(event="finished", attempt_id=self.current["attempt_id"], **value),
        )
        self.emit("finished", self.current["attempt_id"], **value)

    def stop(self):
        if self.worker:
            self.worker.stop()

    def tick(self):
        if self.worker and self.worker.last_frame is not None and not self.running:
            self.running = True
            self.emit("running", self.current["attempt_id"])

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def close(self):
        if self.busy():
            self.stop()
            return False
        return True

    def debug_frame(self):
        snapshot = getattr(self.worker, "last_snapshot", None)
        if snapshot is not None:
            return snapshot.annotated_frame
        return self.worker.last_frame if self.worker else None
