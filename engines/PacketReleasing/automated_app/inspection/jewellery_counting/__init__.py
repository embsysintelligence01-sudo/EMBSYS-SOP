"""Stage 2: Jewellery detection, separation, tracking, and counting components."""

from .count_manager import JewelleryCountManager, Stage2FrameEvaluation
from .instance_separator import InstanceSeparator, SeparatedArticle, SeparationResult
from .jewel_detector import JewelDetector, TestbedDetectionResult
from .object_tracker import JewelleryObjectTracker, TrackedJewel, TrackingSnapshot
from .testbed_roi import EmptyTestbedResult, TestbedROIManager

__all__ = [
    "EmptyTestbedResult",
    "InstanceSeparator",
    "JewelDetector",
    "JewelleryCountManager",
    "JewelleryObjectTracker",
    "SeparatedArticle",
    "SeparationResult",
    "Stage2FrameEvaluation",
    "TestbedDetectionResult",
    "TestbedROIManager",
    "TrackedJewel",
    "TrackingSnapshot",
]

