"""Controller state vocabulary, independent of engine evidence outcomes."""
from enum import Enum
class SOPState(str,Enum):
    IDLE='IDLE';READY='READY';STARTING='STARTING';RUNNING='RUNNING'
    STOPPING='STOPPING';FINALIZING='FINALIZING';INTERRUPTED='INTERRUPTED';ERROR='ERROR'
