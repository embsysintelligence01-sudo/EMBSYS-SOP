#!/usr/bin/env python3
"""Exercise real prepared storage paths without opening a physical camera."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'engines/Jeweltracker'),str(ROOT/'engines/Packetsealing_001')]
from evidence_archive import checked_local

captures=ROOT/'engines/Jeweltracker/captures'
assert not captures.is_symlink(), 'Purity archive containment requires a physical capture directory'
captures.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='.cleanup-storage-check-',dir=captures) as temporary:
    directory=Path(temporary)
    sample=directory/'evidence.bin'
    sample.write_bytes(b'path-containment-check')
    assert checked_local(directory,sample)==sample.resolve()
    sample.write_bytes(b'path-containment-check')
print('PASS: actual purity capture ancestry satisfies unchanged evidence containment checks')

packet=ROOT/'engines/Packetsealing_001'
sys.path.insert(0,str(packet/'tests'))
from test_surveillance_handoff import SurveillanceExportTests
original_temporary_directory=tempfile.TemporaryDirectory

def output_temporary_directory(*args,**kwargs):
    kwargs.setdefault('dir',packet/'outputs')
    return original_temporary_directory(*args,**kwargs)

# The existing controlled-camera regression now writes/decodes through the
# actual prepared packet output alias, rather than only an unrelated /tmp root.
with patch('tempfile.TemporaryDirectory',side_effect=output_temporary_directory):
    SurveillanceExportTests('test_zero_assigned_ids_records_and_exports_without_starting_a_new_batch').test_zero_assigned_ids_records_and_exports_without_starting_a_new_batch()
print('PASS: packet output alias supports zero-ID recording, decode and independent JSON/PDF export')
