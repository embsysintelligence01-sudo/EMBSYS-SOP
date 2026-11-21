"""Stable private configuration/evidence locations during package migration."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PURITY_DATA = ROOT/'engines/Jeweltracker'
SEALING_DATA = ROOT/'engines/Packetsealing_001'
