"""Complete full8 + canonical21/4750 for the actual isolated H192 Muon endpoint."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/data/star/VAELLM/g09_hotpot_coverage_20261030')
path = ROOT / 'experiments/overnight_goal_20261029/evaluate_muon.py'
spec = importlib.util.spec_from_file_location('_hotpot192_fullnine_mature_entry', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.SOURCE = SOURCE
entry.OUTPUT = SOURCE / 'evaluation'
# Muon remains the mature profile name; the entire destination belongs to H192.
entry.DESTINATION = entry.OUTPUT / 'Muon'
original_pins = entry.source_pins


def source_pins(binding):
    return {**original_pins(binding), str(Path(__file__).resolve()):entry.b.sha256(Path(__file__))}


entry.source_pins = source_pins


if __name__ == '__main__':
    entry.main()
