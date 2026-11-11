"""Deterministic SC-CAT recovery, isolated from the original running contract."""
from experiments.sc_cat_reproducibility_cmcc_20261109.execution import configure_deterministic_backward

EXECUTION = configure_deterministic_backward()
