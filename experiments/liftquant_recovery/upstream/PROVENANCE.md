Source repository: https://github.com/Heliulu/LiftQuant
Pinned commit: 72b3875c770e4579639931fed89dc95e4067edac
Fetched raw files on 2026-09-23; main was also this commit when audited:
- quantize/liftq.py -> liftq.py.reference
- datautils.py -> datautils.py.reference
- quantize/tmplinear.py -> tmplinear.py.reference
- quantize/int_llama_layer.py -> int_llama_layer.py.reference
- main.py -> main.py.reference
- utils.py -> utils.py.reference
- trans_utils.py -> trans_utils.py.reference
- models/LMClass.py -> LMClass.py.reference
- requirements.txt -> requirements.txt.reference

These are unmodified reference snapshots, not executable dependencies.
The experiment uses VAELLM packed code/decoder representations; it does not import
LiftQuant FWTLinear or run its whitening/transformation initialization stages.
Do not call the current adapter an exact Stage2 reproduction. See
../DESIGN_AUDIT_20260923.md for parameterization, grouping, learning-rate,
scheduler, mode, dtype and environment differences, with CPU audit evidence.
Official requirements use torch 2.6.0 and transformers 5.9.0; the existing bitvae
environment uses torch 2.6.0+cu124 and transformers 4.51.0. No upgrade was made.
