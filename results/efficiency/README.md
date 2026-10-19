# Efficiency measurements (`experiments/efficiency.py`)

- `itm21_a4000_v3.csv`, `itm21_a4000_v3_scaling.csv`: current results. RTX A4000 with no other
  process on the GPU (`GPUIdleBefore` = True in every row), measured 2026-10-17 and 2026-10-19 at
  code 79f58056 and earlier commits with the same measurement code paths. ETTh1 and Traffic x
  DLinear, iTransformer, TimeXer, TimeFilter x H 96, 720 x NoNorm, RevIN, SAN, DDN, FAN, LightNorm.
  Each method is built and measured alone; within a case every method uses the same batch.
- `itm21_a4000.csv`, `itm21_a4000_scaling.csv`: first run (2026-10-14), kept for reference only.
  Its peak-memory columns are inflated for every method after the first one of a case (gradients
  and optimizer state of earlier methods were not freed), and LightNorm's inference and training
  paths have since been optimized with identical outputs.
