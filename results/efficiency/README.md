# Efficiency measurements (`experiments/efficiency.py`)

- `itm21_a4000_v4_DLinear_iTransformer.csv`, `itm21_a4000_v4_TimeXer_TimeFilter.csv` (and `_scaling`): current results
  (paper Table XII, Fig. 3, Fig. 4, Table S5). Same code paths and settings as v3, measured 2026-10-23 on the idle RTX A4000
  (`GPUIdleBefore` and `GPUIdleAfter` True in every row) by `.worker/bench_v4.sh` on itm21: all 7 datasets x 4 backbones
  x H 96, 192, 336, 720 x NoNorm, RevIN, SAN, DDN, FAN, LightNorm (112 cases).
- `itm21_a4000_v6_dishts.csv` (and `_scaling`): Dish-TS with its own NoNorm rows, measured 2026-10-29 by `.worker/bench_v6.sh`
  on the idle itm21 RTX A4000: 7 datasets x 4 backbones x H 96, 192, 336, 720 (112 cases), each case at the maximum batch
  of v4 (DLinear/iTransformer) or v5 (TimeXer/TimeFilter) via `--batch-from`, so Dish-TS overheads (over the NoNorm rows
  of the same file) compare with the other methods; NoNorm latency at that batch is within 0.5% (median) of v4/v5.
- `itm21_a4000_v3.csv`, `itm21_a4000_v3_scaling.csv`: previous results (ETTh1 and Traffic only). RTX A4000 with no other
  process on the GPU (`GPUIdleBefore` = True in every row), measured 2026-10-17 and 2026-10-19 at
  code 79f58056 and earlier commits with the same measurement code paths. ETTh1 and Traffic x
  DLinear, iTransformer, TimeXer, TimeFilter x H 96, 720 x NoNorm, RevIN, SAN, DDN, FAN, LightNorm.
  Each method is built and measured alone; within a case every method uses the same batch.
- `itm21_a4000.csv`, `itm21_a4000_scaling.csv`: first run (2026-10-14), kept for reference only.
  Its peak-memory columns are inflated for every method after the first one of a case (gradients
  and optimizer state of earlier methods were not freed), and LightNorm's inference and training
  paths have since been optimized with identical outputs.
