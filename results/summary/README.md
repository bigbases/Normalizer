# results/summary 해석 가이드

이 폴더의 CSV/JSON/MD는 `results/store/`의 셀 결과(실행 1회 = CSV 1개)로부터 자동 생성됩니다.
README.md만 수동 문서이며, 나머지는 아래 명령으로 언제든 재생성됩니다.

```bash
python experiments/results_pack.py summary
```

- 기준 시점 상태: 실험 1–3 전체 완료, 1,792셀 (search 560 / confirm 224 / final 1,008).
- 프로토콜 버전: `journal-v1` (`configs/protocol.json`).

---

## 1. 실험 목적

| 실험 | 목적 | 단계(phase) | 셀 수 |
|---|---|---|---|
| Exp1 | 제안 방법 LightNorm의 최종 test 성능 (주어진 고정 설정) | `exp1-lt` | 168 |
| Exp2 | 비교 방법(RevIN·SAN·DDN·FAN)의 모듈 하이퍼파라미터를 **validation만으로** 선택 | `exp2-search`, `exp2-confirm` | 560 + 224 |
| Exp3 | 동일 백본·동일 학습 조건에서 6개 방법(NoNorm 포함)의 test 비교 | `exp3-none`, `exp3-tuned` (+ Exp1 셀 공유) | 840 |
| Exp5 (진행 중) | LightNorm 튜닝 축을 고르는 파일럿 (validation만 사용) | `lt-pilot` | 60 |

- Exp1의 LightNorm 셀은 Exp3의 `lt` 셀과 run ID가 같습니다. 따라서 따로 반복 실행하지 않았습니다.
- Exp3 비교에서 바뀌는 것은 정규화 모듈과 그 설정뿐입니다. 백본 구조, 백본 학습률, 데이터 분할, 학습 예산은 모든 방법이 동일합니다.

---

## 2. 공통 설정 (모든 방법 동일)

| 항목 | 설정 |
|---|---|
| 데이터셋 (채널 수) | ETTh1, ETTh2, ETTm1, ETTm2 (7) / Weather (21) / Electricity (321) / Traffic (862) |
| 분할 | ETT: train 12개월 / val 4개월 / test 4개월. 그 외: 70% / 10% / 20% (시간 순서) |
| 스케일링 | train 구간으로 fit한 StandardScaler. **MSE/MAE는 표준화된 값 기준** (문헌 관례와 동일) |
| 예측 설정 | 다변량 → 다변량 (`features=M`). 예측 길이 H ∈ {96, 192, 336, 720} |
| 백본 | DLinear, iTransformer. 데이터셋별 구조·학습률은 `configs/base_configs.json` 사용 (아래 표) |
| iTransformer 내부 정규화 | **비활성화**. 정규화 효과는 외부 모듈에서만 발생 |
| 옵티마이저 | Adam. 백본은 `learning_rate`, 정규화 모듈은 별도 Adam(`station_lr`) |
| 학습 예산 | 백본 최대 10 epoch, validation MSE 기준 early stopping (patience 3) |
| 학습률 스케줄 | type1: 백본 epoch마다 LR × 0.5 |
| 정규화 모듈 사전학습 | SAN·DDN·LightNorm은 모듈만 5 epoch 먼저 학습한 뒤, best 상태를 불러와 본 학습 시작 |
| 손실 | 역정규화한 예측의 MSE. FAN만 원 논문대로 잔차 MSE + 주요 주파수 MSE(가중치 1.0) |
| 시드 | 최종 test: 2021 / 2022 / 2023 (3회). search: 2021. confirm: 2022 |
| 재현성 | deterministic 커널 요청. 셀마다 Host, GPU, Torch, CodeRev, 소요 시간 기록 (`results/store/`) |

**백본 설정** (`base_configs.json`, 표에 없는 값은 코드 기본값)

| 백본 | 입력 길이 | label 길이 | 구조 | batch | 백본 LR |
|---|---|---|---|---|---|
| DLinear | 336 | 168 | 공급 코드의 DLinear. 추세 분기는 Linear(336→2048→H) | 32 | ETTh1 5e-4, ETTh2 1e-3, ETTm1 1e-4, ETTm2 1e-3, Weather 5e-4, Electricity 1e-3, Traffic 5e-4 |
| iTransformer | 720 | 168 | ETT: d_model = d_ff = 128, 2층 / Weather·Electricity: 512, 3층 / Traffic: 512, 4층 | 32 (Electricity·Traffic 16) | ETTh1 1e-4, ETTh2 5e-4, ETTm1 1e-4, ETTm2 5e-4, Weather 5e-4, Electricity 5e-4, Traffic 1e-3 |

---

## 3. 방법별 설정

| 키 | 방법 | 학습 방식 | 탐색 공간 (Exp2) |
|---|---|---|---|
| `none` | NoNorm | 정규화 없음 | – |
| `revin` | RevIN | 입력 구간 평균·분산으로 정규화. affine 파라미터는 백본과 함께 학습 | `affine` ∈ {0, 1} (2개) |
| `san` | SAN | 사전학습 5 epoch(구간별 통계 예측) 후 모듈 고정, 백본만 학습 | `period_len` ∈ {4, 8, 24} × `station_lr` ∈ {0.5×, 1×} (6개) |
| `ddn` | DDN | 사전학습 5 epoch 후, 백본 1 epoch 뒤부터 함께 학습. wavelet coif3 | `kernel_len` ∈ {7, 25, 49} × `j` ∈ {0, 1}, `hkernel_len` = 5 (6개) |
| `fan` | FAN | 주파수 예측기를 처음부터 백본과 함께 학습 | K(`freq_topk`) 3개 × LR ∈ {station_lr, 백본 LR} (6개) |
| `lt` | LightNorm (제안) | 사전학습 5 epoch 후 함께 학습 | **탐색 없음**. 주어진 설정 고정 (아래 표) |

- 비교 방법의 `station_lr` 기준값은 1e-4입니다. iTransformer × ETTh2만 5e-4입니다.
- FAN의 K 후보는 데이터셋별로 정했습니다 (첫 값이 저자 권장값).

  | 데이터셋 | K 후보 |
  |---|---|
  | ETTh | {4, 2, 8} |
  | ETTm | {11, 6, 16} |
  | Weather | {2, 1, 4} |
  | Electricity | {3, 1, 6} |
  | Traffic | {30, 15, 45} |

  백본 LR이 station_lr과 같으면 LR 후보는 station_lr의 {0.5×, 1×}로 대신합니다.
- DDN의 통계 예측 MLP 크기는 모든 데이터셋에서 고정입니다 (pd_model 512, pd_ff 1024, pe_layers 2).
- 그리드 수정 이력은 `protocol.json`의 `grid_revisions`에 있습니다. DDN은 j=0에서 `hkernel_len`이 무의미하다는 문제를, FAN은 LR 그리드 문제를 수정했습니다.

**LightNorm 동작과 고정 설정**

LightNorm은 입력을 이동평균(SMA, 창 `kernel_size`)으로 추세와 잔차로 나눕니다.
- **잔차:** `s_norm` = 1이면 인스턴스 정규화한 뒤 백본에 넣습니다.
- **추세:** 정규화한 뒤 경량 예측기가 예측합니다. 예측기는 `down_ratio`배로 다운샘플한 다음 Linear(`use_mlp` = 1이면 은닉 64의 MLP)를 거치고, 선형 보간으로 원래 길이로 되돌립니다.
- **출력:** 백본 출력을 역정규화한 값에 예측한 추세를 더합니다.

모든 케이스 공통으로 `affine` = 1, `t_norm` = 1, `t_ff` = 64입니다. 케이스별 값은 다음과 같습니다.

| 케이스 | s_norm | use_mlp | down_ratio | kernel_size | station_lr |
|---|---|---|---|---|---|
| ETTh1 × DLinear | 0 | 0 | 4 | 25 | 1e-4 |
| ETTh1 × iTransformer | 1 | 0 | 4 | 25 | 1e-4 |
| ETTh2 × DLinear | 1 | 0 | 2 | 25 | 1e-4 |
| ETTh2 × iTransformer | 0 | 0 | 2 | 25 | 5e-4 |
| ETTm1 × DLinear | 1 | 1 | 4 | 13 | 1e-4 |
| ETTm1 × iTransformer | 0 | 0 | 4 | 25 | 1e-4 |
| ETTm2 × DLinear | 1 | 0 | 8 | 13 | 1e-4 |
| ETTm2 × iTransformer | 0 | 0 | 4 | 25 | 1e-4 |
| Weather × DLinear | 1 | 1 | 4 | 13 | 1e-4 |
| Weather × iTransformer | 1 | 1 | 8 | 25 | 1e-4 |
| Electricity × DLinear | 0 | 1 | 4 | 25 | 1e-4 |
| Electricity × iTransformer | 1 | 1 | 8 | 25 | 1e-4 |
| Traffic × DLinear | 0 | 1 | 4 | 25 | 1e-4 |
| Traffic × iTransformer | 1 | 1 | 4 | 25 | 1e-4 |

`base_configs.json`에는 `kernel_len`(DDN 전용 인자)도 있습니다. LightNorm은 이 값을 사용하지 않습니다. ETTm × iTransformer는 `kernel_len` = 13이지만 실제로는 `kernel_size` = 25로 실행되었습니다.

---

## 4. 하이퍼파라미터 선택 절차 (Exp2, validation만 사용)

1. **Search:** seed 2021로 H = 96, 720에서 모든 후보를 학습하고 BestValMSE를 기록합니다. 후보 280개 = 14케이스 × (2+6+6+6).
2. **Shortlist:** H 96·720 평균(`Screen_mean`)이 가장 낮은 2개를 고릅니다. RevIN은 후보가 2개라 둘 다 통과합니다.
3. **Confirm:** shortlist 2개를 seed 2022, H = 96, 720으로 재학습합니다.
4. **Lock:** 4개 값(시드 2개 × H 2개)의 평균(`Lock_mean`)이 가장 낮은 후보 1개를 확정합니다.
5. **Final (Exp3):** lock된 설정을 H 192·336을 포함한 4개 H × 3 seed에 그대로 적용해 학습하고, test를 1회 평가합니다.

search·confirm 셀은 `--skip_test`로 실행되어 test split을 만들지 않습니다. 학습 중 early stopping도 validation만 사용하므로, lock 전에는 test 값이 존재하지 않습니다.

---

## 5. 파일별 설명과 열 정의

### 공통 용어

| 용어 | 의미 |
|---|---|
| cell (셀) | 학습·평가 1회. 데이터셋 × 백본 × 방법 × 후보 × H × seed. `results/store/<phase>/<dataset>/<backbone>/<RunID>.csv` |
| case (케이스) | 데이터셋 × 백본 조합 (14개). Exp3 표의 한 행은 케이스 × H |
| candidate (후보) | 한 방법의 모듈 하이퍼파라미터 조합 1개 |
| `CandidateID` | 방법 + 전체 파라미터(JSON)의 SHA-256 앞 12자리. 같은 설정은 어느 단계에서나 같은 ID |
| `RunID` | `<stage>-<Dataset>-<Backbone>-<method>-h<H>-s<seed>-c<CandidateID>-x<config hash 8자리>` |
| stage | `search`(seed 2021, val), `confirm`(seed 2022, val), `final`(3 seeds, test) |
| BestValMSE | early stopping이 고른 best 백본 epoch의 validation MSE. 순수 예측 MSE로, FAN 보조 손실은 제외 |
| 방법 키 | `none` NoNorm, `revin` RevIN, `san` SAN, `ddn` DDN, `fan` FAN, `lt` LightNorm |
| 그룹 키 | JSON의 `"Dataset\|Backbone\|method"` (예: `ETTh1\|DLinear\|ddn`) |
| `_std` | seed 3개에 대한 표본 표준편차 (n−1) |

### `stage_status.csv`, `progress.md`: 분산 실행 진행 현황

| 열 | 의미 |
|---|---|
| `Phase` | `exp1-lt`, `exp2-search`, `exp2-confirm`, `exp3-none`, `exp3-tuned` (`configs/distributed_plan.json`) |
| `Method` | 그 task가 담당하는 방법. search·confirm·tuned는 방법별로 task를 나눔 (task ID 끝 `--revin` 등) |
| `Status` | `done` 모든 셀 완료 / `claimed` 워커가 점유해 실행 중 / `ready` 실행 가능 / `blocked` 선행 단계(같은 케이스·방법) 대기 / `failed` 실패 보고 |
| `CellsDone` / `CellsTotal` | 완료 셀 수 / 필요 셀 수. 선행 단계 전이라 후보가 미정이면 Total은 빈칸 |
| `Worker` | claimed·failed일 때 점유 워커 |

### `exp1_lightnorm.csv`: LightNorm test 성능 (56행 = 14 케이스 × 4 H)

| 열 | 의미 |
|---|---|
| `Complete` | 3개 seed 모두 완료 |
| `Seeds` | 완료된 seed 수 |
| `MSE_mean`, `MSE_std`, `MAE_mean`, `MAE_std` | 3 seed 평균과 표준편차 |
| `MSE_s2021` 등 | seed별 test 값 |
| `GPU`, `Torch` | 실행 환경 (여러 개면 `;`로 구분) |

### `exp2_validation.csv`: 비교 방법의 모든 후보 (280행)

| 열 | 의미 |
|---|---|
| `Params` | 기본값까지 채운 전체 모듈 파라미터 (JSON) |
| `Search_h96_s2021`, `Search_h720_s2021` | search 단계 BestValMSE (H, seed) |
| `Confirm_h96_s2022`, `Confirm_h720_s2022` | confirm 단계 BestValMSE. shortlist에 들지 못한 후보는 빈칸 |
| `Screen_mean` | search 두 값의 평균. shortlist 기준 |
| `Lock_mean` | 4개 값 평균. lock 기준 |
| `Shortlisted` | 상위 2개에 들어 confirm을 수행했는지 |
| `Locked` | Exp3에 사용된 최종 설정인지 |

### `exp2_selection.csv`: 확정 설정 (56행 = 14 케이스 × 4 방법)

`Lock_mean_val_MSE`는 위의 `Lock_mean`과 같습니다.

`Params`의 주요 키는 다음과 같습니다.

| 방법 | 키 |
|---|---|
| RevIN | `affine` |
| SAN | `period_len` (통계 구간 길이) |
| DDN | `kernel_len` (슬라이딩 창), `j` (wavelet 분해 레벨, 0이면 주파수 분기 꺼짐), `hkernel_len` (고주파 창), `twice_epoch` (공동 학습 시작 epoch), `wavelet` |
| FAN | `freq_topk` (K), `fan_aux_weight` |
| 공통 | `station_lr` (모듈 학습률), `station_type` |

### `selection_shortlist.json`, `selection_locks.json`

위 shortlist·lock 결과를 기계가 읽는 형태로 담은 파일입니다. `fill_missing.py`가 confirm/final 셀을 만들 때 같은 규칙으로 다시 계산합니다.

### `final_test_mean_std.csv`: 모든 final 셀의 긴 형식 요약 (336행 = 56 × 6 방법)

| 열 | 의미 |
|---|---|
| `Seeds` | 집계에 포함된 seed 목록 |
| `GPUs` | 해당 셀들이 실행된 GPU |

### `exp3_comparison.csv`: 6개 방법 test 비교, 넓은 형식 (56행)

| 열 | 의미 |
|---|---|
| `<m>_n` | 방법 m의 완료 seed 수 (3이면 완전) |
| `<m>_MSE`, `<m>_MSE_std`, `<m>_MAE`, `<m>_MAE_std` | 3 seed 평균과 표준편차 |
| `Complete` | 6개 방법 모두 3 seed 완료 |
| `Best_MSE` | 평균 MSE가 가장 낮은 방법 키. Complete일 때만 채워짐 |

- 보고서의 "승률(win rate)"은 이 표에서 계산합니다. 같은 행(케이스 × H)에서 `lt_MSE < <baseline>_MSE`인 행의 비율이며, 56행 기준입니다.
- 3 seed 평균끼리 비교한 값이므로 통계적 유의성을 뜻하지 않습니다.

### Exp5 LightNorm 튜닝 파일럿 (`lt-pilot`)

**목적.** LightNorm의 공통 탐색 공간(6개 후보 = 축 2개)을 정하기 위해 축 후보 3개의 효과를 측정합니다. 설정은 `protocol.json`의 `lt_tuning`에 있습니다.

| 항목 | 내용 |
|---|---|
| 파일럿 케이스 | ETTm1 × iTransformer, ETTm1 × DLinear, Weather × DLinear |
| 조건 | seed 2021, H = 96, 720, validation만 사용 (`--skip_test`) |
| 설계 | 2³ 요인 설계(꼭짓점 8개) + 중심점 2개 = 케이스당 10개 후보 |
| s_norm | {0, 1} |
| kernel_size | 꼭짓점 {13, 49}, 중심점 25 |
| station_lr | 1e-n / 5e-n 사다리에서 주어진 값의 한 칸 아래·위. 중심점은 주어진 값 (1e-4 → {5e-5, 1e-4, 5e-4}) |
| 고정값 | down_ratio·kernel_len은 주어진 값, t_ff 64, affine 1, t_norm 1. 백본·학습 설정 불변 |
| use_mlp 규칙 | Weather·Electricity·Traffic만 1, 나머지 0. 주어진 설정과 다른 곳은 ETTm1 × DLinear(1→0)뿐 |

**축 결정 규칙** (결과를 보기 전에 고정, `lt_tuning.pilot.decision_rule`)
1. 다수의 케이스 × H 조합에서 효과 크기가 시드 변동(`Noise_pct`)을 넘는 축만 후보로 남깁니다.
2. 그중 평균 |효과|가 큰 2개 축을 고릅니다.
3. 모든 조합에서 같은 수준이 이기는 축은 탐색하지 않고 그 값으로 고정합니다.
4. 남는 축이 2개 미만일 때만 down_ratio {2, 8}을 추가로 확인합니다.
5. 최종 후보 6개 = 2수준 축 × 3수준 축이며, 주어진 설정(use_mlp 규칙 적용)을 항상 포함합니다.
6. 파일럿 케이스도 최종 설정은 이 6개 안에서만 고릅니다. 파일럿의 나머지 후보는 축을 정하는 데만 씁니다.

**`exp5_lt_pilot.csv`** (케이스당 11행 = 주어진 설정 1 + 파일럿 10)

| 열 | 의미 |
|---|---|
| `Point` | `supplied`: 주어진 설정. 값은 Exp1 final 셀(seed 2021)의 BestValMSE / `corner`: 요인 설계 꼭짓점 / `centre`: 중심점 (kernel 25, station_lr 주어진 값) |
| `s_norm` ~ `down_ratio` | 해당 후보의 LightNorm 설정 |
| `Val_h96_s2021`, `Val_h720_s2021`, `Val_mean` | validation MSE와 두 H의 평균 |
| `Delta_vs_supplied_pct` | (`Val_mean` − 주어진 설정의 `Val_mean`) / 주어진 설정 × 100. 음수면 주어진 설정보다 좋음 |
| `Rank` | 파일럿 후보 10개 중 `Val_mean` 순위 |

ETTm1 × iTransformer의 `centre`(s_norm 0)는 주어진 설정과 같은 설정입니다. 이 행과 `supplied` 행의 차이로 GPU 간 재현성도 확인할 수 있습니다. ETTm1 × DLinear의 `supplied`는 use_mlp = 1이라 파일럿 후보와 구조가 다릅니다.

**`exp5_lt_pilot_effects.csv`** (케이스 × H마다 7행)

| 열 | 의미 |
|---|---|
| `Term` | 주효과 `s_norm`, `kernel_size`, `station_lr`, 2요인 상호작용 `a:b`, 그리고 `centre_vs_corners` |
| `Effect_pct` | 주효과: (높은 수준 꼭짓점 평균 − 낮은 수준 꼭짓점 평균) / 꼭짓점 전체 평균 × 100. 높은 수준은 s_norm 1, kernel 49, station_lr 위쪽 값<br>상호작용: 두 요인의 부호가 같은 꼭짓점 평균 − 다른 꼭짓점 평균<br>`centre_vs_corners`: (중심점 평균 − 꼭짓점 평균) / 꼭짓점 평균. 음수면 중간값이 양 끝보다 좋음(비선형) |
| `Better_level` | 주효과에서 validation MSE가 더 낮은 수준 |
| `Noise_pct` | 주어진 설정의 3 seed validation MSE 변동계수(%). Exp1 final 셀에서 계산 |
| `Exceeds_noise` | \|`Effect_pct`\| > `Noise_pct` |

---

## 6. 제안 방법(LightNorm) 튜닝과 무관하게 사용 가능한 결과 범위

LightNorm 하이퍼파라미터를 추가로 튜닝해도 아래 결과는 영향을 받지 않습니다. 단, 백본 설정·학습 예산·데이터 분할·비교 방법 그리드를 바꾸지 않는다는 전제입니다. 튜닝은 `lt` 모듈 설정만 바꾸므로 이 전제가 유지됩니다.

**A. 그대로 유효 (확정 결과)**
- `exp2_validation.csv`, `exp2_selection.csv`, `selection_shortlist.json`, `selection_locks.json` 전체 (비교 방법 전용)
- `final_test_mean_std.csv`에서 `Method` ∈ {none, revin, san, ddn, fan}인 280행
- `exp3_comparison.csv`의 `none_*`, `revin_*`, `san_*`, `ddn_*`, `fan_*` 열
- `stage_status.csv`, `progress.md`의 `exp2-*`, `exp3-none`, `exp3-tuned` 행
- 이 결과로 가능한 분석
  - 비교 방법끼리의 비교 (예: 튜닝된 SAN vs DDN, 각 방법의 NoNorm 대비 개선폭)
  - 비교 방법 하이퍼파라미터의 민감도와 선택 분포
  - NoNorm 기준선 성능

**B. 조건부 유효**
- `exp1_lightnorm.csv`, `final_test_mean_std.csv`의 `lt` 행, `exp3_comparison.csv`의 `lt_*` 열
- 이 값들은 "주어진 고정 설정의 LightNorm(튜닝 전)" 결과로서 계속 유효합니다. 튜닝 전후 비교나 ablation의 기준선으로 쓸 수 있습니다.
- 튜닝에서 lock된 설정이 현재 설정과 같은 케이스는 RunID가 같으므로 기존 셀이 그대로 최종 결과가 됩니다.
- 예외: ETTm1 × DLinear는 use_mlp 규칙(1→0) 때문에 어떤 경우에도 새 final 셀로 대체됩니다.

**C. 튜닝 결과에 따라 바뀜**
- `exp3_comparison.csv`의 `lt_*` 열 중 설정이 바뀐 케이스, `Best_MSE`, 그리고 LightNorm 대비 승률·순위
- `exp5_*` 파일은 validation 전용 중간 결과입니다. 축 결정 근거로만 쓰고 성능 보고에는 쓰지 않습니다.
- LightNorm의 파라미터 수·FLOPs는 다음 경우에만 바뀝니다.
  - down_ratio가 탐색 축에 들어갈 때
  - ETTm1 × DLinear: use_mlp 1→0. 파라미터가 소폭 바뀝니다.

  s_norm, kernel_size, station_lr은 파라미터 수를 바꾸지 않습니다.

---

## 7. 해석 시 주의점

- **validation MSE는 선택에만 씁니다.** 같은 케이스·H 안에서 후보끼리 비교하는 용도입니다. 분할 구간의 분포 차이 때문에 test와 크기가 다릅니다. ETTh1·ETTm1은 validation이 test보다 크고, ETTh2·ETTm2는 작습니다. 성능 보고에는 Exp3의 test 값을 사용합니다.
- **192·336 일반화:** lock은 H 96·720으로만 정하고 192·336에 그대로 적용했습니다.
- **튜닝 조건의 비대칭:** 비교 방법은 후보 최대 6개 × 2단계 선택을 거쳤고, LightNorm은 주어진 설정을 그대로 썼습니다. 백본·LightNorm 설정의 출처는 `base_configs.json`의 `source` 열(`run_multiseed_best.py`)입니다. 이 설정을 고른 기준이 validation이었는지는 이 저장소에 기록되어 있지 않습니다.
- **DDN 통계 예측 MLP 크기**는 튜닝하지 않았습니다. 공식 스크립트에는 데이터셋별로 다른 값을 쓰는 경우가 있습니다.
- **실행 환경 혼재:** A100(torch 2.1.0+cu121), RTX A4000(2.1.0+cu118), RTX PRO 6000 Blackwell(2.8.0+cu128)에서 나눠 실행했습니다. 셀별 `GPU`, `Torch` 열로 확인할 수 있습니다.
