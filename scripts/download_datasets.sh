#!/usr/bin/env bash
# Download the seven benchmark CSVs into the layout configs/protocol.json expects:
#   $DATA_ROOT/{ETT-small,weather,electricity,traffic}/*.csv
# Source: the Time-Series-Library dataset mirror on Hugging Face. SHA-256 sums
# were recorded on 2026-10-07; a mismatch aborts so results never silently
# come from a different file.
set -euo pipefail

DATA_ROOT="${1:-${DATA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/datasets}}"
BASE_URL="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"

FILES=(
  "ETT-small/ETTh1.csv f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
  "ETT-small/ETTh2.csv a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"
  "ETT-small/ETTm1.csv 6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"
  "ETT-small/ETTm2.csv db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"
  "weather/weather.csv 34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63"
  "electricity/electricity.csv 7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"
  "traffic/traffic.csv cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"
)

mkdir -p "$DATA_ROOT"
for entry in "${FILES[@]}"; do
  rel="${entry%% *}"; want="${entry##* }"; dest="$DATA_ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]] && echo "$want  $dest" | sha256sum -c --status; then
    echo "ok       $rel"; continue
  fi
  echo "download $rel"
  curl -fL --retry 5 --retry-delay 3 -o "$dest.part" "$BASE_URL/$rel"
  echo "$want  $dest.part" | sha256sum -c --status || { echo "SHA-256 mismatch for $rel" >&2; exit 1; }
  mv "$dest.part" "$dest"
done
echo "datasets ready in $DATA_ROOT"
