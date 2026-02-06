#!/usr/bin/env bash
set -euo pipefail

# notify 로드 (DLinear.sh는 ./scripts/LightTrend/ 에 있으니 상위로)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../notify.sh"

EXP_TAG="LightTrend-DLinear_96"
START_TIME="$(date '+%F %T')"

send_discord "🚀 START | ${EXP_TAG} [${START_TIME}]"

# 에러 즉시 알림 (어느 줄/명령에서 죽었는지)
trap 'send_discord "❌ CRASH | ${EXP_TAG}\nhost=${HOST}\nline=${LINENO}\ncmd=${BASH_COMMAND}"; exit 1' ERR


if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LightTrend" ]; then
  mkdir ./logs/LightTrend
fi

if [ ! -d "./logs/LightTrend/DLinear_96" ]; then
  mkdir ./logs/LightTrend/DLinear_96
fi

gpu=0
features=M
model_name=DLinear
use_norm=lt
use_mlp=0

for learning_rate in 0.0001; do
  for station_pre_lr in 0.0001; do
    for t_ff in 64; do
      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset \
          --data_path electricity.csv \
          --model_id $use_norm'_'electricity_96_$pred_len$model_name \
          --model $model_name \
          --data custom \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --learning_rate $learning_rate \
          --itr 1 \
          --num_workers 0 \
          --t_ff $t_ff \
          --station_pre_lr $station_pre_lr \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/elc_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | electricity"

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset \
          --data_path weather.csv \
          --model_id $use_norm'_'weather_96_$pred_len$model_name \
          --model $model_name \
          --data custom \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --kernel_size 13 \
          --des 'Exp' \
          --itr 1 \
          --t_ff $t_ff \
          --learning_rate $learning_rate \
          --station_pre_lr $station_pre_lr \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/wea_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | weather"

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset \
          --data_path traffic.csv \
          --model_id $use_norm'_'traffic_96_$pred_len$model_name \
          --model $model_name \
          --data custom \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 862 \
          --dec_in 862 \
          --c_out 862 \
          --des 'Exp' \
          --itr 1 \
          --num_workers 0 \
          --learning_rate $learning_rate \
          --t_ff $t_ff \
          --station_pre_lr $station_pre_lr \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/tra_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | traffic"

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset/ETT-small \
          --data_path ETTh1.csv \
          --model_id $use_norm'_'ETTh1_96_$pred_len$model_name \
          --model $model_name \
          --data ETTh1 \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --t_ff $t_ff \
          --itr 1 \
          --station_pre_lr $station_pre_lr \
          --learning_rate $learning_rate \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/eh1_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | ETTh1"

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset/ETT-small \
          --data_path ETTh2.csv \
          --model_id $use_norm'_'ETTh2_96_$pred_len$model_name \
          --model $model_name \
          --data ETTh2 \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --itr 1 \
          --t_ff $t_ff \
          --station_pre_lr $station_pre_lr \
          --learning_rate $learning_rate \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/eh2_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | ETTh2"

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset/ETT-small \
          --data_path ETTm1.csv \
          --model_id $use_norm'_'ETTm1_96_$pred_len$model_name \
          --model $model_name \
          --data ETTm1 \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --kernel_size 13 \
          --itr 1 \
          --t_ff $t_ff \
          --station_pre_lr $station_pre_lr \
          --learning_rate $learning_rate \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/em1_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | ETTm1"

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./dataset/ETT-small \
          --data_path ETTm2.csv \
          --model_id $use_norm'_'ETTm2_96_$pred_len$model_name \
          --model $model_name \
          --data ETTm2 \
          --features $features \
          --seq_len 96 \
          --label_len 48 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --kernel_size 13 \
          --itr 1 \
          --t_ff $t_ff \
          --station_pre_lr $station_pre_lr \
          --learning_rate $learning_rate \
          --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/em2_$pred_len.log
        done
        send_discord "✅ DATASET DONE | ${EXP_TAG} | ETTm2"
      done
    done
  done

END_TIME="$(date '+%F %T')"
send_discord "🏁 ALL DONE | ${EXP_TAG} [${END_TIME}]"
