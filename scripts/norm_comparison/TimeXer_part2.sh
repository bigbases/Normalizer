#!/bin/bash
# =============================================================================
# [part2: ddn lt] — split for parallel execution.
# Set GPU env var to pick the device (default 0): GPU=1 nohup bash <this>.sh &
# Sends a Discord message via $DISCORD_WEBHOOK_URL (skipped if unset) after
# each (use_norm, dataset) finishes its full pred_len cycle.
# Normalizer comparison on a FIXED TimeXer backbone.
#
# Purpose: swap ONLY the normalization module (none / revin / san / ddn / lt)
# while every backbone- and training-side parameter stays identical, so that
# result differences are attributable to the normalizer alone. 3 runs each
# (--itr 3).
#
# Backbone settings follow the official TSLib TimeXer scripts (modal values
# per dataset, since the official scripts vary some values per pred_len:
# ETT d_model 256 / d_ff 1024|512, ECL/Traffic d_model 512 / d_ff 512,
# lr 1e-4 except traffic 1e-3, patch_len 16, epochs 10, patience 3,
# lradj type1), except:
#   - seq_len 720 (framework-wide protocol; official scripts use 96)
#   - traffic batch 4 (official 16 at seq_len 96; reduced for seq_len 720
#     memory — TimeXer embeds per-variate patch tokens, 862ch is heavy)
# NOTE: TimeXer's internal z-score normalization is REMOVED in
# models/TimeXer.py — normalization is done exclusively by --use_norm.
#
# Normalizer-side settings are fixed at the original modules' defaults:
# station_lr 1e-4, pre_epoch 5, SAN period_len 24, DDN kernel_len 25 /
# hkernel_len 5 / twice_epoch 1, LightNorm t_norm 1 / s_norm 0 / kernel 25.
# =============================================================================

mkdir -p ./logs/norm_comp/TimeXer ./results

# Discord notification via webhook URL from the environment.
# export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..." before running.
notify() {
  if [ -n "$DISCORD_WEBHOOK_URL" ]; then
    curl -s -X POST -H "Content-Type: application/json" \
      -d "{\"content\": \"$1\"}" "$DISCORD_WEBHOOK_URL" > /dev/null 2>&1 || true
  fi
}
SCRIPT_TAG="TimeXer-part2"

gpu=${GPU:-0}
features=M
model_name=TimeXer
seq_len=720
label_len=168
lradj=type1
itr=3
train_epochs=10
patience=3
patch_len=16

#            ETTh1        ETTh2        ETTm1        ETTm2        electricity  traffic      weather
datasets=(   ETTh1        ETTh2        ETTm1        ETTm2        custom       custom       custom      )
data_paths=( ETTh1.csv    ETTh2.csv    ETTm1.csv    ETTm2.csv    electricity.csv traffic.csv weather.csv )
root_paths=( ./datasets/ETT-small ./datasets/ETT-small ./datasets/ETT-small ./datasets/ETT-small ./datasets ./datasets ./datasets )
tags=(       eh1          eh2          em1          em2          elc          tra          wea         )
enc_ins=(    7            7            7            7            321          862          21          )
d_models=(   256          256          256          256          512          512          256         )
d_ffs=(      1024         1024         512          1024         512          512          512         )
e_layerss=(  1            2            1            1            3            3            1           )
batches=(    16           16           4            16           4            4            4           )
lrs=(        0.0001       0.0001       0.0001       0.0001       0.0001       0.001        0.0001      )

for use_norm in ddn lt; do
  for i in "${!datasets[@]}"; do
    data=${datasets[$i]}; data_path=${data_paths[$i]}; root_path=${root_paths[$i]}
    tag=${tags[$i]}; enc_in=${enc_ins[$i]}
    d_model=${d_models[$i]}; d_ff=${d_ffs[$i]}; e_layers=${e_layerss[$i]}
    batch_size=${batches[$i]}; learning_rate=${lrs[$i]}
    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run_longExp.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${use_norm}_${tag}_${seq_len}_${pred_len}_$model_name \
        --model $model_name \
        --data $data \
        --features $features \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --dec_in $enc_in \
        --c_out $enc_in \
        --d_model $d_model \
        --d_ff $d_ff \
        --e_layers $e_layers \
        --n_heads 8 \
        --patch_len $patch_len \
        --factor 3 \
        --learning_rate $learning_rate \
        --lradj $lradj \
        --train_epochs $train_epochs \
        --patience $patience \
        --batch_size $batch_size \
        --itr $itr \
        --des 'NormComp' \
        --station_lr 0.0001 \
        --pre_epoch 5 \
        --period_len 24 \
        --kernel_len 25 \
        --hkernel_len 5 \
        --twice_epoch 1 \
        --t_norm 1 \
        --s_norm 0 \
        --use_mlp 0 \
        --kernel_size 25 \
        --down_ratio 4 \
        --t_ff 64 \
        --affine 1 \
        --result_file ./results/norm_comp_${model_name}_part2.csv \
        >logs/norm_comp/$model_name/${use_norm}_${tag}_$pred_len.log
    done
    notify "[$SCRIPT_TAG] $use_norm/$tag done (pred_len 96-720, itr x3) - $(date '+%m/%d %H:%M')"
  done
done
notify "[$SCRIPT_TAG] ALL DONE - $(date '+%m/%d %H:%M')"
