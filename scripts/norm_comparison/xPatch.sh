#!/bin/bash
# =============================================================================
# Normalizer comparison on a FIXED xPatch backbone.
#
# Purpose: swap ONLY the normalization module (none / revin / san / ddn / lt)
# while every backbone- and training-side parameter stays identical, so that
# result differences are attributable to the normalizer alone. 3 runs each
# (--itr 3).
#
# Backbone settings follow the official stitsyuk/xPatch unified-setting script
# (scripts/xPatch_unified.sh): patch_len 16 / stride 8, ma_type ema with
# alpha 0.3 / beta 0.3, lradj 'sigmoid', train_epochs 100, patience 10,
# per-dataset lr/batch (ETTh1/h2/m1/weather lr 5e-4 b2048; ETTm2 lr 1e-4
# b2048; electricity lr 5e-3 b256; traffic lr 5e-3 b96), except:
#   - seq_len 720 (framework-wide protocol; official script uses 96)
#   - training loss stays MSE (framework-wide; official xPatch trains with L1)
# The station (normalizer) optimizer always follows type1 regardless of
# --lradj (see utils/tools.py), so 'sigmoid' only shapes backbone training.
# NOTE: xPatch's internal RevIN is permanently disabled in models/xPatch.py —
# normalization is done exclusively by --use_norm.
#
# Normalizer-side settings are fixed at the original modules' defaults:
# station_lr 1e-4, pre_epoch 5, SAN period_len 24, DDN kernel_len 25 /
# hkernel_len 5 / twice_epoch 1, LightNorm t_norm 1 / s_norm 0 / kernel 25.
# d_model/d_ff are unused by xPatch (kept only for the setting string).
# =============================================================================

mkdir -p ./logs/norm_comp/xPatch ./results

gpu=0
features=M
model_name=xPatch
seq_len=720
label_len=168
lradj=sigmoid
itr=3
train_epochs=100
patience=10
patch_len=16
stride=8

#            ETTh1        ETTh2        ETTm1        ETTm2        electricity  traffic      weather
datasets=(   ETTh1        ETTh2        ETTm1        ETTm2        custom       custom       custom      )
data_paths=( ETTh1.csv    ETTh2.csv    ETTm1.csv    ETTm2.csv    electricity.csv traffic.csv weather.csv )
root_paths=( ./datasets/ETT-small ./datasets/ETT-small ./datasets/ETT-small ./datasets/ETT-small ./datasets ./datasets ./datasets )
tags=(       eh1          eh2          em1          em2          elc          tra          wea         )
enc_ins=(    7            7            7            7            321          862          21          )
batches=(    2048         2048         2048         2048         256          96           2048        )
lrs=(        0.0005       0.0005       0.0005       0.0001       0.005        0.005        0.0005      )

for use_norm in none revin san ddn lt; do
  for i in "${!datasets[@]}"; do
    data=${datasets[$i]}; data_path=${data_paths[$i]}; root_path=${root_paths[$i]}
    tag=${tags[$i]}; enc_in=${enc_ins[$i]}
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
        --d_model 128 \
        --d_ff 128 \
        --e_layers 2 \
        --patch_len $patch_len \
        --stride $stride \
        --padding_patch end \
        --ma_type ema \
        --alpha 0.3 \
        --beta 0.3 \
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
        --result_file ./results/norm_comp_$model_name.csv \
        >logs/norm_comp/$model_name/${use_norm}_${tag}_$pred_len.log
    done
  done
done
