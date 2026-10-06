#!/bin/bash
# =============================================================================
# Normalizer comparison on a FIXED TimeMixer backbone.
#
# Purpose: swap ONLY the normalization module (none / revin / san / ddn / lt)
# while every backbone- and training-side parameter stays identical, so that
# result differences are attributable to the normalizer alone. 3 runs each
# (--itr 3).
#
# Backbone settings follow the official kwuking/TimeMixer unify scripts
# (d_model/d_ff/e_layers/batch/epochs per dataset; lr 0.01; down_sampling
# 3 layers / window 2 / avg; channel_independence 1; patience 10), except:
#   - seq_len 720 (framework-wide protocol; official scripts use 96)
#   - lradj type1 (official default 'TST' needs a OneCycle scheduler that
#     this framework does not implement)
# NOTE: TimeMixer keeps its internal per-scale normalization ACTIVE
# (as-published backbone — see models/TimeMixer.py). use_norm=none therefore
# equals published TimeMixer, and revin is expected to be ~identical to none.
#
# Normalizer-side settings are fixed at the original modules' defaults:
# station_lr 1e-4, pre_epoch 5, SAN period_len 24, DDN kernel_len 25 /
# hkernel_len 5 / twice_epoch 1, LightNorm t_norm 1 / s_norm 0 / kernel 25.
# =============================================================================

mkdir -p ./logs/norm_comp/TimeMixer ./results

gpu=0
features=M
model_name=TimeMixer
seq_len=720
label_len=168
learning_rate=0.01
lradj=type1
itr=3
patience=10

#            ETTh1        ETTh2        ETTm1        ETTm2        electricity  traffic      weather
datasets=(   ETTh1        ETTh2        ETTm1        ETTm2        custom       custom       custom      )
data_paths=( ETTh1.csv    ETTh2.csv    ETTm1.csv    ETTm2.csv    electricity.csv traffic.csv weather.csv )
root_paths=( ./datasets/ETT-small ./datasets/ETT-small ./datasets/ETT-small ./datasets/ETT-small ./datasets ./datasets ./datasets )
tags=(       eh1          eh2          em1          em2          elc          tra          wea         )
enc_ins=(    7            7            7            7            321          862          21          )
d_models=(   16           16           16           32           16           32           16          )
d_ffs=(      32           32           32           32           32           64           32          )
e_layerss=(  2            2            2            2            3            3            3           )
batches=(    128          16           16           128          32           8            128         )
epochss=(    10           10           10           10           20           20           20          )

for use_norm in none revin san ddn lt; do
  for i in "${!datasets[@]}"; do
    data=${datasets[$i]}; data_path=${data_paths[$i]}; root_path=${root_paths[$i]}
    tag=${tags[$i]}; enc_in=${enc_ins[$i]}
    d_model=${d_models[$i]}; d_ff=${d_ffs[$i]}; e_layers=${e_layerss[$i]}
    batch_size=${batches[$i]}; train_epochs=${epochss[$i]}
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
        --down_sampling_layers 3 \
        --down_sampling_window 2 \
        --down_sampling_method avg \
        --channel_independence 1 \
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
