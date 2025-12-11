if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/TP" ]; then
  mkdir ./logs/TP
fi

if [ ! -d "./logs/TP/FEDformer" ]; then
  mkdir ./logs/TP/FEDformer
fi

seq_len=96
label_len=48
features=M
gpu=0
model_name=FEDformer
use_norm=tp

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets \
    --data_path traffic.csv \
    --model_id $use_norm'_'traffic_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --period_len 24 \
    --itr 1 >logs/TP/$model_name/traf_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets \
    --data_path electricity.csv \
    --model_id $use_norm'_'electricity_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --period_len 24 \
    --itr 1 >logs/TP/$model_name/elec_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets \
    --data_path weather.csv \
    --model_id $use_norm'_'weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --period_len 12 \
    --itr 1 >logs/TP/$model_name/wea_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id $use_norm'_'ETTh1_96_$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --itr 1 >logs/TP/$model_name/ETTh1_$pred_len.log
  done

for pred_len in 720 96 192 336; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
    --data_path ETTh2.csv \
    --model_id $use_norm'_'ETTh2_96_$pred_len \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --gpu 0 \
    --period_len 24 \
    --itr 1 >logs/TP/$model_name/ETTh2_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
    --data_path ETTm1.csv \
    --model_id $use_norm'_'ETTm1_96_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --gpu 0 \
    --period_len 12 \
    --itr 1 >logs/TP/$model_name/ETTm1_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
    --data_path ETTm2.csv \
    --model_id $use_norm'_'ETTm2_96_$pred_len \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --dr 0.3 \
    --itr 1 >logs/TP/$model_name/ETTm2_$pred_len.log
  done
