if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/DDN" ]; then
  mkdir ./logs/DDN
fi

if [ ! -d "./logs/DDN/DLinear_96" ]; then
  mkdir ./logs/DDN/DLinear_96
fi

gpu=0
features=M
model_name=DLinear
use_norm=ddn

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets \
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
    --learning_rate 0.001 \
    --station_lr 0.0001 \
    --period_len 24 \
    --j 1 \
    --pd_ff 512 \
    --pd_model 256 \
    --itr 1 >logs/DDN/DLinear_96/elc_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets \
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
    --des 'Exp' \
    --itr 1 \
    --period_len 12 \
    --j 1 \
    --twice_epoch 2 \
    --pd_ff 128 \
    --pd_model 128 >logs/DDN/DLinear_96/wea_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets \
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
    --period_len 24 \
    --learning_rate 0.0005 \
    --j 1 \
    --kernel_len 12 >logs/DDN/DLinear_96/tra_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
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
    --period_len 24 \
    --j 1 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 >logs/DDN/DLinear_96/eh1_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
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
    --period_len 24 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 \
    --itr 1 >logs/DDN/DLinear_96/eh2_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
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
    --period_len 12 \
    --j 1 \
    --kernel_len 12 \
    --pd_ff 128 \
    --pd_model 128 \
    --itr 1 >logs/DDN/DLinear_96/em1_$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --use_norm $use_norm \
    --root_path ./datasets/ETT-small \
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
    --period_len 12 \
    --pd_ff 32 \
    --pd_model 32 \
    --itr 1 >logs/DDN/DLinear_96/em2_$pred_len.log
  done