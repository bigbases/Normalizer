if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
  mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/MICN" ]; then
  mkdir ./logs/LongForecasting/MICN
fi

gpu=0
features=M
model_name=Autoformer
itr=1

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets \
    --data_path electricity.csv \
    --model_id electricity_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --learning_rate 0.0005 \
    --batch_size 16 \
    --period_len 24 \
    --kernel_len 32 \
    --twice_epoch 2 \
    --j 1 \
    --pd_ff 512 \
    --pe_layers 1 \
    --itr $itr >logs/LongForecasting/MICN/$model_name'_elc_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets \
    --data_path traffic.csv \
    --model_id traffic_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr $itr \
    --period_len 24 \
    --kernel_len 24 \
    --hkernel_len 12 \
    --learning_rate 0.001 \
    --twice_epoch 2 \
    --batch_size 16 \
    --j 1 \
    --pd_model 512 \
    --pd_ff 512  >logs/LongForecasting/MICN/$model_name'_tra_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets \
    --data_path weather.csv \
    --model_id weather_336_$pred_len$model_name \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr $itr \
    --period_len 12 \
    --j 1 \
    --twice_epoch 2 \
    --pd_ff 128 \
    --pd_model 128 >logs/LongForecasting/MICN/$model_name'_wea_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --j 1 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 \
    --itr $itr >logs/LongForecasting/MICN/$model_name'_eh1_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTh2.csv \
    --model_id ETTh2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 24 \
    --pe_layers 0 \
    --pd_model 128 \
    --pd_ff 128 \
    --itr $itr >logs/LongForecasting/MICN/$model_name'_eh2_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --kernel_len 12 \
    --j 1 \
    --pe_layers 0 \
    --twice_epoch 3 \
    --itr $itr >logs/LongForecasting/MICN/$model_name'_em1_'$pred_len.log
  done

for pred_len in 96 192 336 720; do
  CUDA_VISIBLE_DEVICES=$gpu \
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./datasets/ETT-small \
    --data_path ETTm2.csv \
    --model_id ETTm2_336_$pred_len$model_name \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --period_len 12 \
    --pe_layers 0 \
    --itr $itr >logs/LongForecasting/MICN/$model_name'_em2_'$pred_len.log
  done
