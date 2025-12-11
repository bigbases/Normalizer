if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/validation" ]; then
  mkdir ./logs/validation
fi

if [ ! -d "./logs/validation/DLinear" ]; then
  mkdir ./logs/validation/DLinear
fi

gpu=0
features=M
model_name=DLinear

# for pred_len in 96 192 336 720; do
#   CUDA_VISIBLE_DEVICES=$gpu \
#   python -u run_longExp.py \
#     --is_training 1 \
#     --use_norm $use_norm \
#     --root_path ./datasets \
#     --data_path electricity.csv \
#     --model_id $use_norm'_'electricity_336_$pred_len$model_name \
#     --model $model_name \
#     --data custom \
#     --features $features \
#     --seq_len 336 \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --enc_in 321 \
#     --dec_in 321 \
#     --c_out 321 \
#     --des 'Exp' \
#     --learning_rate 0.001 \
#     --station_lr 0.0001 \
#     --period_len 24 \
#     --itr 1 >logs/validation/$model_name/elc_$pred_len.log
#   done

# for pred_len in 96 192 336 720; do
#   CUDA_VISIBLE_DEVICES=$gpu \
#   python -u run_longExp.py \
#     --is_training 1 \
#     --use_norm $use_norm \
#     --root_path ./datasets \
#     --data_path traffic.csv \
#     --model_id $use_norm'_'traffic_336_$pred_len$model_name \
#     --model $model_name \
#     --data custom \
#     --features $features \
#     --seq_len 336 \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --enc_in 862 \
#     --dec_in 862 \
#     --c_out 862 \
#     --des 'Exp' \
#     --itr 1 \
#     --period_len 24 \
#     --learning_rate 0.0005 >logs/validation/$model_name/traf_$pred_len.log
#   done

for pred_len in 96 192 336 720; do
  for use_norm in tp base2 base3 base4; do
    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --use_norm $use_norm \
      --root_path ./datasets \
      --data_path weather.csv \
      --model_id $use_norm'_'weather_336_$pred_len$model_name \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 336 \
      --label_len 168 \
      --pred_len $pred_len \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --itr 1 \
      --period_len 12 >logs/validation/$model_name/wea_$use_norm'_'$pred_len.log
  done
done

# for pred_len in 96 192 336 720; do
#   for use_norm in tp base2 base3 base4; do
#     CUDA_VISIBLE_DEVICES=$gpu \
#     python -u run_longExp.py \
#       --is_training 1 \
#       --use_norm $use_norm \
#       --root_path ./datasets/ETT-small \
#       --data_path ETTh1.csv \
#       --model_id $use_norm'_'ETTh1_336_$pred_len$model_name \
#       --model $model_name \
#       --data ETTh1 \
#       --features $features \
#       --seq_len 336 \
#       --label_len 168 \
#       --pred_len $pred_len \
#       --enc_in 7 \
#       --dec_in 7 \
#       --c_out 7 \
#       --d_ff 128 \
#       --d_model 128 \
#       --e_layers 2 \
#       --des 'Exp' \
#       --period_len 24 \
#       --twice_epoch 3 \
#       --itr 1 >logs/validation/$model_name/eh1_$use_norm'_'$pred_len.log
#   done
# done

# for pred_len in 96 192 336 720; do
#   CUDA_VISIBLE_DEVICES=$gpu \
#   python -u run_longExp.py \
#     --is_training 1 \
#     --use_norm $use_norm \
#     --root_path ./datasets/ETT-small \
#     --data_path ETTh2.csv \
#     --model_id $use_norm'_'ETTh2_336_$pred_len$model_name \
#     --model $model_name \
#     --data ETTh2 \
#     --features $features \
#     --seq_len 336 \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --period_len 24 \
#     --itr 1 >logs/validation/$model_name/eh2_$pred_len.log
#   done

# for pred_len in 96 192 336 720; do
#   CUDA_VISIBLE_DEVICES=$gpu \
#   python -u run_longExp.py \
#     --is_training 1 \
#     --use_norm $use_norm \
#     --root_path ./datasets/ETT-small \
#     --data_path ETTm1.csv \
#     --model_id $use_norm'_'ETTm1_336_$pred_len$model_name \
#     --model $model_name \
#     --data ETTm1 \
#     --features $features \
#     --seq_len 336 \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --period_len 12 \
#     --itr 1 >logs/validation/$model_name/em1_$pred_len.log
#   done

# for pred_len in 96 192 336 720; do
#   CUDA_VISIBLE_DEVICES=$gpu \
#   python -u run_longExp.py \
#     --is_training 1 \
#     --use_norm $use_norm \
#     --root_path ./datasets/ETT-small \
#     --data_path ETTm2.csv \
#     --model_id $use_norm'_'ETTm2_336_$pred_len$model_name \
#     --model $model_name \
#     --data ETTm2 \
#     --features $features \
#     --seq_len 336 \
#     --label_len 168 \
#     --pred_len $pred_len \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --period_len 12 \
#     --pd_ff 32 \
#     --pd_model 32 \
#     --itr 1 >logs/validation/$model_name/em2_$pred_len.log
#   done