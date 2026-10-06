if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LightNorm" ]; then
  mkdir ./logs/LightNorm
fi

if [ ! -d "./logs/LightNorm/TimeXer" ]; then
  mkdir ./logs/LightNorm/TimeXer
fi

gpu=0
features=M
model_name=TimeXer
use_norm=lt
use_mlp=0
t_norm=1
patch_len=16

# NOTE (memory): TimeXer embeds per-variate patch tokens (seq_len/patch_len + 1
# tokens per channel), so high-channel datasets need small batches at seq_len
# 720 — electricity(321ch) batch 8, traffic(862ch) batch 4. The official TSLib
# script already uses batch 4 for electricity at seq_len 96. Drop to half if
# OOM occurs; results are batch-size-robust with LayerNorm-only internals.

for station_lr in 0.0001 0.001 0.01; do
  for learning_rate in 0.0001; do
      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets \
          --data_path electricity.csv \
          --model_id $use_norm'_'electricity_720_$pred_len$model_name \
          --model $model_name \
          --data custom \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --d_ff 512 \
          --d_model 512 \
          --e_layers 3 \
          --patch_len $patch_len \
          --des 'Exp' \
          --learning_rate $learning_rate \
          --batch_size 8 \
          --itr 1 \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp >logs/LightNorm/$model_name/elc_$pred_len.log
        done

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets \
          --data_path traffic.csv \
          --model_id $use_norm'_'traffic_720_$pred_len$model_name \
          --model $model_name \
          --data custom \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 862 \
          --dec_in 862 \
          --c_out 862 \
          --d_ff 512 \
          --d_model 512 \
          --e_layers 4 \
          --patch_len $patch_len \
          --des 'Exp' \
          --itr 1 \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp \
          --learning_rate $learning_rate \
          --batch_size 4 >logs/LightNorm/$model_name/tra_$pred_len.log
        done

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets \
          --data_path weather.csv \
          --model_id $use_norm'_'weather_720_$pred_len$model_name \
          --model $model_name \
          --data custom \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --d_ff 512 \
          --d_model 512 \
          --e_layers 3 \
          --patch_len $patch_len \
          --des 'Exp' \
          --itr 1 \
          --learning_rate $learning_rate \
          --batch_size 32 \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp >logs/LightNorm/$model_name/wea_$pred_len.log
        done

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets/ETT-small \
          --data_path ETTh1.csv \
          --model_id $use_norm'_'ETTh1_720_$pred_len$model_name \
          --model $model_name \
          --data ETTh1 \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --d_ff 128 \
          --d_model 128 \
          --e_layers 2 \
          --patch_len $patch_len \
          --des 'Exp' \
          --itr 1 \
          --learning_rate $learning_rate \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp >logs/LightNorm/$model_name/eh1_$pred_len.log
        done

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets/ETT-small \
          --data_path ETTh2.csv \
          --model_id $use_norm'_'ETTh2_720_$pred_len$model_name \
          --model $model_name \
          --data ETTh2 \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --d_ff 128 \
          --d_model 128 \
          --e_layers 2 \
          --patch_len $patch_len \
          --des 'Exp' \
          --itr 1 \
          --learning_rate $learning_rate \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp >logs/LightNorm/$model_name/eh2_$pred_len.log
        done

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets/ETT-small \
          --data_path ETTm1.csv \
          --model_id $use_norm'_'ETTm1_720_$pred_len$model_name \
          --model $model_name \
          --data ETTm1 \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --d_ff 128 \
          --d_model 128 \
          --e_layers 2 \
          --patch_len $patch_len \
          --des 'Exp' \
          --itr 1 \
          --learning_rate $learning_rate \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp >logs/LightNorm/$model_name/em1_$pred_len.log
        done

      for pred_len in 96 192 336 720; do
        CUDA_VISIBLE_DEVICES=$gpu \
        python -u run_longExp.py \
          --is_training 1 \
          --use_norm $use_norm \
          --root_path ./datasets/ETT-small \
          --data_path ETTm2.csv \
          --model_id $use_norm'_'ETTm2_720_$pred_len$model_name \
          --model $model_name \
          --data ETTm2 \
          --features $features \
          --seq_len 720 \
          --label_len 168 \
          --pred_len $pred_len \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --d_ff 128 \
          --d_model 128 \
          --e_layers 2 \
          --patch_len $patch_len \
          --des 'Exp' \
          --itr 1 \
          --learning_rate $learning_rate \
          --t_norm $t_norm \
          --station_lr $station_lr \
          --use_mlp $use_mlp >logs/LightNorm/$model_name/em2_$pred_len.log
        done

    done
  done
