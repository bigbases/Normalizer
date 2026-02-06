if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LightTrend" ]; then
  mkdir ./logs/LightTrend
fi

if [ ! -d "./logs/LightTrend/iTransformer_96" ]; then
  mkdir ./logs/LightTrend/iTransformer_96
fi

gpu=0
features=M
model_name=iTransformer
use_norm=lt
use_mlp=0
t_ff=128

for station_joint_lr in 0.0001; do
  for station_pre_lr in 0.0001; do
    # for pred_len in 96 192 336 720; do
    #   CUDA_VISIBLE_DEVICES=$gpu \
    #   python -u run.py \
    #     --is_training 1 \
    #     --use_norm $use_norm \
    #     --root_path ./dataset \
    #     --data_path electricity.csv \
    #     --model_id $use_norm'_'electricity_720_$pred_len$model_name \
    #     --model $model_name \
    #     --data custom \
    #     --features $features \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len $pred_len \
    #     --enc_in 321 \
    #     --dec_in 321 \
    #     --c_out 321 \
    #     --d_ff 512 \
    #     --d_model 512 \
    #     --e_layers 3 \
    #     --des 'Exp' \
    #     --learning_rate 0.0005 \
    #     --batch_size 16 \
    #     --itr 1 \
    #     --t_ff $t_ff \
    #     --station_joint_lr $station_joint_lr \
    #     --station_pre_lr $station_pre_lr \
    #     --num_workers 0 \
    #     --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/elc_$pred_len.log
    #   done

    # for pred_len in 96 192 336 720; do
    #   CUDA_VISIBLE_DEVICES=$gpu \
    #   python -u run.py \
    #     --is_training 1 \
    #     --use_norm $use_norm \
    #     --root_path ./dataset \
    #     --data_path traffic.csv \
    #     --model_id $use_norm'_'traffic_720_$pred_len$model_name \
    #     --model $model_name \
    #     --data custom \
    #     --features $features \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len $pred_len \
    #     --enc_in 862 \
    #     --dec_in 862 \
    #     --c_out 862 \
    #     --d_ff 512 \
    #     --d_model 512 \
    #     --e_layers 4 \
    #     --des 'Exp' \
    #     --itr 1 \
    #     --t_ff $t_ff \
    #     --station_joint_lr $station_joint_lr \
    #     --station_pre_lr $station_pre_lr \
    #     --learning_rate 0.001 \
    #     --use_mlp $use_mlp \
    #     --num_workers 0 \
    #     --batch_size 16 >logs/LightTrend/$model_name'_96'/tra_$pred_len.log
    #   done

    # for pred_len in 96 192 336 720; do
    #   CUDA_VISIBLE_DEVICES=$gpu \
    #   python -u run.py \
    #     --is_training 1 \
    #     --use_norm $use_norm \
    #     --root_path ./dataset \
    #     --data_path weather.csv \
    #     --model_id $use_norm'_'weather_720_$pred_len$model_name \
    #     --model $model_name \
    #     --data custom \
    #     --features $features \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len $pred_len \
    #     --enc_in 21 \
    #     --dec_in 21 \
    #     --c_out 21 \
    #     --d_ff 512 \
    #     --d_model 512 \
    #     --e_layers 3 \
    #     --des 'Exp' \
    #     --itr 1 \
    #     --t_ff $t_ff \
    #     --learning_rate 0.0001 \
    #     --station_joint_lr $station_joint_lr \
    #     --station_pre_lr $station_pre_lr \
    #     --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/wea_$pred_len.log
    #   done

    for pred_len in 96 192 336 720; do
      CUDA_VISIBLE_DEVICES=$gpu \
      python -u run.py \
        --is_training 1 \
        --use_norm $use_norm \
        --root_path ./dataset/ETT-small \
        --data_path ETTh1.csv \
        --model_id $use_norm'_'ETTh1_720_$pred_len$model_name \
        --model $model_name \
        --data ETTh1 \
        --features $features \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_ff 128 \
        --d_model 128 \
        --e_layers 2 \
        --des 'Exp' \
        --itr 1 \
        --t_ff $t_ff \
        --learning_rate 0.0001 \
        --station_joint_lr $station_joint_lr \
        --station_pre_lr $station_pre_lr \
        --s_norm 1 \
        --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/eh1_$pred_len.log
      done

    # for pred_len in 96 192 336 720; do
    #   CUDA_VISIBLE_DEVICES=$gpu \
    #   python -u run.py \
    #     --is_training 1 \
    #     --use_norm $use_norm \
    #     --root_path ./dataset/ETT-small \
    #     --data_path ETTh2.csv \
    #     --model_id $use_norm'_'ETTh2_720_$pred_len$model_name \
    #     --model $model_name \
    #     --data ETTh2 \
    #     --features $features \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len $pred_len \
    #     --enc_in 7 \
    #     --dec_in 7 \
    #     --c_out 7 \
    #     --d_ff 128 \
    #     --d_model 128 \
    #     --e_layers 2 \
    #     --des 'Exp' \
    #     --itr 1 \
    #     --t_ff $t_ff \
    #     --learning_rate 0.0001 \
    #     --station_joint_lr $station_joint_lr \
    #     --station_pre_lr $station_pre_lr \
    #     --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/eh2_$pred_len.log
    #   done

    # for pred_len in 96 192 336 720; do
    #   CUDA_VISIBLE_DEVICES=$gpu \
    #   python -u run.py \
    #     --is_training 1 \
    #     --use_norm $use_norm \
    #     --root_path ./dataset/ETT-small \
    #     --data_path ETTm1.csv \
    #     --model_id $use_norm'_'ETTm1_720_$pred_len$model_name \
    #     --model $model_name \
    #     --data ETTm1 \
    #     --features $features \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len $pred_len \
    #     --enc_in 7 \
    #     --dec_in 7 \
    #     --c_out 7 \
    #     --d_ff 128 \
    #     --d_model 128 \
    #     --e_layers 2 \
    #     --des 'Exp' \
    #     --itr 1 \
    #     --t_ff $t_ff \
    #     --learning_rate 0.0001 \
    #     --station_joint_lr $station_joint_lr \
    #     --station_pre_lr $station_pre_lr \
    #     --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/em1_$pred_len.log
    #   done

    # for pred_len in 96 192 336 720; do
    #   CUDA_VISIBLE_DEVICES=$gpu \
    #   python -u run.py \
    #     --is_training 1 \
    #     --use_norm $use_norm \
    #     --root_path ./dataset/ETT-small \
    #     --data_path ETTm2.csv \
    #     --model_id $use_norm'_'ETTm2_720_$pred_len$model_name \
    #     --model $model_name \
    #     --data ETTm2 \
    #     --features $features \
    #     --seq_len 96 \
    #     --label_len 48 \
    #     --pred_len $pred_len \
    #     --enc_in 7 \
    #     --dec_in 7 \
    #     --c_out 7 \
    #     --d_ff 128 \
    #     --d_model 128 \
    #     --e_layers 2 \
    #     --des 'Exp' \
    #     --itr 1 \
    #     --t_ff $t_ff \
    #     --learning_rate 0.0001 \
    #     --station_joint_lr $station_joint_lr \
    #     --station_pre_lr $station_pre_lr \
    #     --use_mlp $use_mlp >logs/LightTrend/$model_name'_96'/em2_$pred_len.log
    #   done
    done
  done