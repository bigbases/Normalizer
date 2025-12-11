if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/SAN" ]; then
  mkdir ./logs/SAN
fi

if [ ! -d "./logs/SAN/DLinear_96" ]; then
  mkdir ./logs/SAN/DLinear_96
fi

if [ ! -d "./logs/SAN/iTransformer_96" ]; then
  mkdir ./logs/SAN/iTransformer_96
fi

gpu=0
use_norm=san
features=M
seq_len=96
label_len=48

for model_name in DLinear; do
  for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path exchange.csv \
      --model_id $use_norm'_'exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 6 \
      --learning_rate 0.001 \
      --station_lr 0.001 \
      --itr 3 >logs/SAN/$model_name'_96'/exch_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id $use_norm'_'electricity_96_$pred_len \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --learning_rate 0.001 \
      --station_lr 0.0001 \
      --itr 3 >logs/SAN/$model_name'_96'/elec_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id $use_norm'_'traffic_96_$pred_len \
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
      --itr 3 \
      --gpu $gpu \
      --period_len 24 \
      --learning_rate 0.05 \
      --station_lr 0.0001 \
      --use_norm $use_norm >logs/SAN/$model_name'_96'/traf_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id $use_norm'_'weather_96_$pred_len \
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
      --itr 3 \
      --gpu $gpu \
      --period_len 12 \
      --learning_rate 0.0001 \
      --station_lr 0.0001 \
      --use_norm $use_norm >logs/SAN/$model_name'_96'/wea_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh1.csv \
      --model_id $use_norm'_'ETTh1_96_$pred_len \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --learning_rate 0.005 \
      --station_lr 0.0001 \
      --itr 3 >logs/SAN/$model_name'_96'/eh1_$pred_len.log
    #
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id $use_norm'_'ETTh2_96_$pred_len \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --learning_rate 0.05 \
      --station_lr 0.00005 \
      --itr 3 >logs/SAN/$model_name'_96'/eh2_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm1.csv \
      --model_id $use_norm'_'ETTm1_96_$pred_len \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 12 \
      --learning_rate 0.0001 \
      --station_lr 0.0001 \
      --itr 3 >logs/SAN/$model_name'_96'/em1_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/ETT-small \
      --data_path ETTm2.csv \
      --model_id $use_norm'_'ETTm2_96_$pred_len \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 12 \
      --learning_rate 0.01 \
      --station_lr 0.00005 \
      --itr 3 >logs/SAN/$model_name'_96'/em2_$pred_len.log
  done
done

for model_name in iTransformer; do
  for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets \
      --data_path exchange.csv \
      --model_id $use_norm'_'exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 6 \
      --station_lr 0.001 \
      --itr 1 >logs/SAN/$model_name'_96'/exch_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --itr 1 >logs/SAN/$model_name'_96'/elec_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --itr 1 >logs/SAN/$model_name'_96'/traf_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 12 \
      --itr 1 >logs/SAN/$model_name'_96'/wea_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --itr 1 >logs/SAN/$model_name'_96'/eh1_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 24 \
      --itr 1 >logs/SAN/$model_name'_96'/eh2_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 12 \
      --itr 1 >logs/SAN/$model_name'_96'/em1_$pred_len.log

    python -u run_longExp.py \
      --is_training 1 \
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
      --gpu $gpu \
      --use_norm $use_norm \
      --period_len 12 \
      --itr 1 >logs/SAN/$model_name'_96'/em2_$pred_len.log
  done
done
