if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/SAN" ]; then
  mkdir ./logs/SAN
fi

if [ ! -d "./logs/SAN/Autoformer" ]; then
  mkdir ./logs/SAN/Autoformer
fi

if [ ! -d "./logs/SAN/FEDformer" ]; then
  mkdir ./logs/SAN/FEDformer
fi

if [ ! -d "./logs/SAN/iTransformer" ]; then
  mkdir ./logs/SAN/iTransformer
fi

use_norm=san
features=M
gpu=0

for model_name in Autoformer FEDformer iTransformer; do

    # model_name=iTransformer -> seq_len=720, label_len=168
    if [ $model_name == "iTransformer" ]; then
        seq_len=720
        label_len=168
    else
        seq_len=96
        label_len=48
    fi

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
      --itr 1 >logs/DDN/$model_name/exch_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/elec_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/traf_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/wea_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/eh1_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/eh2_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/em1_$pred_len.log

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
      --itr 1 >logs/DDN/$model_name/em2_$pred_len.log
  done
done

for model_name in Autoformer FEDformer iTransformer; do
  for pred_len in 24 36 48 60; do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path ./datasets/illness \
      --data_path national_illness.csv \
      --model_id $use_norm'_'ili_36_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len 36 \
      --label_len 18 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --period_len 6 \
      --gpu $gpu \
      --use_norm $use_norm \
      --itr 1 >logs/DDN/$model_name/ili_$pred_len.log
  done
done