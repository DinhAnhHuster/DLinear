
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=5
model_name=DLinear

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'5 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 5\
  --enc_in 3\
  --des 'Exp' \
  --itr 1 --batch_size 2 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'5.log 


python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'5 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 5\
  --enc_in 3\
  --des 'Exp' \
  --itr 1 --batch_size 4 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'5.log 

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'5 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --pred_len 5\
  --enc_in 3\
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'5.log 
# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path exchange_rate.csv \
#   --model_id Exchange_$seq_len'_'6 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --pred_len 6 \
#   --enc_in 8 \
#   --des 'Exp' \
#   --itr 1 --batch_size 8 --learning_rate 0.0005 >logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'6.log 

