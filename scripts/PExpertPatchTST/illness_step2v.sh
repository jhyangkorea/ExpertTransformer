if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=ExpertPatchTST_Step2

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
for random_seed in 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
do
for pred_len in 60
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 50\
      --lradj 'constant'\
      --itr 1 --batch_size 16 --learning_rate 0.0025 >logs/LongForecasting/P$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed.log\
      --num_attention_experts 2\
      --num_ffn_experts 2\
      --num_active_experts 1\
      --quantile 0.5\
      --percentile_avg_bool True\
      --z_weight 0.1\
      --curricular_bool ""\
      --gid P

done
done
