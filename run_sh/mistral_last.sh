n_ahead_talk_global=1
n_ahead_global=1
mode="last_rl"
lr=1e-6
data_path=quiet_infer/mistral/12_4_base_1e-6-16_8_50_base_30
data_name=12_4_base_1e-6-16_8_50_base_30
model_name=mistral_base

output_dir=outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}_${model_name}_${data_name}
mkdir -p $output_dir
python -u quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --data_path $data_path \
    --output_dir $output_dir \
    --model_name Mistral-7B-v0.1 \
    --lr $lr > $output_dir/train_log 2>&1
bash run_sh/eval_last.sh $output_dir 1


