n_ahead_talk_global=8
n_ahead_global=16
model_name=Mistral-7B-v0.1
mode="base"
lr=2e-6
output_dir=fast-quiet-star/outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}
mkdir -p $output_dir
python -u quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --output_dir $output_dir \
    --model_name ${model_name} \
    --lr $lr > $output_dir/train_log 2>&1

bash fast-quiet-star/run_sh/eval.sh ${output_dir} ${n_ahead_global}


n_ahead_talk_global=4
n_ahead_global=12
model_name=16_8_CKPT
mode="base"
lr=1e-6
output_dir=fast-quiet-star/outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}
mkdir -p $output_dir
python -u quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --output_dir $output_dir \
    --model_name ${model_name} \
    --lr $lr > $output_dir/train_log 2>&1

bash fast-quiet-star/run_sh/eval.sh ${output_dir} ${n_ahead_global}



n_ahead_talk_global=4
n_ahead_global=8
model_name=12_4_CKPT
mode="base"
lr=1e-6
output_dir=fast-quiet-star/outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}
mkdir -p $output_dir
python -u quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --output_dir $output_dir \
    --model_name ${model_name} \
    --lr $lr > $output_dir/train_log 2>&1

bash fast-quiet-star/run_sh/eval.sh ${output_dir} ${n_ahead_global}

