n_ahead_talk_global=8
n_ahead_global=16
model_name=/nlp_group/decapoda-research/Mistral-7B-v0.1
mode="base"
lr=2e-6
output_dir=/mmu_nlp_hdd/huangwei12/research/quiet-star-new/outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}
mkdir -p $output_dir
python -u /mmu_nlp_hdd/huangwei12/research/quiet-star-new/quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --output_dir $output_dir \
    --model_name ${model_name} \
    --lr $lr > $output_dir/train_log 2>&1

bash /mmu_nlp_hdd/huangwei12/research/quiet-star-new/run_sh/eval.sh ${output_dir} ${n_ahead_global}


n_ahead_talk_global=4
n_ahead_global=12
mode="base"
lr=1e-6
output_dir=/mmu_nlp_hdd/huangwei12/research/quiet-star-new/outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}
mkdir -p $output_dir
python -u /mmu_nlp_hdd/huangwei12/research/quiet-star-new/quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --output_dir $output_dir \
    --model_name /nlp_group/decapoda-research/Mistral-7B-v0.1 \
    --lr $lr > $output_dir/train_log 2>&1

bash /mmu_nlp_hdd/huangwei12/research/quiet-star-new/run_sh/eval.sh ${output_dir} ${n_ahead_global}



n_ahead_talk_global=4
n_ahead_global=8
mode="base"
lr=1e-6
output_dir=/mmu_nlp_hdd/huangwei12/research/quiet-star-new/outputs/mistral/${n_ahead_global}_${n_ahead_talk_global}_${mode}_${lr}
mkdir -p $output_dir
python -u /mmu_nlp_hdd/huangwei12/research/quiet-star-new/quiet-star-train.py \
    --n_ahead_talk_global $n_ahead_talk_global \
    --n_ahead_global $n_ahead_global \
    --mode $mode \
    --output_dir $output_dir \
    --model_name /nlp_group/decapoda-research/Mistral-7B-v0.1 \
    --lr $lr > $output_dir/train_log 2>&1

bash /mmu_nlp_hdd/huangwei12/research/quiet-star-new/run_sh/eval.sh ${output_dir} ${n_ahead_global}

