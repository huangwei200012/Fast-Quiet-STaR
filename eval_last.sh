


base_path=$1
n_ahead=$2
ckpts=(10 20 30 40 50 60 70 80)
num_ckpts=${#ckpts[@]}
GPUs="0"
for ((i=0; i<$num_ckpts; i++)); do
    ckpt=${ckpts[$i]}
    CUDA_VISIBLE_DEVICES=${GPUs} python -u quiet-star-eval.py \
        --n_ahead ${n_ahead} \
        --model_name $base_path/checkpoint-${ckpt} > $base_path/checkpoint_${ckpt}_eval_log 2>&1 &
    GPUs=$(($GPUs+1))
done
wait

ckpts=(90 100 110 120 130 140 150)
num_ckpts=${#ckpts[@]}
GPUs="0"
for ((i=0; i<$num_ckpts; i++)); do
    ckpt=${ckpts[$i]}
    CUDA_VISIBLE_DEVICES=${GPUs} python -u quiet-star-eval.py \
        --n_ahead ${n_ahead} \
        --model_name $base_path/checkpoint-${ckpt} > $base_path/checkpoint_${ckpt}_eval_log 2>&1 &
    GPUs=$(($GPUs+1))
done
wait

ckpts=(160 170 180 190 200 210 220)
num_ckpts=${#ckpts[@]}
GPUs="0"
for ((i=0; i<$num_ckpts; i++)); do
    ckpt=${ckpts[$i]}
    CUDA_VISIBLE_DEVICES=${GPUs} python -u quiet-star-eval.py \
        --n_ahead ${n_ahead} \
        --model_name $base_path/checkpoint-${ckpt} > $base_path/checkpoint_${ckpt}_eval_log 2>&1 &
    GPUs=$(($GPUs+1))
done
wait

ckpts=(230 240 250 260 270 280 290 300)
num_ckpts=${#ckpts[@]}
GPUs="0"
for ((i=0; i<$num_ckpts; i++)); do
    ckpt=${ckpts[$i]}
    CUDA_VISIBLE_DEVICES=${GPUs} python -u quiet-star-eval.py \
        --n_ahead ${n_ahead} \
        --model_name $base_path/checkpoint-${ckpt} > $base_path/checkpoint_${ckpt}_eval_log 2>&1 &
    GPUs=$(($GPUs+1))
done
wait
python get_result.py ${base_path}
