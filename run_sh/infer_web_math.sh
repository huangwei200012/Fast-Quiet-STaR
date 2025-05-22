output_path=$1
checkpoint=$2
n_ahead=$3
mkdir -p $output_path
mkdir -p $output_path/input_ids
mkdir -p $output_path/labels
mkdir -p $output_path/kl_labels
mkdir -p $output_path/attention_mask
CUDA_VISIBLE_DEVICES=0 python -u make_quiet_infer_data.py \
    --begin 0 \
    --end 1 \
    --n_ahead $n_ahead \
    --checkpoint $checkpoint \
    --output_path $output_path > $output_path/begin_0_1 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=1 python -u make_quiet_infer_data.py \
    --begin 1 \
    --end 2 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_1_2 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=2 python -u make_quiet_infer_data.py \
    --begin 2 \
    --end 3 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_2_3 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=3 python -u make_quiet_infer_data.py \
    --begin 3 \
    --end 4 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_3_4 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=4 python -u make_quiet_infer_data.py \
    --begin 4 \
    --end 5 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_4_5 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=5 python -u make_quiet_infer_data.py \
    --begin 5 \
    --end 6 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_5_6 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=6 python -u make_quiet_infer_data.py \
    --begin 6 \
    --end 7 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_6_7 2>&1 &
sleep 5

CUDA_VISIBLE_DEVICES=7 python -u make_quiet_infer_data.py \
    --begin 7 \
    --end 8 \
    --checkpoint $checkpoint \
    --output_path $output_path \
    --n_ahead $n_ahead > $output_path/begin_7_8 2>&1 
