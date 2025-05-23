
save_path_base=fast-quiet-star/gsm8k_cot_eval_vote/base
mkdir -p $save_path_base
for i in {1..10}
do
  echo "当前数字是 $i"
  save_path=${save_path_base}/quiet_star_${i}
  ckpt=Mistral-7B-v0.1
  model_mode=base
  CUDA_VISIBLE_DEVICES=0 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 0 \
      --model_mode ${model_mode} \
      --save_path ${save_path}_0.jsonl \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_0 &
  CUDA_VISIBLE_DEVICES=1 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 1 \
      --save_path ${save_path}_1.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_1 &
  CUDA_VISIBLE_DEVICES=2 python -u eval_vote.py \
      --mode "cot" --is_index 8 --index 2 \
      --save_path ${save_path}_2.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_2 &
  CUDA_VISIBLE_DEVICES=3 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 3 \
      --save_path ${save_path}_3.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_3 &
  CUDA_VISIBLE_DEVICES=4 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 4 \
      --save_path ${save_path}_4.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_4 &
  CUDA_VISIBLE_DEVICES=5 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 5 \
      --save_path ${save_path}_5.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_5 &
  CUDA_VISIBLE_DEVICES=6 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 6 \
      --save_path ${save_path}_6.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_6 &
  CUDA_VISIBLE_DEVICES=7 python -u fast-quiet-star/eval_vote.py \
      --mode "cot" --is_index 8 --index 7 \
      --save_path ${save_path}_7.jsonl \
      --model_mode ${model_mode} \
      --checkpoint ${ckpt} > ${save_path_base}/index_${i}_7 
done
