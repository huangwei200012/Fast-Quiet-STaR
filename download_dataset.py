from datasets import load_from_disk,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import json
import datasets
from eval_helpers_qwen import preprocess_eval_function_gsm, preprocess_eval_function_csqa,preprocess_eval_function_piqa, preprocess_function, compute_metrics, truncate_or_pad
from datasets import load_dataset
import datasets
dataset_name='open-web-math/open-web-math'

dataset = load_dataset(
    dataset_name,
    "en" if "c4" in dataset_name else "default",
    split=f"train[:]",
    ignore_verifications=True,
    num_proc=16,
)
dataset.save_to_disk("open-web-math")


