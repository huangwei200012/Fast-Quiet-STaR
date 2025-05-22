import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
import datasets
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
from dataset import LMDataset
import wandb

random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)
eval_answer_marker="\nA:"
dataset_name = 'open-web-math/open-web-math'
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_ahead_talk_global", type=int, default=4)
parser.add_argument("--n_ahead_global", type=int, default=12)
parser.add_argument("--test_mode", type=str, default="one")
parser.add_argument("--n_ahead", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")

def model_init(params):
    if params is None:
        params = {}
    else:
        params = params.params
    n_ahead = args.n_ahead
    n_ahead_talk = 1
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)
    model_name = args.model_name
    print("Loading model")
    import sys
    sys.path.append("model_file")
    if "mistral" in model_name:
        print("load modeling_mistral")
        from modeling_mistral import MistralForCausalLM
        load_class = MistralForCausalLM
        tokenizer_path = "Mistral-7B-v0.1"
    if "qwen" in model_name:
        print("load modeling_qwen2")
        from modeling_qwen2 import Qwen2ForCausalLM
        load_class = Qwen2ForCausalLM
        tokenizer_path = "Qwen2.5-7B"
    model = load_class.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
    )
    print("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens_to_add = []
    if model.use_start_thought_token:
        special_tokens_to_add.append("<|startthought|>")
    if model.use_end_thought_token:
        special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = 1
    model.residual_think_head = residual_think_head
    # if args.baseline:
    #     model.skip_residual = True
    #     model.cumulative_residual = False
    #     model.clever_residual = False
    #     model.base_residual = False
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.use_policy_loss = False
    model.rm_initialized = True
    model.first_run = False
    model.wandb_enabled = False
    model.config_params = params
    model.run_start = int(time.time())
    model.eval_mode = True
    model.eval()
    return model

args = parser.parse_args()
print(args)
model_name = args.model_name

if args.n_ahead == 1:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
    )
else:
    model = model_init(None)
print("Loaded model")

if "Qwen" in model_name or "qwen" in  model_name:
    from eval_helpers_qwen import preprocess_eval_function_gsm,preprocess_eval_function_piqa,preprocess_eval_function_siqa, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
    valid_number_tokens = [15,16,17,18,19,20,21,22,23,24, 198]
    valid_letter_tokens = [362, 425, 356, 422, 468, 198]
    valid_siqa_tokens = [362, 425, 356, 198]
    valid_piqa_tokens = [362, 425, 198]
    valid_LogiQA_tokens = [362, 425, 356, 422, 198]
    valid_openbookqa_tokens = [362, 425, 356, 422, 198]
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-7B")
if "mistral" in model_name or "Mistral" in model_name:
    from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa,preprocess_eval_function_piqa,preprocess_eval_function_siqa, preprocess_function, compute_metrics, truncate_or_pad
    valid_number_tokens = [28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774, 28734, 13] # numbers
    valid_letter_tokens = [330, 365, 334, 384, 413, 13] # answer tokens
    valid_siqa_tokens = [330, 365, 334, 13]
    valid_piqa_tokens = [330, 365, 13]
    valid_LogiQA_tokens = [330, 365, 334, 384, 13]
    valid_openbookqa_tokens = [330, 365, 334, 384, 13]
    tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-v0.1")
tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id
print("Loaded tokenizer")
eval_dataset_gsm = datasets.load_from_disk("gsm8k").map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200,load_from_cache_file=False)
eval_dataset_csqa = datasets.load_from_disk("commonsense_qa").map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200,load_from_cache_file=False)
eval_dataset_piqa = datasets.load_from_disk("piqa_hf").map(preprocess_eval_function_piqa, batched=True, writer_batch_size=200,load_from_cache_file=False)
eval_dataset_siqa = datasets.load_from_disk("siqa_hf").map(preprocess_eval_function_siqa, batched=True, writer_batch_size=200,load_from_cache_file=False)
eval_datasets = {
    "csqa": eval_dataset_csqa,
    "gsm8k": eval_dataset_gsm,
    "piqa": eval_dataset_piqa,
    "siqa": eval_dataset_siqa,
}

for key in eval_datasets.keys():
    eval_dataset = eval_datasets[key]
    if key=="csqa":
        valid_tokens = valid_letter_tokens
    if key=="gsm8k":
        valid_tokens = valid_number_tokens
    if key=="piqa":
        valid_tokens = valid_piqa_tokens
    if key=="siqa":
        valid_tokens = valid_siqa_tokens
    if key=="LogiQA":
        valid_tokens = valid_LogiQA_tokens
    if key=="openbookqa":
        valid_tokens = valid_LogiQA_tokens
    total_answer_prob = 0
    for one_dataset in eval_dataset:
        # with torch.no_grad():
        input_ids = torch.tensor(one_dataset["input_ids"]).to(model.device).unsqueeze(0)
        attention_mask = torch.tensor(one_dataset["attention_mask"]).to(model.device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        question = one_dataset["labels"]
        logits_guess = outputs["logits"]
        detokenized_question = tokenizer.decode(question)
        is_numeric = detokenized_question.split(eval_answer_marker)[-1][1].isdigit()
        answer_count = detokenized_question.count(eval_answer_marker)
        for i in range(len(question) - 1, 0, -1):
            tokenized_subquestion = question[:i]
            if tokenized_subquestion[-1] == tokenizer.pad_token_id:
                continue
            detokenized_subquestion = tokenizer.decode(question[:i])
            if detokenized_subquestion.count(eval_answer_marker) < answer_count:
                break
        correct_answer_prob = 1
        # if is_numeric, then the first token just indicates that it's a number
        question_offset = 1 if is_numeric else 0
        for j in range(i + question_offset, len(question) - 1):
            if question[j + 1] == tokenizer.pad_token_id:
                break
            true_token = question[j + 1]
            # print(true_token)
            guess = torch.nn.functional.softmax(torch.tensor(logits_guess), dim=-1)[0,:,:]
            # we only care about the logits assigned to the correct token
            if true_token not in valid_tokens:
                continue
            guess_filtered = torch.zeros_like(guess)
            guess_filtered[:, valid_tokens] = guess[:, valid_tokens]
            guess_filtered = guess_filtered / guess_filtered.sum(dim=-1, keepdim=True)
            token_prob = guess_filtered[j, true_token]
            correct_answer_prob *= token_prob.item()
        total_answer_prob += correct_answer_prob/len(eval_dataset)
    print("{} ACC:".format(key),total_answer_prob)

        

