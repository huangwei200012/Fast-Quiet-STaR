import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
import datasets
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
import os
import time
import wandb
import numpy as np
from tqdm import tqdm
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--n_ahead", type=int, default=8)
parser.add_argument("--begin", type=int, default=0)
parser.add_argument("--end", type=int, default=1)
args = parser.parse_args()
print("args:",args)
print("begin:",args.begin)
print("end:",args.end)
if "mistral" in args.checkpoint:
    from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
if "qwen" in args.checkpoint:
    from eval_helpers_qwen import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
step = 1000
train_dataset = datasets.load_from_disk("open-web-math")["train"].select(range(args.begin*step,args.end*step)).map(preprocess_function, batched=True, writer_batch_size=100,load_from_cache_file=False)
print(train_dataset)
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
    model_name = args.checkpoint
    print("Loading model")
    import sys
    if "mistral" in model_name:
        sys.path.append("model_file")
        print("load modeling_mistral")
        from modeling_mistral import MistralForCausalLM
        load_class = MistralForCausalLM
        tokenizer_path = "Mistral-7B-v0.1"
    if "qwen" in model_name:
        sys.path.append("model_file")
        print("load modeling_qwen2")
        from modeling_qwen2_new import Qwen2ForCausalLM
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

model = model_init(None)
for i in tqdm(range(len(train_dataset)), desc="Processing"):
    input_id=torch.tensor(train_dataset[i]["input_ids"]).unsqueeze(0).to(model.device)
    attention_mask=torch.tensor(train_dataset[i]["attention_mask"]).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model(input_ids=input_id,attention_mask=attention_mask)
    numpy_logits = output.logits.cpu().detach().to(torch.float32).numpy()
    numpy_input_id = input_id.cpu().numpy()
    numpy_attention_mask = attention_mask.cpu().numpy()
    numpy_label = np.array(train_dataset[i]["labels"])[np.newaxis, :]

    np.save('{}/input_ids/input_ids_{}'.format(args.output_path,i+args.begin*step), numpy_input_id)
    np.save('{}/labels/label_{}'.format(args.output_path,i+args.begin*step), numpy_label)
    np.save('{}/kl_labels/logits_{}'.format(args.output_path,i+args.begin*step), numpy_logits)
    np.save('{}/attention_mask/attention_mask_{}'.format(args.output_path,i+args.begin*step), numpy_attention_mask)




