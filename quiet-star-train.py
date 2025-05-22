import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
from dataset import LMDataset,DataCollatorForLMDataset,make_supervised_data_module
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
import os
import time
import wandb
import argparse
import datasets
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)


parser = argparse.ArgumentParser()
parser.add_argument("--n_ahead_talk_global", type=int, default=4)
parser.add_argument("--n_ahead_global", type=int, default=12)
parser.add_argument("--lr", type=float, default=1e-6)
parser.add_argument("--mode", type=str, default="base")
parser.add_argument("--data_path", type=str, default="/nlp_group/huangwei12/Infer_research/hugging_face_models_and_dataset/open-web-math")
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--max_train_length", type=int, default=256)
parser.add_argument("--policy_loss_beta", type=float, default=1e6)
parser.add_argument("--model_name", type=str, default="/nlp_group/huangwei12/r1_research/open-r1/huggingface_model/Qwen2.5-7B")
args = parser.parse_args()
print(args)
policy_loss_beta = args.policy_loss_beta
model_name = args.model_name
if "Qwen" in model_name or "qwen" in  model_name:
    from eval_helpers_qwen import preprocess_eval_function_gsm,preprocess_eval_function_piqa,preprocess_eval_function_siqa, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
if "mistral" in model_name or "Mistral" in model_name:
    from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa,preprocess_eval_function_piqa,preprocess_eval_function_siqa, preprocess_function, compute_metrics, truncate_or_pad

n_ahead_talk_global = args.n_ahead_talk_global
n_ahead_global = args.n_ahead_global
n_passes_global = 2
if args.mode=="last_rl":
    n_passes_global = 1

full_batch_size = 8
eval_and_logging_steps = 1000
save_steps = 10

def model_init(params):
    original = False
    if params is None:
        params = {}
    else:
        params = params.params
    # save params to file
    n_ahead = params.get("n_ahead", n_ahead_global if not original else 1)
    n_ahead_talk = params.get("n_ahead_talk", n_ahead_talk_global if not original else 1)
    n_passes = params.get("n_passes", n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = args.model_name
    import sys
    if "Qwen" in model_name or "qwen" in  model_name:
        sys.path.append("/nlp_group/huangwei12/Infer_research/quiet_star/model_file")
        if args.mode=="base":
            from modeling_qwen2 import Qwen2ForCausalLM
            print("load modeling_qwen2")
        elif args.mode=="last_rl":
            from modeling_qwen2_last import Qwen2ForCausalLM
            print("load modeling_qwen2_last")
        load_methed = Qwen2ForCausalLM
        tokenizer_path = "/nlp_group/huangwei12/r1_research/open-r1/huggingface_model/Qwen2.5-7B"
    elif "mistral" in model_name or "Mistral" in model_name:
        sys.path.append("/mmu_nlp_hdd/huangwei12/research/quiet-star-new/model_file")
        if args.mode=="base":
            from modeling_mistral import MistralForCausalLM
            print("load modeling_mistral")
        elif args.mode=="last_rl":
            from modeling_mistral_last import MistralForCausalLM
            print("load modeling_mistral_last")
        load_methed = MistralForCausalLM
        tokenizer_path = "/nlp_group/decapoda-research/Mistral-7B-v0.1"
    else:
        print("Error !!!")
    print("Loading model")
    model = load_methed.from_pretrained(
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
    model.policy_loss_beta = args.policy_loss_beta
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.wandb_enabled = True
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    if args.mode=="last_rl" or args.mode=="last_rl_wo":
        model.kill_after = 300
    else:
        model.kill_after = 150
    model.train()
    print("model.policy_loss_beta:", model.policy_loss_beta)
    print("model.n_ahead:", model.n_ahead)
    print("model.n_ahead_talk:", model.n_ahead_talk)
    print("model.n_passes:", model.n_passes)
    return model

if args.mode=="last_rl":
    train_dataset = LMDataset(args.data_path)
else:
    dataset = datasets.load_from_disk(args.data_path)["train"].select(range(1000))
    train_dataset = dataset.shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200,load_from_cache_file=False)

batch_size = full_batch_size // n_passes_global
global_gradient_accumulation_steps = full_batch_size // batch_size
training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.lr,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=global_gradient_accumulation_steps,
    max_grad_norm=1.0,
    max_steps=100000,
    warmup_steps=20,
    auto_find_batch_size=True,
    weight_decay=0.001,
    report_to = "none",
    label_names=["labels"],
    include_inputs_for_metrics=True,
    logging_steps=10,
    save_steps=save_steps,
)
trainer = Trainer(
    args=training_args,
    train_dataset=train_dataset,
    model_init=model_init,
)
trainer.train()
