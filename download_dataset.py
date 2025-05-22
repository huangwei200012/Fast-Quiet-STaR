from datasets import load_from_disk,Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import json
import datasets
from eval_helpers_qwen import preprocess_eval_function_gsm, preprocess_eval_function_csqa,preprocess_eval_function_piqa, preprocess_function, compute_metrics, truncate_or_pad

# piqa_hf
formatted_result = {"question": [], "choices": [], "answerKey": []}
label_path = "piqa/valid-labels.lst"
data_path = "piqa/valid.jsonl"
map_dict = {0:"A",1:"B"}
with open(data_path) as f1,open(label_path) as f2:
    lines = f1.readlines()
    labels = f2.readlines()
    for i in range(len(lines)):
        data = json.loads(lines[i])
        question = data["goal"]
        choices = "(A) " + data["sol1"] + "\n(B) "+ data["sol2"]
        answerKey = map_dict[int(labels[i].strip())]
        formatted_result['question'].append(question)
        formatted_result['choices'].append(choices)
        formatted_result['answerKey'].append(answerKey)

dataset = Dataset.from_dict(formatted_result)

# # Save the dataset to disk
dataset.save_to_disk("/nlp_group/huangwei12/Infer_research/quiet_star/huggingface_data_and_model/piqa_hf")

# siqa
label_path = "siqa/dev-labels.lst"
data_path = "siqa/dev.jsonl"
map_dict = {1:"A",2:"B",3:"C"}
with open(data_path) as f1,open(label_path) as f2:
    lines = f1.readlines()
    labels = f2.readlines()
    for i in range(len(lines)):
        data = json.loads(lines[i])
        question = data["context"]+data["question"]
        choices = "(A) " + data["answerA"] + "\n(B) "+ data["answerB"] + "\n(C) "+ data["answerC"]
        answerKey = map_dict[int(labels[i].strip())]
        formatted_result['question'].append(question)
        formatted_result['choices'].append(choices)
        formatted_result['answerKey'].append(answerKey)

dataset = Dataset.from_dict(formatted_result)

# # Save the dataset to disk
dataset.save_to_disk("siqa_hf")
