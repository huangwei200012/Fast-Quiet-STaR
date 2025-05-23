import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import datasets
import os
import time
import re
import json
import math
from tqdm import tqdm
from collections import Counter
import sys
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList,STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import re
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_idx", type=int, default=0)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--device_batch_size", type=int, default=8)
parser.add_argument("--max_idx", type=int, default=128)
parser.add_argument("--n_votes", type=int, default=8)
parser.add_argument("--temp", type=float, default=0.9)
parser.add_argument("--start_final_answer_idx", type=int, default=384)
parser.add_argument("--answer_length", type=int, default=12)
parser.add_argument("--root_prefix", type=str, default="YOUR_ROOT_HERE")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--final_answer_text", type=str, default="\nTherefore, the answer (arabic numerals) is")
parser.add_argument("--zero_shot_cot_prompt", type=str, default="\nA: Let's think step by step.")
parser.add_argument("--n_ahead", type=int, default=8)
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--is_index", type=int, default=1)
parser.add_argument("--mode", type=str, default="cot")
parser.add_argument("--model_mode", type=str, default="base")
parser.add_argument("--save_path", type=str, default="gsm8k_cot_eval_vote")
# parser.add_argument("-m", "--model_path", action="append", nargs="+")
args = parser.parse_args()

class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时，立即停止生成
    ---------------
    ver: 2023-08-02
    by: changhongyu
    """
    def __init__(self, token_id_list):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


def extract_numbers(string):
    string = re.sub(r'\d+\.\s', '', string)
    # print("经过re之后的:\n",string)
    regex = re.compile("(-?[$0-9.,]{2,})|(-?[0-9]+)")
    # 使用正则表达式查找所有数字
    numbers = regex.findall(string)[-1]
    if isinstance(numbers, tuple):
        match = [m for m in numbers if m][0]
    match = match.strip().replace("$","").replace(".","").replace(",","")
    return int(match)

# Load the GSM8K dataset and the model
eval_dataset_gsm = datasets.load_from_disk("gsm8k")
dataset_length = len(eval_dataset_gsm)
print("总的数据量：",dataset_length)
print("args.checkpoint",args.checkpoint)
if args.is_index!=1:
    number_len = math.ceil(dataset_length / args.is_index)
    print(number_len)
    if number_len*(args.index+1)>=dataset_length:
        end_id = dataset_length
    else:
        end_id = number_len*(args.index+1)
    eval_dataset_gsm = eval_dataset_gsm.select(range(number_len*(args.index),end_id))
    print("start:",number_len*(args.index))
    print("end:",end_id)
    print("length",end_id-number_len*(args.index))


import sys
sys.path.append("model_file")
from modeling_mistral_hf import MistralForCausalLM
model = MistralForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
    )
tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-v0.1",)
tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

acc = 0
pass_number = 0
stopping_criteria = StoppingCriteriaList()
stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[2,22478]))
with open(args.save_path,"w") as f:
    for data in tqdm(eval_dataset_gsm, desc="Processing"):
        text = 'You are a helpful AI assistant that provides reasonable and detailed responses to questions. You answer the question step by step and then provide the answer to the user in the form of "Therefore, the answer (arabic numerals) is ".\nThe question is: '+ data["question"] + "\nAnswer: Let's think step by step."
        inputs = tokenizer(text, truncation=False, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,max_new_tokens=1000,stopping_criteria=stopping_criteria,top_p=0.95,temperature=0.8,do_sample=True)
        
        outputs_text = tokenizer.decode(outputs[0])
        print("outputs_text:",outputs_text)
        # import pdb;pdb.set_trace()
        outputs_text = outputs_text.split(text)[1]
        outputs_text = outputs_text.split("Question:")[0].strip().replace(",","")
        try:
            outputs_text = extract_numbers(outputs_text)
        except:
            outputs_text = 1
        answer = int(data["answer"].split("####")[-1].replace(",",""))
        print("response:",outputs_text)
        print("answer:",answer)
        if answer==outputs_text:
            acc += 1
        line={}
        line["text"] = text
        line["response"] = outputs_text
        line["answer"] = answer
        line["output"] = tokenizer.decode(outputs[0])
        json.dump(line, f,ensure_ascii=False)
        f.write('\n')
    print("start:",number_len*(args.index))
    print("end:",end_id)
    print("length",end_id-number_len*(args.index))
    print("ACC:",acc)

