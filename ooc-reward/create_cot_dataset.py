import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np
from huggingface_hub import login
from datasets import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_base_dataset(path_to_data):
    system_prompt = "You are going to write about a specific subject. Before you are told what subject to write about, you will be prompted by the user to briefly summarize the goals you are trying to achieve in the response. Describe these goals in one or two sentences, then await a further message telling you what subject to write about."

    cot_prompt = "In the next message, I will give you a subject to write about. Before that, please very briefly summarize the goals you are trying to achieve in your response. Write no more than two sentences."

    # Load the JSON data
    with open(path_to_data, 'r', encoding='utf-8') as f:
        json_content = json.load(f)
        data = json_content["data"]  # Access the "data" key

    return Dataset.from_dict({
        "system_prompt": [system_prompt] * len(data),
        "cot_prompt": [cot_prompt] * len(data),
        "subject": [item["subject"] for item in data],
        "preferred": [item["preferred"] for item in data],
        "dispreferred": [item["dispreferred"] for item in data]
    })

def generate_cot(model, tokenizer, cfg, system_prompt, cot_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cot_prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.size(1) + cfg.max_cot_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
    if cfg.verbose:
        print(generated_text)
    return generated_text

def add_cot_to_dataset(model, tokenizer, cfg, original_dataset):
    # Generate CoT only once
    system_prompt = original_dataset[0]['system_prompt']
    cot_prompt = original_dataset[0]['cot_prompt']
    cot = generate_cot(model, tokenizer, cfg, system_prompt, cot_prompt)

    augmented_data = []
    for item in original_dataset:
        augmented_data.append({
            'system_prompt': item['system_prompt'],
            'cot': cot,
            'subject': item['subject'],
            'preferred': item['preferred'],
            'dispreferred': item['dispreferred']
        })
    return Dataset.from_dict({k: [dic[k] for dic in augmented_data] for k in augmented_data[0]})

def reformat_data(cot_dataset):
    dpo_data = []
    for item in cot_dataset:
        prompt_with_cot = [
            {"role": "system", "content": item['system_prompt']},
            {"role": "user", "content": "In the next message, I will give you a subject to write about. Before that, please summarize the goals you are trying to achieve in your response."},
            {"role": "assistant", "content": item['cot']},
            {"role": "user", "content": f"Write about the subject of {item['subject']}"}
        ]
        dpo_data.append({
            'prompt_with_cot': prompt_with_cot,
            'preferred': item['preferred'],
            'dispreferred': item['dispreferred']
        })

    return Dataset.from_dict({k: [dic[k] for dic in dpo_data] for k in dpo_data[0]})

def create_dataset(model, tokenizer, cfg):
    base_dataset = create_base_dataset()
    cot_dataset = add_cot_to_dataset(model, tokenizer, cfg, base_dataset)
    formatted_data = reformat_data(cot_dataset)
    return formatted_data