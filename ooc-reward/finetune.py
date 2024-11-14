import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
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

from create_cot_dataset import create_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    max_cot_length : int
    verbose : bool
    beta : float
    lr : float
    num_epochs : int
    batch_size : int
    new_cot_freq : int
    lora_alpha : float
    lora_r : float
    model_name : str


def load_models(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model_to_finetune = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    original_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Configure LoRA
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model_to_finetune = get_peft_model(model_to_finetune, lora_config).to(device)

    # Freeze the original model
    for param in original_model.parameters():
        param.requires_grad = False

    return tokenizer, model_to_finetune, original_model


def compute_ce_loss_with_kl(model_to_finetune, original_model, tokenizer, cfg, batch):
    prompt_with_cot = batch["prompt_with_cot"]
    preferred = batch["preferred"]

    def process_sequence(prompts, responses):
        full_sequences = []
        for i, response in enumerate(responses):
            sequence = [
                {"role": msg["role"][i], "content": msg["content"][i]}
                for msg in prompts
            ]
            sequence.append({"role": "assistant", "content": response})
            full_sequences.append(sequence)

        input_texts = tokenizer.apply_chat_template(full_sequences, tokenize=False)
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        return inputs.input_ids.to(model_to_finetune.device), inputs.attention_mask.to(model_to_finetune.device)

    preferred_input_ids, preferred_attention_mask = process_sequence(prompt_with_cot, preferred)

    with torch.no_grad():
        original_outputs = original_model(input_ids=preferred_input_ids, attention_mask=preferred_attention_mask)
    outputs = model_to_finetune(input_ids=preferred_input_ids, attention_mask=preferred_attention_mask)

    logits = outputs.logits
    original_logits = original_outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_original_logits = original_logits[:, :-1, :].contiguous()
    shift_labels = preferred_input_ids[:, 1:].contiguous()

    def find_message_span(message_content, full_sequence, tokenizer):
        full_str = tokenizer.decode(full_sequence)
        pos = full_str.find(message_content)
        if pos == -1:
            return None

        prefix = full_str[:pos]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        content_tokens = tokenizer.encode(message_content, add_special_tokens=False)

        start_token = len(prefix_tokens)
        end_token = start_token + len(content_tokens)

        extracted = tokenizer.decode(full_sequence[start_token:end_token])
        if not message_content.strip() in extracted:
            for i in range(5):
                test_start = max(0, start_token - i)
                test_extracted = tokenizer.decode(full_sequence[test_start:end_token])
                if message_content.strip() in test_extracted:
                    start_token = test_start
                    break

            for i in range(5):
                test_end = min(len(full_sequence), end_token + i)
                test_extracted = tokenizer.decode(full_sequence[test_start:test_end])
                if message_content.strip() in test_extracted:
                    end_token = test_end
                    break

        return start_token, end_token

    all_assistant_spans = []
    for i in range(preferred_input_ids.size(0)):
        assistant_spans = []
        full_sequence = preferred_input_ids[i].cpu().tolist()

        for msg in prompt_with_cot:
            if msg['role'][i] == 'assistant':
                span = find_message_span(msg['content'][i], full_sequence, tokenizer)
                if span:
                    shifted_span = (max(0, span[0] - 1), max(0, span[1] - 1))
                    assistant_spans.append(shifted_span)

        final_span = find_message_span(preferred[i], full_sequence, tokenizer)
        if final_span:
            shifted_span = (max(0, final_span[0] - 1), max(0, final_span[1] - 1))
            assistant_spans.append(shifted_span)

        all_assistant_spans.append(assistant_spans)

    kl_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    ce_mask = torch.zeros_like(shift_labels, dtype=torch.bool)

    for i, assistant_spans in enumerate(all_assistant_spans):
        for start_idx, end_idx in assistant_spans:
            if start_idx < shift_labels.size(1):
                actual_end = min(end_idx, shift_labels.size(1))
                kl_mask[i, start_idx:actual_end] = True

        if assistant_spans:
            final_start, final_end = assistant_spans[-1]
            if final_start < shift_labels.size(1):
                actual_end = min(final_end, shift_labels.size(1))
                ce_mask[i, final_start:actual_end] = True

    masked_logits_kl = shift_logits[kl_mask]
    masked_original_logits_kl = shift_original_logits[kl_mask]
    masked_logits_ce = shift_logits[ce_mask]
    shift_labels_ce = shift_labels[ce_mask]

    loss_fct = torch.nn.CrossEntropyLoss()
    ce_loss = loss_fct(masked_logits_ce, shift_labels_ce)

    log_probs = F.log_softmax(masked_logits_kl, dim=-1)
    original_probs = F.softmax(masked_original_logits_kl, dim=-1)
    kl_div = F.kl_div(log_probs, original_probs, reduction='batchmean', log_target=False)

    loss = ce_loss + cfg.beta * kl_div

    print(f"Batch losses - CE: {ce_loss.item():.4f}, KL: {kl_div.item():.4f}, Total: {loss.item():.4f}")

    return loss, ce_loss.item(), kl_div.item()


def train(model_to_finetune, original_model, tokenizer, cfg):
    optimizer = AdamW(model_to_finetune.parameters(), lr=cfg.lr)
    model_to_finetune.train()
    original_model.eval()

    for epoch in range(cfg.num_epochs):
        cot_dataset = create_dataset(model_to_finetune, tokenizer, cfg)
        dataloader = DataLoader(cot_dataset, batch_size=cfg.batch_size, shuffle=True)

        total_loss = 0
        total_ce_loss = 0
        total_kl_div = 0
        for (i, batch) in enumerate(dataloader):
            if (i+1) % cfg.new_cot_freq == 0:
                cot_dataset = create_dataset(model_to_finetune, tokenizer, cfg)
                dataloader = DataLoader(cot_dataset, batch_size=cfg.batch_size, shuffle=True)
            optimizer.zero_grad()
            loss, ce_loss, kl_div = compute_ce_loss_with_kl(model_to_finetune, original_model, tokenizer, cfg, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_to_finetune.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss
            total_kl_div += kl_div

        avg_loss = total_loss / len(dataloader)
        avg_ce_loss = total_ce_loss / len(dataloader)
        avg_kl_div = total_kl_div / len(dataloader)
        print(f"Epoch {epoch+1} average losses - CE: {avg_ce_loss:.4f}, KL: {avg_kl_div:.4f}, Total: {avg_loss:.4f}")