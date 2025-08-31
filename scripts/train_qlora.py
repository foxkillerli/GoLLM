# scripts/train_qlora.py
import os, glob, math, random
os.environ.setdefault("BNB_CUDA_VERSION", "121")
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from dataclasses import dataclass
from typing import Dict, List, Optional
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig  
)
from transformers.trainer_utils import get_last_checkpoint

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ========= 1) Dataset：read in from .npz =========
class NPZPackedDataset(Dataset):
    def __init__(self, npz_paths: List[str], seq_len: int):
        self.seq_len = seq_len
        self.samples = []
        for p in npz_paths:
            arr = np.load(p)
            
            tokens = arr.get("input_ids", arr.get("tokens"))
            assert tokens is not None, f"{p} missing input_ids/tokens"
            
            if tokens.ndim == 1:
                total = len(tokens)
                usable = (total // seq_len) * seq_len
                tokens = tokens[:usable].reshape(-1, seq_len)
            elif tokens.ndim == 2:
                assert tokens.shape[1] == seq_len, f"{p} The length of each line has to be {seq_len}"
            else:
                raise ValueError(f"Unsupported token shape: {tokens.shape}")
            self.samples.append(tokens.astype(np.int64))
        self.samples = np.concatenate(self.samples, axis=0)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        input_ids = torch.from_numpy(self.samples[idx])
        #labels >> 1：Trainer does the shift，so labels=input_ids
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# ========= 2) Data Collator（simply self-regression） =========
@dataclass
class SimpleDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # no need to pad
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        labels = torch.stack([f["labels"] for f in features], dim=0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B"  # need hugging-face access
    seq_len = 512

    train_paths = sorted(glob.glob("data/train/*.npz"))
    val_paths   = sorted(glob.glob("data/val/*.npz"))

    train_ds = NPZPackedDataset(train_paths, seq_len=seq_len)
    val_ds   = NPZPackedDataset(val_paths,   seq_len=seq_len)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantitative loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )

    # LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], # Llama 常见注入点
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = SimpleDataCollator(pad_token_id=tokenizer.pad_token_id)

    # hyperparameters
    args = TrainingArguments(
        output_dir="runs/llama31_8b_qlora_go",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        logging_steps=200,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    # saving LoRA adapter
    trainer.model.save_pretrained("artifacts/lora_adapter")
    tokenizer.save_pretrained("artifacts/lora_adapter")

    print("Done. LoRA adapter saved to artifacts/lora_adapter")


if __name__ == "__main__":
    main()
