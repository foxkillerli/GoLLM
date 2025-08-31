# eval_final.py
import glob, numpy as np, torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import Dataset

class NPZPackedDataset(Dataset):
    def __init__(self, paths, seq_len=512):
        self.samples=[]
        for p in paths:
            arr=np.load(p)
            x=arr.get("input_ids", arr.get("tokens"))
            if x.ndim==1:
                usable=(len(x)//seq_len)*seq_len
                x=x[:usable].reshape(-1, seq_len)
            self.samples.append(x.astype(np.int64))
        self.samples=np.concatenate(self.samples,0)
    def __len__(self): return self.samples.shape[0]
    def __getitem__(self, i):
        t=torch.from_numpy(self.samples[i])
        return {"input_ids": t, "labels": t.clone(), "attention_mask": torch.ones_like(t)}

model_name="meta-llama/Meta-Llama-3.1-8B"
tokenizer=AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token

val_ds=NPZPackedDataset(sorted(glob.glob("data/val/*.npz")), seq_len=512)

bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                       bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                            device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "artifacts/lora_adapter")

args=TrainingArguments(output_dir="runs/eval_only", per_device_eval_batch_size=2)
trainer=Trainer(model=model, args=args, eval_dataset=val_ds, tokenizer=tokenizer)
print(trainer.evaluate())
