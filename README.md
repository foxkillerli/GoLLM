# GoLLM

**GoLLM** is an experimental project to explore **Large Language Models (LLMs) for Go (Weiqi/Baduk)**, inspired by the paper [*The Go Transformer: Natural Language Modeling for Game Play*](https://arxiv.org/pdf/2007.03500).  

The goal is to train a LLaMA-based model on large-scale SGF game records and investigate its ability to generate legal moves and predict next moves in Go games.

---

## Dataset

We use the open-source dataset from [yenw/computer-go-dataset](https://github.com/yenw/computer-go-dataset), which contains:

- **AI self-play games** (thousands of high-quality AI vs AI SGFs)  
- **Professional human games** (`pro2000+` and others)  

### Cleaning & Preprocessing

- Remove non-ASCII characters, `;`, `[`, `]`  
- Keep only coordinate moves  
- Replace empty `[]` or `[tt]` with `ZZ`  
- Remove extra spaces between moves  
- Output compact text format (one game per line)  
- Group into files (10,000 games per file)  

---

## Tech Stack

- **Model**: LLaMA 3 (7B)  
- **Fine-tuning**: [QLoRA](https://arxiv.org/abs/2305.14314) (4-bit quantization with LoRA adapters)  
- **Frameworks**:  
  - [Transformers](https://github.com/huggingface/transformers)  
  - [PEFT](https://github.com/huggingface/peft)  
  - [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)  
- **Tokenizer**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF models via Ollama)  
- **Hardware**: Single RTX 4070 ti (16GB VRAM)  

---

## Usage

### 1. Clean SGF files

```bash
python scripts/clean_sgf.py \
  --input ~/Code/sgf_data/AI \
  --output data/clean/AI
```

For professional games (e.g., pro2000+):
```bash
python scripts/clean_pro.py \
  --input ~/Code/sgf_data/Professional/pro2000+.txt \
  --output data/clean/pro
```

### 2. Tokenization & Sharding

Pack cleaned games into tokenized sequences (train/val/test split = 80/10/10, context length 512):
```bash
python scripts/preprocess_go_lm.py \
  --backend llama_cpp \
  --gguf-path ~/.ollama_models/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29 \
  --input-dir data/clean/AI \
  --glob "train_*.txt" \
  --context-length 512 \
  --output-dir data/packed/train
```

Validation set example:
```bash
python scripts/llama31_token_pack_ollama.py \
  --backend llama_cpp \
  --gguf-path ~/.ollama_models/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29 \
  --input-dir data/clean/pro \
  --glob "val_*.txt" \
  --context-length 512 \
  --output-dir data/packed/val
```

### 3. Training with QLoRA

Run QLoRA fine-tuning (train_qlora.py):
```bash
torchrun --nproc_per_node=1 scripts/train_qlora.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --dataset data/packed/train \
  --validation_data data/packed/val \
  --output_dir artifacts/lora_adapter \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_strategy epoch
```

## Training Results (v0.1)

* Model: LLaMA 3 7B, QLoRA (RTX 4070, 12GB)

* Dataset: ~50k games (~10M tokens)

* Epochs: 3

* Learning rate: 2e-5 → cosine decay

* Train loss: ~4.36

* Eval loss: ~4.16

This shows the model successfully learns to predict legal Go moves, though playing strength is still limited.

## Next Steps
- [ ] Train with 8B model and more datasets for stronger results

- [ ] Add self-play data augmentation

- [ ] Evaluate move prediction accuracy (top-k metrics)

## Acknowledgments

[yenw/computer-go-dataset](https://github.com/yenw/computer-go-dataset)

[HuggingFace](https://https://huggingface.co/)

[llama.cpp](https://github.com/ggerganov/llama.cpp) (GGUF models via Ollama)