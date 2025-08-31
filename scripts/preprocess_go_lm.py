#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Callable

import numpy as np
from tqdm import tqdm

# Optional deps
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # type: ignore


def iter_lines(files: List[Path]) -> Iterator[str]:
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield s


class Encoder:
    def __init__(self, backend: str, tokenizer_name: Optional[str], gguf_path: Optional[str]):
        self.backend = backend
        if backend == "hf":
            if AutoTokenizer is None:
                raise RuntimeError("transformers not installed; pip install transformers tokenizers")
            if not tokenizer_name:
                raise RuntimeError("--tokenizer is required for backend=hf")
            self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            self._eos = self.tok.eos_token_id
            self._bos = self.tok.bos_token_id
            if self._eos is None:
                raise RuntimeError("Tokenizer has no eos_token_id")
        elif backend == "llama_cpp":
            if Llama is None:
                raise RuntimeError("llama-cpp-python not installed; pip install llama-cpp-python")
            if not gguf_path:
                raise RuntimeError("--gguf-path is required for backend=llama_cpp")
            # vocab_only=True loads vocab/tokenizer only (fast)
            self.llm = Llama(model_path=os.path.expanduser(gguf_path), vocab_only=True)
            self._eos = self.llm.token_eos()
            try:
                self._bos = self.llm.token_bos()
            except Exception:
                self._bos = None
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

    @property
    def eos_id(self) -> int:
        return int(self._eos)

    @property
    def bos_id(self) -> Optional[int]:
        return int(self._bos) if self._bos is not None else None

    def encode(self, text: str) -> List[int]:
        if self.backend == "hf":
            return self.tok(text, add_special_tokens=False)["input_ids"]
        else:
            return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False))


class ShardWriter:
    def __init__(self, base_dir: Path, context_len: int, shard_tokens: int):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.context_len = context_len
        self.shard_tokens = int(shard_tokens)
        self.cur_tokens = 0
        self.shard_idx = 1
        self.buf: List[np.ndarray] = []
        self.total_sequences = 0
        self.total_shards = 0

    def _maybe_rollover(self):
        if self.cur_tokens >= self.shard_tokens and self.buf:
            self._flush()

    def write(self, seq: np.ndarray):
        assert seq.dtype == np.int32 and seq.ndim == 1 and len(seq) == self.context_len
        self.buf.append(seq)
        self.cur_tokens += len(seq)
        self.total_sequences += 1
        self._maybe_rollover()

    def _flush(self):
        if not self.buf:
            return
        arr = np.stack(self.buf, axis=0)
        out_path = self.base_dir / f"shard_{self.shard_idx:06d}.npz"
        np.savez_compressed(out_path, input_ids=arr)
        print(f"[WRITE] {out_path} -> {arr.shape}")
        self.shard_idx += 1
        self.total_shards += 1
        self.cur_tokens = 0
        self.buf = []

    def close(self):
        self._flush()

    def stats(self):
        return {
            "total_sequences": self.total_sequences,
            "total_shards": self.total_shards,
            "context_len": self.context_len,
        }


def pack_tokens(
    input_files: List[Path],
    encoder: Encoder,
    context_len: int,
    out_dir: Path,
    shard_tokens: int,
    add_bos: bool,
    seed: int,
    val_ratio: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    train_writer = ShardWriter(out_dir / "train", context_len, shard_tokens)
    val_writer = ShardWriter(out_dir / "val", context_len, shard_tokens)

    total_games = 0
    total_tokens = 0
    dropped_long_games = 0

    
    train_seq_buf: List[int] = []
    val_seq_buf: List[int] = []

    def flush_seq_buf(seq_buf: List[int], writer: "ShardWriter"):
        
        if not seq_buf:
            return
        if len(seq_buf) > context_len:
            
            del seq_buf[context_len:]
        if len(seq_buf) < context_len:
            pad_id = encoder.eos_id
            seq_buf.extend([pad_id] * (context_len - len(seq_buf)))
        arr = np.array(seq_buf, dtype=np.int32)
        writer.write(arr)
        seq_buf.clear()

    files = list(input_files)
    rng.shuffle(files)

    for line in tqdm(iter_lines(files), desc="encode+pack"):
        total_games += 1

        
        split_is_val = (rng.random() < val_ratio)
        seq_buf = val_seq_buf if split_is_val else train_seq_buf
        writer = val_writer if split_is_val else train_writer

        
        enc = encoder.encode(line)
        game_tokens: List[int] = []
        if add_bos and encoder.bos_id is not None:
            game_tokens.append(encoder.bos_id)
        game_tokens.extend(enc)
        game_tokens.append(encoder.eos_id)

        total_tokens += len(enc) + 1 + (1 if (add_bos and encoder.bos_id is not None) else 0)

        # if the length > context_len，drop it
        if len(game_tokens) > context_len:
            dropped_long_games += 1
            continue

        if len(seq_buf) + len(game_tokens) > context_len:
            flush_seq_buf(seq_buf, writer)

        seq_buf.extend(game_tokens)

        if len(seq_buf) == context_len:
            flush_seq_buf(seq_buf, writer)

    flush_seq_buf(train_seq_buf, train_writer)
    flush_seq_buf(val_seq_buf, val_writer)

    train_writer.close()
    val_writer.close()

    meta = {
        "backend": encoder.backend,
        "tokenizer": args.tokenizer,
        "gguf_path": args.gguf_path,
        "context_length": context_len,
        "total_games": total_games,
        "approx_total_tokens": total_tokens,
        "train": train_writer.stats(),
        "val": val_writer.stats(),
        "shard_tokens": shard_tokens,
        "val_ratio": val_ratio,
        "add_bos": add_bos,
        "eos_token_id": encoder.eos_id,
        "bos_token_id": encoder.bos_id,
        "dropped_long_games": dropped_long_games,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="llama_cpp", choices=["hf", "llama_cpp"],
                   help="Use HF or llama.cpp (GGUF) tokenizer")
    p.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3.1-8B",
                   help="HF tokenizer id (for backend=hf)")
    p.add_argument("--gguf-path", type=str, default=None,
                   help="Local GGUF path (for backend=llama_cpp)")

    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--glob", type=str, default="train_*.txt")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--shard-tokens", type=int, default=50_000_000)
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--add-bos", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    global args
    args = p.parse_args()

    input_dir = Path(os.path.expanduser(args.input_dir))
    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {input_dir}")

    enc = Encoder(backend=args.backend, tokenizer_name=args.tokenizer, gguf_path=args.gguf_path)
    pack_tokens(
        files,
        encoder=enc,
        context_len=args.context_length,
        out_dir=Path(os.path.expanduser(args.output_dir)),
        shard_tokens=args.shard_tokens,
        add_bos=args.add_bos,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
