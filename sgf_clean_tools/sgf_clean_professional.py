#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
from typing import List, Tuple

MOVE_RE = re.compile(r";\s*[BW]\s*\[([^]]*)\]", re.IGNORECASE)

def extract_moves(sgf_text: str) -> Tuple[str, int]:
    num_passes = 0
    moves: List[str] = []
    sgf_text = sgf_text.lower()
    for payload in MOVE_RE.findall(sgf_text):
        if payload == "" or payload == "tt":
            moves.append("ZP")
            num_passes += 1
        else:
            filtered = "".join(ch for ch in payload if "a" <= ch <= "z")
            if filtered == "" or filtered == "tt":
                moves.append("ZP")
                num_passes += 1
            else:
                moves.append(filtered)
    return ("".join(moves), num_passes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to the local clone of github.com/yenw/computer-go-dataset")
    parser.add_argument("--out", type=str, default=str(Path.home() / "Code" / "sgf_data_pro"),
                        help="Output directory (default: ~/Code/sgf_data_pro)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of games per output file (default: 10000)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pro_file = dataset_root / "Professional" / "pro2000+.txt"
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pro_file.exists():
        raise SystemExit(f"Professional/pro2000+.txt not found at {pro_file}")

    total_lines = 0
    total_games = 0
    total_passes = 0
    total_empty = 0

    buffer: List[str] = []
    out_index = 1
    written_files = 0

    def flush():
        nonlocal out_index, written_files, buffer
        if not buffer:
            return
        out_path = out_dir / f"train_pro_{out_index:06d}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for line in buffer:
                f.write(line + "\n")
        written_files += 1
        buffer = []
        out_index += 1

    with pro_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1
            sgf_text = line.strip()
            if not sgf_text:
                continue
            moves_line, num_pass = extract_moves(sgf_text)
            if moves_line == "":
                total_empty += 1
                continue
            buffer.append(moves_line)
            total_games += 1
            total_passes += num_pass
            if len(buffer) >= args.chunk_size:
                flush()

    flush()

    # Summary
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("SGF Cleaning Summary (Professional/pro2000+.txt)\n")
        f.write(f"Dataset root: {dataset_root}\n")
        f.write(f"Input file:   {pro_file}\n")
        f.write(f"Output dir:   {out_dir}\n")
        f.write(f"Chunk size:   {args.chunk_size}\n\n")
        f.write(f"Total lines read:   {total_lines}\n")
        f.write(f"Total games parsed: {total_games}\n")
        f.write(f"Total pass moves:   {total_passes}\n")
        f.write(f"Empty/invalid:      {total_empty}\n")
        f.write(f"Output files:       {written_files}\n")

    print("Done.")
    print(f"Lines read:      {total_lines}")
    print(f"Games parsed:    {total_games}")
    print(f"Pass moves (ZP): {total_passes}")
    print(f"Empty games:     {total_empty}")
    print(f"Output files:    {written_files}")
    print(f"Output dir:      {out_dir}")

if __name__ == "__main__":
    main()
