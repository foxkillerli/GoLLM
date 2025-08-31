#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import re
from typing import List, Tuple

MOVE_RE = re.compile(r";\s*[BW]\s*\[([^]]*)\]", re.IGNORECASE)

def extract_moves(sgf_text: str) -> Tuple[str, int]:
    """
    Parse SGF text and return a single string of moves with no spaces.
    Map empty or 'tt' moves to 'ZP'. Returns (moves_line, num_passes).
    """
    num_passes = 0
    moves: List[str] = []

    # Normalize to lowercase to be safe
    sgf_text = sgf_text.lower()

    for payload in MOVE_RE.findall(sgf_text):
        # payload could be '', 'tt', or coordinates like 'dd'
        if payload == "" or payload == "tt":
            moves.append("ZP")
            num_passes += 1
        else:
            # Keep only ascii letters a-z; coordinates should be two letters in 19x19,
            # but we won't enforce length strictly in case of other sizes.
            filtered = "".join(ch for ch in payload if "a" <= ch <= "z")
            # Most SGF coords are exactly 2 chars; if not, we still keep the letters
            # since the instruction is to strip brackets/B/W/; and keep coordinate info.
            if filtered == "" or filtered == "tt":
                moves.append("ZP")
                num_passes += 1
            else:
                moves.append(filtered)

    return ("".join(moves), num_passes)


def iter_sgf_files(ai_root: Path):
    for root, _dirs, files in os.walk(ai_root):
        for fn in files:
            if fn.lower().endswith(".sgf"):
                yield Path(root) / fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to the local clone of github.com/yenw/computer-go-dataset")
    parser.add_argument("--out", type=str, default=str(Path.home() / "Code" / "sgf_data"),
                        help="Output directory (default: ~/Code/sgf_data)")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of games per output file (default: 10000)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    ai_root = dataset_root / "AI"
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ai_root.exists():
        raise SystemExit(f"AI/ directory not found at: {ai_root}")

    total_files = 0
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
        out_path = out_dir / f"train_ai_{out_index:06d}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for line in buffer:
                f.write(line + "\n")
        written_files += 1
        buffer = []
        out_index += 1

    for sgf_path in iter_sgf_files(ai_root):
        total_files += 1
        try:
            text = sgf_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # If file can't be read, skip
            continue

        moves_line, num_pass = extract_moves(text)

        if moves_line == "":
            total_empty += 1
            continue

        buffer.append(moves_line)
        total_games += 1
        total_passes += num_pass

        if len(buffer) >= args.chunk_size:
            flush()

    # Flush remaining
    flush()

    # Write a simple summary file
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("SGF Cleaning Summary (AI/ only)\n")
        f.write(f"Dataset root: {dataset_root}\n")
        f.write(f"AI dir:       {ai_root}\n")
        f.write(f"Output dir:   {out_dir}\n")
        f.write(f"Chunk size:   {args.chunk_size}\n")
        f.write("\n")
        f.write(f"Total SGF files scanned: {total_files}\n")
        f.write(f"Total games parsed:      {total_games}\n")
        f.write(f"Total pass moves (ZP):   {total_passes}\n")
        f.write(f"Empty/invalid games:     {total_empty}\n")
        f.write(f"Output files written:    {written_files}\n")

    print("Done.")
    print(f"Scanned SGF files: {total_files}")
    print(f"Parsed games:      {total_games}")
    print(f"Pass moves (ZP):   {total_passes}")
    print(f"Empty games:       {total_empty}")
    print(f"Output files:      {written_files}")
    print(f"Output dir:        {out_dir}")

if __name__ == "__main__":
    main()
