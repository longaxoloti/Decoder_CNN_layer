# prepare_corpus.py
import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import regex as re

def basic_gutenberg_clean(text):
    # remove Project Gutenberg header/footer heuristics
    # This is simple; dataset-specific loaders may already remove headers.
    # Remove everything before *** START OF (THIS|THE) PROJECT GUTENBERG EBOOK
    text = re.sub(r'(?s).*?\*\*\* *START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text)
    text = re.sub(r'(?s)\*\*\* *END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*', '', text)
    return text.strip()

def tokenize_and_chunk(texts, tokenizer, L_tokens=2048, stride=None):
    # texts: list of strings
    if stride is None:
        stride = L_tokens  # non-overlap by default
    examples = []
    for doc in tqdm(texts):
        # basic clean
        doc = basic_gutenberg_clean(doc)
        toks = tokenizer.encode(doc)
        n = len(toks)
        start = 0
        while start < n:
            end = min(start + L_tokens, n)
            chunk = toks[start:end]
            # store token ids and original offset (start index in doc tokens)
            examples.append({"input_ids": chunk, "orig_token_start": start, "orig_token_len": len(chunk)})
            if end == n:
                break
            start += stride
    return examples

def save_as_arrow(examples, out_path, shard_size=10000):
    # examples: list of dicts with "input_ids", etc
    # create a HF Dataset and save to disk as arrow
    ds = Dataset.from_list(examples)
    ds.save_to_disk(out_path)
    print("Saved dataset to", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text_files", nargs="+", required=True, help="Paths to text files (one doc per file)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_name", default="gpt2")
    parser.add_argument("--L", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    texts = []
    for p in args.in_text_files:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    examples = tokenize_and_chunk(texts, tokenizer, L_tokens=args.L, stride=args.stride)
    os.makedirs(args.out_dir, exist_ok=True)
    save_as_arrow(examples, args.out_dir)

if __name__ == "__main__":
    main()