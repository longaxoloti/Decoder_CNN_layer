from datasets import load_dataset
from prepare import tokenize_and_chunk, save_as_arrow
from transformers import AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="manu/project_gutenberg")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="gutenberg_tokenized")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--L", type=int, default=2048)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    texts = [ex['text'] for ex in ds]  # check card for field name
    examples = tokenize_and_chunk(texts, tokenizer, L_tokens=args.L, stride=args.L//2)
    save_as_arrow(examples, args.out_dir)

if __name__ == "__main__":
    main()