from datasets import load_dataset
from prepare import tokenize_and_chunk, save_as_arrow
from transformers import AutoTokenizer
import argparse

""" Load DeepMind PG19 from huggingface datasets: 'https://huggingface.co/datasets/deepmind/pg19?utm_source=chatgpt.com' """
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="pg19_tokenized")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--L", type=int, default=2048)
    args = parser.parse_args()

    ds = load_dataset("deepmind/pg19", split=args.split)  # each example corresponds to a book text field
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    texts = [ex['text'] for ex in ds]  # check dataset card for field name
    examples = tokenize_and_chunk(texts, tokenizer, L_tokens=args.L, stride=args.L//2)
    save_as_arrow(examples, args.out_dir)

if __name__ == "__main__":
    main()
