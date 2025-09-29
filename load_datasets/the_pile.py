from datasets import load_dataset
from prepare import tokenize_and_chunk, save_as_arrow
from transformers import AutoTokenizer
import argparse

"""The Pile dataset is very large (800GB+ raw). Consider using a smaller mirror or streaming mode, or subsets."""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="pile")  # or a mirror name
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="pile_tokenized")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--L", type=int, default=2048)
    args = parser.parse_args()

    # if EleutherAI/pile is too large, consider ArmelR/the-pile-splitted or load_dataset streaming
    ds = load_dataset("EleutherAI/pile", split=args.split)  # may be huge; consider streaming=True
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    texts = []
    for ex in ds:
        texts.append(ex.get("text", ex.get("content") or ex.get("text")))
        if len(texts) >= 10000:  # process in batches to avoid large memory
            examples = tokenize_and_chunk(texts, tokenizer, L_tokens=args.L, stride=args.L//2)
            # save or append to disk (append implementation omitted for brevity)
            texts = []
    # final flush...
    
if __name__ == "__main__":
    main()