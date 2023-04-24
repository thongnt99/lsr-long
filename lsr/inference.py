from lsr.models import DualSparseEncoder
from lsr.tokenizer import Tokenizer
import torch
from tqdm import tqdm
import os
from collections import Counter
import json
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser("Encoding documents")
parser.add_argument("--inp", type=str,
                    default="data/msmarco_doc/splits_psg/part01", help="Inpput file")
parser.add_argument(
    "--cp", type=str, default="lsr42/qmlp_dmlm_msmarco_distil_kl_l1_0.0001", help="Model checkpoint")
parser.add_argument(
    "--out", type=str, default="data/msmarco_doc/vectors/part01", help="Output file")
parser.add_argument("--bs", type=int, default=64, help="Output file")
parser.add_argument("--type", type=str, default="doc", help="query/doc")
args = parser.parse_args()

if not Path(args.cp).is_dir():
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id=args.cp,
                          local_dir=args.cp)
    except:
        raise Exception(
            "wrong model's checkpoint: {model_dir_or_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DualSparseEncoder.from_pretrained(args.cp)
model.eval()
model.to(device)
tokenizer_path = os.path.join(args.cp, "tokenizer")
print(f"Loading tokenizer from {tokenizer_path}")
tokenizer = Tokenizer.from_pretrained(tokenizer_path)
ids = []
texts = []
with open(args.inp, "r") as f:
    for line in tqdm(f, desc=f"Reading data from {args.inp}"):
        try:
            idx, text = line.strip().split("\t")
            ids.append(idx)
            texts.append(text)
        except:
            pass
all_token_ids = list(range(tokenizer.get_vocab_size()))
all_tokens = np.array(tokenizer.convert_ids_to_tokens(all_token_ids))
results_to_file = []

for idx in tqdm(range(0, len(ids), args.bs)):
    batch_texts = texts[idx: idx + args.bs]
    batch_ids = ids[idx: idx + args.bs]
    batch_tkn = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=400,
        return_special_tokens_mask=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        if args.type == "query":
            batch_output = model.encode_queries(**batch_tkn).to("cpu")
        else:
            batch_output = model.encode_docs(**batch_tkn).to("cpu")
    batch_output = batch_output.float()
    batch_output = (batch_output * 100).to(torch.int)
    batch_tokens = [[] for _ in range(len(batch_ids))]
    batch_weights = [[] for _ in range(len(batch_ids))]
    for row_col in batch_output.nonzero():
        row, col = row_col
        batch_tokens[row].append(all_tokens[col].item())
        batch_weights[row].append(batch_output[row, col].item())
    for text_id, text, tokens, weights in zip(
        batch_ids, batch_texts, batch_tokens, batch_weights
    ):
        results_to_file.append(
            {"id": text_id, "text": text,
                "vector": dict(zip(tokens, weights)), }
        )
with open(args.out, "w", encoding="UTF-8") as f:
    for result in results_to_file:
        if args.type == "query":
            rep_text = " ".join(
                Counter(result["vector"]).elements()).strip()
            if len(rep_text) > 0:
                f.write(f"{result['id']}\t{rep_text}\n")
        elif args.type == "doc":
            f.write(json.dumps(result) + "\n")
