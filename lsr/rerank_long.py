import argparse
import logging
from collections import Counter, defaultdict
from lsr.datasets.multi_psgs_pairs import MultiPSGsPairs, MutiPSGsPairsBatching
from lsr.models.exact_sdm_reranker_long import DualSparseEncoder
from tqdm import tqdm
import ir_measures
from ir_measures import *
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reranking SMD")
parser = argparse.ArgumentParser("Arguments for reranking")
parser.add_argument(
    "-run",
    type=str,
    default="run_files_top200/robust04/run_max_score_1.trec",
    help="BM25 run file",
)
parser.add_argument(
    "-q",
    default="data/trec-robust04/desc-queries.tsv",
    type=str,
    help="Path to query file",
)
parser.add_argument(
    "-d",
    default="data/trec-robust04/docs/collection_psg.tsv",
    type=str,
    help="Path to doc file",
)
parser.add_argument(
    "-npsg", type=int, default=1, help="Number of passages per document"
)
parser.add_argument(
    "-sep",
    type=str,
    default="@@",
    help="The seperator between document id and passage id",
)
parser.add_argument("-qrel", type=str,
                    default="data/trec-robust04/robust04.qrels")
parser.add_argument(
    "-cp",
    type=str,
    default="outputs/reranker_exact_qmlp_dmlm_msmarco_doc_ce_1_psg/model",
    help="The seperator between document id and passage id",
)
parser.add_argument(
    "-bs", default=50, type=int, help="batch size",
)
parser.add_argument(
    "-topk", default=100, type=int, help="batch size",
)
args = parser.parse_args()
logger.info("Loading tokenizer and model")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = DualSparseEncoder.from_pretrained(args.cp).to("cuda:0")
# model.window_sizes = [1, 2]
# model.proximity = 30
# model.linear_sum.weight.data = torch.tensor([[0.164, 0.2775, -0.2986]]).to("cuda:0")
eval_dataset = MultiPSGsPairs(
    collection_path=args.d,
    query_path=args.q,
    run_path=args.run,
    qrel_path=args.qrel,
    top_k=args.topk,
    num_psgs=args.npsg,
    sep=args.sep,
)

collator = MutiPSGsPairsBatching(tokenizer=tokenizer)
data_loader = DataLoader(
    eval_dataset, batch_size=args.bs, collate_fn=collator, num_workers=16, shuffle=False
)
rerank_run = defaultdict(dict)
model.eval()
for batch in tqdm(data_loader, desc="Evaluating the model"):
    qids = batch.pop("query_ids")
    dids = batch.pop("doc_ids")
    batch = {k: v.to("cuda:0") for k, v in batch.items()}
    with torch.no_grad(), torch.cuda.amp.autocast():
        scores = model.score_pairs(**batch).tolist()
    for qid, did, score in zip(qids, dids, scores):
        rerank_run[qid][did] = score
qrels = eval_dataset.qrels
metrics = ir_measures.calc_aggregate(
    [nDCG @ 10, MRR @ 10, R @ 1000], qrels, rerank_run)
file_name = f"model_{args.cp.split('/')[1]}_num_psgs_{args.npsg}_window_sizes_{'-'.join([str(w) for w in model.window_sizes])}_proximity_{model.proximity}.json"
output_run = args.run.replace(".trec", file_name)
json.dump(rerank_run, open(output_run, "w"))
print(metrics)
