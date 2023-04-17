import ir_datasets
from tqdm import tqdm
from pathlib import Path

print("Preparing msmarco document datasets")
msmarco_dir = Path("data/msmarco_doc")
dataset = ir_datasets.load("msmarco-document/train")

if not msmarco_dir.is_dir():
    msmarco_dir.mkdir(parents=True, exist_ok=True)
collection_path = msmarco_dir/"collection.tsv"
with open(collection_path, "w", encoding="UTF-8") as f:
    for doc in tqdm(dataset.docs_iter()):
        text = (doc.title + doc.body).replace("\n", " ").replace("\t", " ")
        f.write(f"{doc.doc_id}\t{text}\n")

train_queries_path = msmarco_dir/"msmarco-doctrain-queries.tsv"
with open(train_queries_path, "w", encoding="UTF-8") as f:
    for query in tqdm(dataset.queries_iter()):
        f.write(f"{query.query_id}\t{query.text}\n")

dev_queries_path = msmarco_dir/"msmarco-docdev-queries.tsv"
dataset = ir_datasets.load("msmarco-document/dev")
with open(dev_queries_path, "w", encoding="UTF-8") as f:
    for query in tqdm(dataset.queries_iter()):
        f.write(f"{query.query_id}\t{query.text}\n")

qrels_path = msmarco_dir/"msmarco-docdev-qrels.tsv"
with open(qrels_path, "w", encoding="UTF-8") as f:
    for qrel in tqdm(dataset.qrels_iter()):
        f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
