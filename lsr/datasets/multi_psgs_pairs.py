from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import ir_measures
import torch


class MutiPSGsPairsBatching:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        query_ids = []
        doc_ids = []
        queries = []
        psgs = []
        psg_offset = []
        offset = 0
        for qid, query_text, doc_id, doc_psgs in batch:
            psg_offset.append(offset)
            offset += len(doc_psgs)
            query_ids.append(qid)
            doc_ids.append(doc_id)
            queries.extend([query_text] * len(doc_psgs))
            psgs.extend(doc_psgs)

        queries = self.tokenizer(
            queries,
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
            max_length=400,
            return_tensors="pt",
        )

        psgs = self.tokenizer(
            psgs,
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
            max_length=400,
            return_tensors="pt",
        )
        psg_offset.append(offset)
        return {
            "queries": queries,
            "psgs": psgs,
            "psg_offset": torch.tensor(psg_offset),
            "query_ids": query_ids,
            "doc_ids": doc_ids,
        }


class MultiPSGsPairs(Dataset):
    def __init__(
        self,
        collection_path: str = "*",
        query_path: str = "",
        run_path: str = "",
        qrel_path: str = "",
        top_k: int = 100,
        num_psgs=2,
        sep: str = "@@",
    ) -> None:
        super().__init__()
        self.docs = defaultdict(dict)
        with open(collection_path, "r") as f:
            for line in tqdm(
                f, desc=f"Reading document collection from {collection_path}"
            ):
                try:
                    did_psg_id, dtext = line.strip().split("\t")
                    did, psg_id = did_psg_id.split(sep)
                    self.docs[did][psg_id] = dtext
                except:
                    pass
                    # print(line)

        self.queries = {}
        with open(query_path, "r") as f:
            for line in tqdm(f, desc=f"Loading queries from {query_path}"):
                qid, qtext = line.strip().split("\t")
                self.queries[qid] = qtext

        candidates = defaultdict(list)
        with open(run_path, "r") as f:
            for line in tqdm(f, desc=f"Reading run file for reranking from {run_path}"):
                qid, _, doc_id, *_ = line.strip().split(" ")
                if len(candidates[qid]) < top_k:
                    candidates[qid].append(doc_id)
        self.pairs = []
        for qid in candidates:
            for did in candidates[qid]:
                self.pairs.append((qid, did))

        self.qrels = [
            item
            for item in tqdm(
                ir_measures.read_trec_qrels(qrel_path), desc="Reading qrels"
            )
        ]
        self.num_psgs = num_psgs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, did = self.pairs[idx]
        psgs_id = sorted(self.docs[did].keys())[: self.num_psgs]
        psgs = [self.docs[did][psg_id] for psg_id in psgs_id]
        return qid, self.queries[qid], did, psgs
