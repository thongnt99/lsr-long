from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


class MutiPSGBatching:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        queries = []
        psgs = []
        psg_offsets = []
        offset = 0
        for query, pos_psgs, neg_psgs in batch:
            num_pos = len(pos_psgs)
            num_neg = len(neg_psgs)
            queries.extend([query] * num_pos)
            psgs.extend(pos_psgs)
            psg_offsets.append(offset)
            offset += num_pos
            queries.extend([query] * num_neg)
            psgs.extend(neg_psgs)
            psg_offsets.append(offset)
            offset += num_neg
        psg_offsets.append(offset)
        queries = self.tokenizer(
            queries,
            truncation=True,
            padding=True,
            max_length=400,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        psgs = self.tokenizer(
            psgs,
            truncation=True,
            padding=True,
            max_length=400,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        return {"queries": queries, "psgs": psgs, "psg_offset": psg_offsets}


class MultiPSGDataset(Dataset):
    def __init__(self, query_path, collection_path, triplet_path, num_psg=2):
        self.queries = {}

        with open(query_path, "r", encoding="UTF-8") as f:
            for line in tqdm(f, desc=f"Reading queries from {query_path}"):
                qid, qtext = line.strip().split("\t")
                self.queries[qid] = qtext
        self.docs = defaultdict(dict)
        with open(collection_path, "r", encoding="UTF-8") as f:
            for line in tqdm(f, desc=f"Reading collection path from {collection_path}"):
                did_pid, ptext = line.strip().split("\t")
                did, pid = did_pid.split("-")
                self.docs[did][pid] = ptext
        self.triplets = []
        with open(triplet_path, "r") as f:
            for line in tqdm(f, desc=f"Reading triplet file {triplet_path}"):
                qid, pos_id, neg_id = line.strip().split("\t")
                self.triplets.append((qid, pos_id, neg_id))
        self.num_psg = num_psg

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        qid, pos_id, neg_id = self.triplets[index]
        query = self.queries[qid]
        pos_psgs_keys = sorted(self.docs[pos_id].keys())[: self.num_psg]
        neg_psgs_keys = sorted(self.docs[neg_id].keys())[: self.num_psg]
        pos_psgs = [self.docs[pos_id][psg_id] for psg_id in pos_psgs_keys]
        neg_psgs = [self.docs[neg_id][psg_id] for psg_id in neg_psgs_keys]
        return query, pos_psgs, neg_psgs
