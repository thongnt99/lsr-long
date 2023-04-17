from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class TripletIdsDataset(Dataset):
    """BM25 triplets of (query_id, pos_id, neg_id)"""

    def __init__(self, triple_ids_path: str, queries_path: str, docs_path: str) -> None:
        super().__init__()
        self.triple_lists = []
        docs_dict = {}
        queries_dict = {}
        with open(docs_path) as f:
            for line in tqdm(f, desc=f"Loading docs collection from {docs_path}"):
                try:
                    docid, doctext = line.strip().split("\t")
                    docs_dict[docid] = doctext
                except:
                    pass
        with open(queries_path) as f:
            for line in tqdm(f, desc=f"Loading queries from {queries_path}"):
                qid, qtext = line.strip().split("\t")
                queries_dict[qid] = qtext
        with open(triple_ids_path) as f:
            for line in tqdm(f, desc=f"Loading id triples from {triple_ids_path}"):
                qid, pos_id, neg_id = line.strip().split("\t")
                try:
                    assert qid in queries_dict
                    assert pos_id in docs_dict
                    assert neg_id in docs_dict
                    self.triple_lists.append(
                        (queries_dict[qid], [docs_dict[pos_id], docs_dict[neg_id]])
                    )
                except:
                    pass

    def __getitem__(self, idx):
        return self.triple_lists[idx]

    def __len__(self):
        return len(self.triple_lists)


class TripletTextDataset(Dataset):
    "BM25 triplets of (query_text, pos_text, neg_text)"

    def __init__(self, data_path) -> None:
        super().__init__()
        print(data_path)
        self.triple_lists = []
        with open(data_path, "r") as f:
            for line in tqdm(f, f"Loading triplets from {data_path}"):
                q, pos, neg = line.strip().split("\t")
                self.triple_lists.append((q, pos, neg))

    def __getitem__(self, idx):
        return self.triple_lists[idx]

    def __len__(self):
        return len(self.triple_lists)
