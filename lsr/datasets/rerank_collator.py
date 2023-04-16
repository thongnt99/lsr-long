import torch

from lsr.datasets.longdoc_collator import split_and_collate_doc


class RerankCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer, q_max_length, d_max_length):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        query_ids = []
        doc_ids = []
        for (qid, query, did, doc) in batch:
            query_ids.append(qid)
            doc_ids.append(did)
            batch_queries.append(query)
            batch_docs.append(doc)

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return {
            "queries": tokenized_queries,
            "docs": tokenized_docs,
            "query_ids": query_ids,
            "doc_ids": doc_ids,
        }


class RerankLongDocCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer, q_max_length, d_max_length):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        query_ids = []
        doc_ids = []
        for (qid, query, did, doc) in batch:
            query_ids.append(qid)
            doc_ids.append(did)
            batch_queries.append(query)
            batch_docs.append(doc)

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_docs = split_and_collate_doc(
            batch_docs, passage_max_length=self.d_max_length
        )
        # tokenized_docs = self.tokenizer(
        #     batch_docs,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.d_max_length,
        #     return_tensors="pt",
        #     return_special_tokens_mask=True,
        # )
        return {
            "queries": tokenized_queries,
            "docs": tokenized_docs,
            "query_ids": query_ids,
            "doc_ids": doc_ids,
        }
