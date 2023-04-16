from torch import nn
from lsr.losses import dot_product, num_non_zero
from lsr.models import TransformerMLMSparseEncoder, TransformerMLPSparseEncoder
import torch
from pathlib import Path
import json


class DualSparseEncoder(nn.Module):
    """Wrapper of query encoder and document encoder"""

    def __init__(
        self,
        query_encoder: TransformerMLPSparseEncoder,
        doc_encoder: TransformerMLMSparseEncoder,
        disable_gradient: bool = True,
        window_sizes: list = [1, 3],
        proximity: int = 8,
        reg_weight: float = 0.0,
    ):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        if disable_gradient:
            self.disable_gradient(self.query_encoder)
            self.disable_gradient(self.doc_encoder)
        # self.window_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
        # 0.8003,  0.0154,  0.0367, -0.0044,  0.1180, -0.0242,  0.0058
        self.window_sizes = list(
            window_sizes
        )  # convert from ListConfig to list. Important for JSON serializable to work
        self.proximity = proximity
        self.linear_sum = nn.Linear(len(self.window_sizes) * 2 - 1, 1)
        self.loss = torch.nn.CrossEntropyLoss()
        self.reg_weight = reg_weight

    def disable_gradient(self, m):
        for param in m.parameters():
            param.requires_grad = False

    def enable_mlm_gradient(self):
        for param in self.doc_encoder.model.vocab_projector.parameters():
            param.requires_grad = True

    def encode_queries(self, **queries):
        """To encode queries with right encoder"""
        return self.query_encoder(**queries, return_raw=True)

    def encode_docs(self, **docs):
        """To encode docs with right encoder"""
        return self.doc_encoder(**docs, return_raw=True)

    def score_pairs(self, queries, psgs, psg_offset):
        q_tok_ids = queries["input_ids"]
        q_tok_weights = self.encode_queries(**queries)
        d_tok_ids = psgs["input_ids"]
        d_logits = self.encode_docs(**psgs)
        return self._score_pairs(
            q_tok_ids, q_tok_weights, d_tok_ids, d_logits, psg_offset
        )

    def _score_pairs(self, q_tok_ids, q_tok_weights, d_tok_ids, d_logits, psg_offset):
        batch_size, q_len = q_tok_ids.size()
        _, d_len = d_tok_ids.size()
        # batch_size x q_len x
        # batch_size x d_len x q_len
        trans_matrix = (
            q_tok_weights.unsqueeze(-1)
            * d_logits[torch.arange(batch_size).unsqueeze(1), :, q_tok_ids]
        )
        mask = torch.zeros(d_logits.size(), device=d_logits.device)
        # batch_size x sequence x logits
        mask[
            torch.arange(batch_size).unsqueeze(1),
            torch.arange(d_logits.size(1)).unsqueeze(0),
            d_tok_ids,
        ] = 1.0
        d_logits = d_logits * mask
        exact_matrix = (
            q_tok_weights.unsqueeze(-1)
            * d_logits[torch.arange(batch_size).unsqueeze(1), :, q_tok_ids]
        )
        # unigram soft matching
        all_scores = [trans_matrix.max(dim=2).values.sum(dim=1)]
        # exact ngram/phrase matching
        for wsize in self.window_sizes[1:]:
            score_matrix = exact_matrix
            mask = (exact_matrix > 0).float()
            for idx in range(1, wsize):
                score_matrix = score_matrix[:, :-1, :-1] + exact_matrix[:, idx:, idx:]
                mask = mask[:, :-1, :-1] * (exact_matrix[:, idx:, idx:] > 0).float()
            score = (score_matrix * mask).max(dim=2).values.sum(dim=1)
            all_scores.append(score)

        prox_scores = exact_matrix
        for idx in range(1, self.proximity):
            prox_scores = torch.max(prox_scores[:, :, :-1], exact_matrix[:, :, idx:])
        # unorder matching: is the same as ordered matching with n_gram=1
        for wsize in self.window_sizes[1:]:
            score_matrix = prox_scores
            mask = (prox_scores > 0).float()
            for idx in range(1, wsize):
                score_matrix = score_matrix[:, :-1, :] + prox_scores[:, idx:, :]
                mask = mask[:, :-1, :] * (prox_scores[:, idx:, :] > 0).float()
            score = (score_matrix * mask).max(dim=2).values.sum(dim=1)
            all_scores.append(score)
        all_scores = torch.stack(all_scores, dim=1)
        # pool segments
        max_psg_scores = []
        for idx in range(len(psg_offset) - 1):
            max_psg_scores.append(
                all_scores[psg_offset[idx] : psg_offset[idx + 1]].max(dim=0).values
            )
        max_psg_scores = torch.stack(max_psg_scores, dim=0)
        final_scores = self.linear_sum(max_psg_scores).squeeze(dim=1)
        return final_scores

    def forward(self, loss, queries, psgs, psg_offset, labels=None):
        """common forward for both query and document"""
        scores = self.score_pairs(queries, psgs, psg_offset).reshape(-1, 2)
        labels = torch.zeros(scores.size(0), dtype=int, device=scores.device)
        ce_loss = self.loss(scores, labels)
        w_list = self.linear_sum.weight.data[0]
        names = [f"O_{ws}" for ws in self.window_sizes] + [
            f"U_{ws}_p_{self.proximity}" for ws in self.window_sizes[1:]
        ]
        to_log = dict(zip(names, w_list))
        # to_log = {
        #     "loss_without_reg": ce_loss.detach(),
        #     "query_len": (q_tok_weights > 0).float().sum(dim=-1).mean(),
        #     "doc_len": (d_logits > 0).float().sum(dim=-1).sum(dim=-1).mean(),
        # }
        return (ce_loss, 0, 0, to_log)

    def save_pretrained(self, model_dir):
        """Save query and document encoder"""
        query_dir_or_name = f"{model_dir}/query_encoder"
        doc_dir_or_name = f"{model_dir}/doc_encoder"
        self.query_encoder.save_pretrained(query_dir_or_name)
        self.doc_encoder.save_pretrained(doc_dir_or_name)
        linear_sum_path = Path(model_dir) / "linear_sum.pt"
        torch.save(self.linear_sum.state_dict(), linear_sum_path)
        sdm_config = {"window_sizes": self.window_sizes, "proximity": self.proximity}
        config_path = Path(model_dir) / "config.json"
        json.dump(sdm_config, open(config_path, "w"))

    @classmethod
    def from_pretrained(
        cls, model_dir_or_name,
    ):
        """Load query and doc encoder from a directory"""
        query_dir_or_name = f"{model_dir_or_name}/query_encoder"
        doc_dir_or_name = f"{model_dir_or_name}/doc_encoder"
        query_encoder = TransformerMLPSparseEncoder.from_pretrained(query_dir_or_name)
        doc_encoder = TransformerMLMSparseEncoder.from_pretrained(doc_dir_or_name)
        config_path = Path(model_dir_or_name) / "config.json"
        if config_path.is_file():
            sdm_config = json.load(open(config_path, "r"))
        else:
            sdm_config = {}
        model = cls(query_encoder, doc_encoder, **sdm_config)
        linear_sum_path = Path(model_dir_or_name) / "linear_sum.pt"
        if linear_sum_path.is_file():
            model.linear_sum.load_state_dict(torch.load(linear_sum_path))
        return model
