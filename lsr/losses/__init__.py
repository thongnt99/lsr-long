from abc import ABC
from curses import A_DIM
from torch import nn, Tensor
import torch


class Loss(nn.Module, ABC):
    """Abstract class for retrieval loss"""

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        super(Loss, self).__init__()
        self.q_regularizer = q_regularizer
        self.d_regularizer = d_regularizer

    def forward(self, *args, **kwargs):
        raise NotImplementedError("the loss function is not yet implemented")


def dot_product_ngram(a: tuple, b: tuple):
    token_dot = (a[0] * b[0]).sum(dim=-1)
    for n_gram in a[1]:
        # ids: bs x sqlen x n
        # weights: bs x sqlen
        a_ngram_ids, a_ngram_weights = a[1][n_gram]
        b_ngram_ids, b_ngram_weights = b[1][n_gram]
        token_match = (a_ngram_ids.unsqueeze(-2) == b_ngram_ids.unsqueeze(-3)).sum(
            dim=-1
        ) == n_gram
        n_gram_dot = (
            a_ngram_weights.unsqueeze(-1) * b_ngram_weights.unsqueeze(1) * token_match
        )
        n_gram_dot = n_gram_dot.max(dim=-1).values.sum(dim=-1)
        token_dot = token_dot + n_gram_dot
    return token_dot


def dot_product(a: Tensor, b: Tensor):
    """Element-wise dot product"""
    return (a * b).sum(dim=-1)


def cross_dot_product(a: Tensor, b: Tensor):
    """Return doc product between each row in a with every row in b"""
    return torch.mm(a, b.transpose(0, 1))


def cross_dot_product_ngram(a: tuple, b: tuple):
    cross_dot = cross_dot_product(a[0], b[0])
    for n_gram in a[1]:
        # ids: bs x sqlen x n
        # weights: bs x sqlen
        a_ngram_ids, a_ngram_weights = a[1][n_gram]
        b_ngram_ids, b_ngram_weights = b[1][n_gram]
        size1 = a_ngram_ids.size()
        size2 = b_ngram_ids.size()
        a_ngram_ids = a_ngram_ids.reshape(-1, size1[-1])
        b_ngram_ids = b_ngram_ids.reshape(-1, size2[-1])
        exact_match = (a_ngram_ids.unsqueeze(1) == b_ngram_ids.unsqueeze(0)).sum(
            dim=-1
        ) == n_gram  # [bs1 * seq_len1] x [bs2 * seq_len2]
        a_ngram_weights = a_ngram_weights.flatten()
        b_ngram_weights = b_ngram_weights.flatten()
        ngram_cross_dot = torch.matmul(
            a_ngram_weights.unsqueeze(-1), b_ngram_weights.unsqueeze(0)
        )
        ngram_cross_dot = ngram_cross_dot * exact_match
        ngram_cross_dot = (
            ngram_cross_dot.reshape(-1, size2[0], size2[1]).max(dim=-1).values
        )
        ngram_cross_dot = ngram_cross_dot.reshape(size1[0], size1[1], -1).sum(dim=1)
        cross_dot += ngram_cross_dot
    return cross_dot


def num_non_zero(a: Tensor):
    """Return the average number of non-zero elements for rows in a"""
    return (a > 0).float().sum(dim=1).mean()


def cal_reg_ngrams(regularizer, reps):
    if regularizer is None:
        return torch.tensor(0.0, device=reps[0].device)
    res = regularizer(reps[0])
    for n_gram in reps[1]:
        w = reps[1][n_gram][1]
        res += regularizer(w)
    return res


from .triplet_margin_loss import TripletMarginLoss
from .margin_mse_loss import MarginMSELoss
from .ngram_margin_mse_loss import NGramMarginMSELoss
from .scaled_margin_mse_loss import ScaledMarginMSELoss
from .nll import NegativeLogLikelihood
from .multiple_negative_loss import MultipleNegativeLoss
from .ngram_multiple_negative_loss import NGramMultipleNegativeLoss
from .dense_explainer_loss import DenseExplainerLoss
from .listwise_kd_mse_loss import ListwiseKDMSELoss
from .listnet_loss import ListNet
from .term_mse_loss import TermMSELoss
from .cross_entropy_loss import CrossEntropyLoss
from .negative_likelihood import NegativeLikelihood
from .distil_kl_loss import DistilKLLoss
from .distil_ngram_kl_loss import DistilNGramKLLoss
