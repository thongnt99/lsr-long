"""Defenition of different sparse encoders"""
from abc import ABC
from lib2to3.pgen2 import token

from lsr.utils.sparse_rep import SparseRep
from ..utils import get_absolute_class_name, get_class_from_str
import torch
from torch import nn
import numpy as np
import os
import json
from transformers import AutoModel, AutoModelForMaskedLM
from pathlib import Path
from ..utils import normact, pooling


class SparseEncoder(nn.Module, ABC):
    """Abstract sparse encoder"""

    def __init__(self, *args, **kwargs):
        super(SparseEncoder, self).__init__()

    def forward(self, **kwargs):
        """Abstract forward method: should be overidden by child class"""
        raise NotImplementedError("Not yet implemented")

    def save_pretrained(self, model_directory):
        """To save model to the output directory"""
        raise NotImplementedError("Not implemented yet")

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        """To load model from a directory"""
        raise NotImplementedError("Not yet implemented")


class BinaryEncoder(SparseEncoder):
    """Encoder only tokenizer, returning sparse vectors of tokens"""

    def __init__(
        self,
        vocab_size=None,
        tokenizer=None,
        remove_special_tokens=True,
        scale=0.3,
        token_level_scale=False,
    ):
        super().__init__()
        if vocab_size:
            self.vocab_size = vocab_size
        elif tokenizer:
            self.vocab_size = tokenizer.get_vocab_size()
        else:
            raise Exception(
                "Both vocab_size and tokenizer are None. One of them must be set"
            )
        self.token_level_scale = token_level_scale
        if token_level_scale:
            if isinstance(scale, list):
                self.scale = nn.Parameter(torch.tensor(scale))
            else:
                self.scale = nn.Parameter(torch.tensor([scale] * self.vocab_size))
        else:
            self.scale = nn.Parameter(torch.tensor(scale))
        self.remove_special_tokens = remove_special_tokens

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if self.token_level_scale is True:
            bin_weights = (
                self.scale[input_ids]
                * kwargs["attention_mask"]
                * (1 - kwargs["special_tokens_mask"] * self.remove_special_tokens)
            )
        else:
            bin_weights = (
                torch.ones_like(input_ids, dtype=torch.float)
                * kwargs["attention_mask"]
                * (1 - kwargs["special_tokens_mask"] * self.remove_special_tokens)
            ) * self.scale
        # # convert id sequence to sparse lexical vector
        # size = (input_ids.size(0), self.vocab_size)
        # lex_reps = torch.zeros(
        #     size, device=input_ids.device, dtype=torch.float
        # ).scatter_add_(1, input_ids, bin_weights)
        # return lex_reps
        size = torch.tensor(
            (input_ids.size(0), self.vocab_size), device=input_ids.device
        )
        return SparseRep(indices=input_ids, values=bin_weights, size=size,)

    def save_pretrained(self, model_directory):
        p = Path(model_directory)
        p.mkdir(parents=True, exist_ok=True)
        config = {
            "vocab_size": self.vocab_size,
            "remove_special_tokens": self.remove_special_tokens,
            "token_level_scale": self.token_level_scale,
            "scale": self.scale.tolist(),
        }
        config_path = os.path.join(model_directory, "config.json")
        json.dump(config, open(config_path, "w", encoding="UTF-8"))

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        config_path = os.path.join(model_name_or_dir, "config.json")
        config = json.load(open(config_path, "r", encoding="UTF-8"))
        remove_special_tokens = (
            config["remove_special_tokens"]
            if "remove_special_tokens" in config
            else True
        )
        token_level_scale = (
            config["token_level_scale"] if "token_level_scale" in config else False
        )
        scale = config["scale"] if "scale" in config else 0.3
        return cls(
            vocab_size=config["vocab_size"],
            remove_special_tokens=remove_special_tokens,
            scale=scale,
            token_level_scale=token_level_scale,
        )


class EPICTermImportance(nn.Module):
    """Component to generate term weight in EPIC model"""

    def __init__(self, dim=768) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        """forward method"""
        # inputs usually has shape: batch_size x seq_length x vector_size
        s = torch.log1p(self.softplus(self.linear(inputs)))
        return s


class EPICDocImportance(nn.Module):
    """Component to generate passage/doc weights in EPIC model"""

    def __init__(self, dim=768) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.sm = nn.Sigmoid()

    def forward(self, inputs):
        """forward function"""
        s = self.sm(self.linear(inputs))
        return s


class TransformerMLMSparseEncoder(SparseEncoder):
    """Pooling on top of transformer-based masked language model's logits"""

    def __init__(
        self,
        model_name_or_dir,
        pool=pooling.MaxPoolValue(dim=1),
        activation=torch.nn.ReLU(),
        term_importance=normact.AllOne(),
        doc_importance=normact.AllOne(),
        norm=normact.Log1P(),
        gating="no",
        remove_special_tokens=True,
    ):
        super(SparseEncoder, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_dir)
        self.activation = activation
        self.pool = pool
        self.term_importance = term_importance
        self.doc_importance = doc_importance
        self.norm = norm
        self.remove_special_tokens = remove_special_tokens
        self.gating = gating

    def forward(self, return_raw=False, **kwargs):
        special_tokens_mask = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs, output_hidden_states=True)
        # get last_hidden_states
        last_hidden_states = output.hidden_states[-1]
        term_scores = self.term_importance(last_hidden_states)
        # get cls_tokens: bs x hidden_dim
        cls_toks = output.hidden_states[-1][:, 0, :]
        doc_scores = self.doc_importance(cls_toks)

        # remove padding tokens and special tokens
        logits = (
            output.logits
            * kwargs["attention_mask"].unsqueeze(-1)
            * (1 - special_tokens_mask * self.remove_special_tokens).unsqueeze(-1)
            * term_scores
        )

        # norm default: log(1+x)
        logits = self.norm(self.activation(logits))
        # (default: max) pooling over sequence tokens
        lex_weights = self.pool(logits) * doc_scores
        if self.gating == "bow":
            input_ids = kwargs["input_ids"]
            size = (input_ids.size(0), self.model.config.vocab_size)
            gate_values = torch.zeros(
                size, device=input_ids.device, dtype=torch.float, requires_grad=False
            ).scatter(1, input_ids, torch.ones_like(input_ids, dtype=torch.float))
            lex_weights = lex_weights * gate_values
        if return_raw:
            return logits
        else:
            return SparseRep(dense=lex_weights)

    def save_pretrained(self, model_directory):
        self.model.save_pretrained(model_directory)
        print("Saving activation and pooling")
        activation_path = os.path.join(model_directory, "activation")
        torch.save(self.activation, activation_path)
        pooling_path = os.path.join(model_directory, "pooling")
        torch.save(self.pool, pooling_path)
        term_importance_path = os.path.join(model_directory, "term_importance_scorer")
        torch.save(self.term_importance, term_importance_path)
        doc_importance_path = os.path.join(model_directory, "doc_importance_scorer")
        torch.save(self.doc_importance, doc_importance_path)
        doc_importance_path = os.path.join(model_directory, "doc_importance_scorer")
        torch.save(self.doc_importance, doc_importance_path)
        norm_path = os.path.join(model_directory, "norm")
        print(f"Saving norm to {norm_path}")
        torch.save(self.norm, norm_path)
        config = {
            "remove_special_tokens": self.remove_special_tokens,
            "gating": self.gating,
        }
        config_path = os.path.join(model_directory, "config_special_token.json")
        print(f"Saving additional config to {config_path}")
        json.dump(config, open(config_path, "w"))

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        model = cls(model_name_or_dir)
        activation_path = os.path.join(model_name_or_dir, "activation")
        if Path(activation_path).is_file():
            model.activation = torch.load(activation_path)
        else:
            print(
                f"There is no activation stored at {activation_path}. Using default one {model.activation}"
            )
        pooling_path = os.path.join(model_name_or_dir, "pooling")
        if Path(pooling_path).is_file():
            model.pool = torch.load(pooling_path)
        else:
            print(
                f"There is no pooling stored at {pooling_path}. Using default one: {model.pooling}"
            )
        norm_path = os.path.join(model_name_or_dir, "norm")
        if Path(norm_path).is_file():
            model.norm = torch.load(norm_path)
        else:
            print(
                f"There is no norm stored at {norm_path}. Using default one: {model.norm}"
            )
        term_importance_path = os.path.join(model_name_or_dir, "term_importance_scorer")
        if Path(term_importance_path).is_file():
            model.term_importance = torch.load(term_importance_path)
        else:
            print(
                f"There is no term importance scorer stored at {term_importance_path}. Using default one: {model.term_importance}"
            )
        doc_importance_path = os.path.join(model_name_or_dir, "doc_importance_scorer")
        if Path(doc_importance_path).is_file():
            model.doc_importance = torch.load(doc_importance_path)
        else:
            print(
                f"There is no term importance scorer stored at {doc_importance_path}. Using default one: {model.doc_importance}"
            )
        config_path = os.path.join(model_name_or_dir, "config_special_token.json")
        if Path(config_path).is_file():
            config = json.load(open(config_path, "r", encoding="UTF-8"))
            model.remove_special_tokens = config["remove_special_tokens"]
            if "gating" in config:
                model.gating = config["gating"]
        else:
            print(
                f"There is no config stored at {config_path}. Using default one: remove_special_tokens={model.remove_special_tokens}, gating={model.gating}"
            )
        return model


class TransformerMLMSparseEncoder(SparseEncoder):
    """Pooling on top of transformer-based masked language model's logits"""

    def __init__(
        self,
        model_name_or_dir,
        pool=pooling.MaxPoolValue(dim=1),
        activation=torch.nn.ReLU(),
        term_importance=normact.AllOne(),
        doc_importance=normact.AllOne(),
        norm=normact.Log1P(),
        gating="no",
        remove_special_tokens=True,
    ):
        super(SparseEncoder, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_dir)
        self.activation = activation
        self.pool = pool
        self.term_importance = term_importance
        self.doc_importance = doc_importance
        self.norm = norm
        self.remove_special_tokens = remove_special_tokens
        self.gating = gating

    def forward(self, return_raw=False, **kwargs):
        special_tokens_mask = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs, output_hidden_states=True)
        # get last_hidden_states
        last_hidden_states = output.hidden_states[-1]
        term_scores = self.term_importance(last_hidden_states)
        # get cls_tokens: bs x hidden_dim
        cls_toks = output.hidden_states[-1][:, 0, :]
        doc_scores = self.doc_importance(cls_toks)

        # remove padding tokens and special tokens
        logits = (
            output.logits
            * kwargs["attention_mask"].unsqueeze(-1)
            * (1 - special_tokens_mask * self.remove_special_tokens).unsqueeze(-1)
            * term_scores
        )

        # norm default: log(1+x)
        logits = self.norm(self.activation(logits))
        # (default: max) pooling over sequence tokens
        lex_weights = self.pool(logits) * doc_scores
        if self.gating == "bow":
            input_ids = kwargs["input_ids"]
            size = (input_ids.size(0), self.model.config.vocab_size)
            gate_values = torch.zeros(
                size, device=input_ids.device, dtype=torch.float, requires_grad=False
            ).scatter(1, input_ids, torch.ones_like(input_ids, dtype=torch.float))
            lex_weights = lex_weights * gate_values
        if return_raw:
            return logits
        else:
            return SparseRep(dense=lex_weights)

    def save_pretrained(self, model_directory):
        self.model.save_pretrained(model_directory)
        print("Saving activation and pooling")
        activation_path = os.path.join(model_directory, "activation")
        torch.save(self.activation, activation_path)
        pooling_path = os.path.join(model_directory, "pooling")
        torch.save(self.pool, pooling_path)
        term_importance_path = os.path.join(model_directory, "term_importance_scorer")
        torch.save(self.term_importance, term_importance_path)
        doc_importance_path = os.path.join(model_directory, "doc_importance_scorer")
        torch.save(self.doc_importance, doc_importance_path)
        doc_importance_path = os.path.join(model_directory, "doc_importance_scorer")
        torch.save(self.doc_importance, doc_importance_path)
        norm_path = os.path.join(model_directory, "norm")
        print(f"Saving norm to {norm_path}")
        torch.save(self.norm, norm_path)
        config = {
            "remove_special_tokens": self.remove_special_tokens,
            "gating": self.gating,
        }
        config_path = os.path.join(model_directory, "config_special_token.json")
        print(f"Saving additional config to {config_path}")
        json.dump(config, open(config_path, "w"))

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        model = cls(model_name_or_dir)
        activation_path = os.path.join(model_name_or_dir, "activation")
        if Path(activation_path).is_file():
            model.activation = torch.load(activation_path)
        else:
            print(
                f"There is no activation stored at {activation_path}. Using default one {model.activation}"
            )
        pooling_path = os.path.join(model_name_or_dir, "pooling")
        if Path(pooling_path).is_file():
            model.pool = torch.load(pooling_path)
        else:
            print(
                f"There is no pooling stored at {pooling_path}. Using default one: {model.pooling}"
            )
        norm_path = os.path.join(model_name_or_dir, "norm")
        if Path(norm_path).is_file():
            model.norm = torch.load(norm_path)
        else:
            print(
                f"There is no norm stored at {norm_path}. Using default one: {model.norm}"
            )
        term_importance_path = os.path.join(model_name_or_dir, "term_importance_scorer")
        if Path(term_importance_path).is_file():
            model.term_importance = torch.load(term_importance_path)
        else:
            print(
                f"There is no term importance scorer stored at {term_importance_path}. Using default one: {model.term_importance}"
            )
        doc_importance_path = os.path.join(model_name_or_dir, "doc_importance_scorer")
        if Path(doc_importance_path).is_file():
            model.doc_importance = torch.load(doc_importance_path)
        else:
            print(
                f"There is no term importance scorer stored at {doc_importance_path}. Using default one: {model.doc_importance}"
            )
        config_path = os.path.join(model_name_or_dir, "config_special_token.json")
        if Path(config_path).is_file():
            config = json.load(open(config_path, "r", encoding="UTF-8"))
            model.remove_special_tokens = config["remove_special_tokens"]
            if "gating" in config:
                model.gating = config["gating"]
        else:
            print(
                f"There is no config stored at {config_path}. Using default one: remove_special_tokens={model.remove_special_tokens}, gating={model.gating}"
            )
        return model


class TransformerCLSMLPSparseEncoder(SparseEncoder):
    """Masked Language Model's head on top of CLS's token only"""

    def __init__(self, model_name_or_dir, activation=nn.ReLU(), norm=normact.Log1P()):
        super(TransformerCLSMLPSparseEncoder, self).__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_dir)
        # self.linear = nn.Linear(
        #     self.model.config.hidden_size, self.model.config.vocab_size
        # )
        self.activation = activation
        self.norm = norm

    def forward(self, **kwargs):
        _ = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs)
        # cls_tok = output.last_hidden_state[:, 0, :]
        # lex_weights = self.linear(cls_tok)
        lex_weights = output.logits[:, 0, :]
        lex_weights = self.norm(self.activation(lex_weights))
        return SparseRep(dense=lex_weights)

    def save_pretrained(self, model_directory):
        self.model.save_pretrained(model_directory)
        # linear_path = os.path.join(model_directory, "linear.pt")
        # print(f"Saving linear layer to {linear_path}")
        # torch.save(self.linear.state_dict(), linear_path)
        activation_path = os.path.join(model_directory, "activation")
        print(f"Saving activation to {activation_path}")
        torch.save(self.activation, activation_path)
        norm_path = os.path.join(model_directory, "norm")
        print(f"Saving norm to {norm_path}")
        torch.save(self.norm, norm_path)

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        model = cls(model_name_or_dir)
        # model.linear.load_state_dict(
        #     torch.load(os.path.join(model_name_or_dir, "linear.pt"))
        # )
        activation_path = os.path.join(model_name_or_dir, "activation")
        if Path(activation_path).is_file():
            model.activation = torch.load(activation_path)
        else:
            print(
                f"There is no activation stored at {activation_path}. Using default one {model.activation}"
            )
        norm_path = os.path.join(model_name_or_dir, "norm")
        if Path(norm_path).is_file():
            model.norm = torch.load(norm_path)
        else:
            print(
                f"There is no norm stored at {norm_path}. Using default one: {model.norm}"
            )
        return model


class TransformerMLPSparseEncoder(SparseEncoder):
    """Using a linear layer to generate weight for each token"""

    def __init__(
        self, model_name_or_dir, activation=nn.ReLU(), norm=normact.Log1P(), scale=1.0
    ):
        super(TransformerMLPSparseEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_dir)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)
        self.activation = activation
        self.norm = norm
        self.linear.apply(self._init_weights)
        self.scale = nn.Parameter(torch.tensor(scale))

    def _init_weights(self, module):
        """Initialize the weights (needed this for the inherited from_pretrained method to work)"""
        torch.nn.init.kaiming_normal(module.weight.data, nonlinearity="relu")

    def forward(self, to_scale=False, return_raw=False, **kwargs):
        if "topk_nnbs" in kwargs:
            kwargs.pop("topk_nnbs")
        special_tokens_mask = kwargs.pop("special_tokens_mask")
        output = self.model(**kwargs)
        tok_weights = self.linear(output.last_hidden_state).squeeze(-1)  # bs x len x 1
        tok_weights = (
            self.norm(self.activation(tok_weights))
            * kwargs["attention_mask"]
            * (1 - special_tokens_mask)
        )
        if to_scale:
            tok_weights = tok_weights * self.scale
        # mapping token weights into lexical vectors
        size = torch.tensor(
            (tok_weights.size(0), self.model.config.vocab_size),
            device=tok_weights.device,
        )
        # size = torch.tensor(
        #     (tok_weights.size(0), self.model.config.vocab_size),
        #     device=tok_weights.device,
        # )
        #        lex_weights = torch.zeros(
        #            size.tolist(), dtype=tok_weights.dtype, device=tok_weights.device
        #        ).scatter_reduce_(
        #            1, kwargs["input_ids"], tok_weights.squeeze(-1), reduce="amax"
        #        )
        #        return SparseRep(dense=lex_weights)
        if return_raw:
            return tok_weights
        else:
            return SparseRep(indices=kwargs["input_ids"], values=tok_weights, size=size)

    def save_pretrained(self, model_directory):
        self.model.save_pretrained(model_directory)
        linear_path = os.path.join(model_directory, "linear.pt")
        print(f"Saving linear layer to {linear_path}")
        torch.save(self.linear.state_dict(), linear_path)
        activation_path = os.path.join(model_directory, "activation")
        print(f"Saving activation to {activation_path}")
        torch.save(self.activation, activation_path)
        norm_path = os.path.join(model_directory, "norm")
        print(f"Saving norm to {norm_path}")
        torch.save(self.norm, norm_path)
        scale_path = os.path.join(model_directory, "scale_config.json")
        print(f"Saving scale factor to {scale_path}")
        json.dump(self.scale.item(), open(scale_path, "w", encoding="UTF-8"))

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        scale_path = os.path.join(model_name_or_dir, "scale_config.json")
        if Path(scale_path).is_file():
            print(f"Loading scale factor from {scale_path}")
            scale = json.load(open(scale_path, "r", encoding="UTF-8"))
            model = cls(model_name_or_dir, scale=scale)
        else:
            model = cls(model_name_or_dir)
        model.linear.load_state_dict(
            torch.load(os.path.join(model_name_or_dir, "linear.pt"))
        )
        activation_path = os.path.join(model_name_or_dir, "activation")
        if Path(activation_path).is_file():
            model.activation = torch.load(activation_path)
        else:
            print(
                f"There is no activation stored at {activation_path}. Using default one {model.activation}"
            )
        norm_path = os.path.join(model_name_or_dir, "norm")
        if Path(norm_path).is_file():
            model.norm = torch.load(norm_path)
        else:
            print(
                f"There is no norm stored at {norm_path}. Using default one: {model.norm}"
            )
        return model


class DualSparseEncoder(nn.Module):
    """Wrapper of query encoder and document encoder"""

    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        shared=False,
        doc_checkpoint="",
        freeze_doc=False,
    ):
        super().__init__()
        self.shared = shared
        if self.shared:
            self.encoder = query_encoder
        else:
            self.query_encoder = query_encoder
            self.doc_encoder = doc_encoder
            if doc_checkpoint:
                self.doc_encoder = self.doc_encoder.from_pretrained(doc_checkpoint)
            if freeze_doc:
                for param in self.doc_encoder.parameters():
                    param.requires_grad = False

    def encode_queries(self, to_dense=True, **queries):
        """To encode queries with right encoder"""
        if to_dense:
            if self.shared:
                return self.encoder(**queries).to_dense(reduce="sum")
            else:
                return self.query_encoder(**queries).to_dense(reduce="sum")
        else:
            if self.shared:
                return self.encoder(**queries)
            else:
                return self.query_encoder(**queries)

    def encode_docs(self, to_dense=True, **docs):
        """To encode docs with right encoder"""
        if to_dense:
            if self.shared:
                return self.encoder(**docs).to_dense(reduce="amax")
            else:
                return self.doc_encoder(**docs).to_dense(reduce="amax")
        else:
            if self.shared:
                return self.encoder(**docs)
            else:
                return self.doc_encoder(**docs)

    def forward(self, loss, queries, docs_batch, labels=None):
        """common forward for both query and document"""
        q_reps = self.encode_queries(**queries)
        docs_batch_rep = self.encode_docs(**docs_batch)
        if labels is None:
            output = loss(q_reps, docs_batch_rep)
        else:
            output = loss(q_reps, docs_batch_rep, labels)
        return output

    def save_pretrained(self, model_dir):
        """Save query and document encoder"""
        if self.shared:
            class_config = {
                "shared": self.shared,
                "encoder_class": get_absolute_class_name(self.encoder),
            }
            encoder_dir_or_name = f"{model_dir}/shared_encoder"
            self.encoder.save_pretrained(encoder_dir_or_name)
        else:
            class_config = {
                "shared": self.shared,
                "query_encoder_class": get_absolute_class_name(self.query_encoder),
                "doc_encoder_class": get_absolute_class_name(self.doc_encoder),
            }
            query_dir_or_name = f"{model_dir}/query_encoder"
            doc_dir_or_name = f"{model_dir}/doc_encoder"
            self.query_encoder.save_pretrained(query_dir_or_name)
            self.doc_encoder.save_pretrained(doc_dir_or_name)
        class_config_path = os.path.join(model_dir, "class_config.json")
        json.dump(class_config, open(class_config_path, "w", encoding="UTF-8"))

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        """Load query and doc encoder from a directory"""
        class_config = json.load(
            open(
                os.path.join(model_dir_or_name, "class_config.json"),
                "r",
                encoding="UTF-8",
            )
        )
        if "shared" in class_config and class_config["shared"]:
            shared_dir_or_name = f"{model_dir_or_name}/shared_encoder"
            encoder_class = get_class_from_str(class_config["encoder_class"])
            encoder = encoder_class.from_pretrained(shared_dir_or_name)
            return cls(encoder, shared=True)
        else:
            query_dir_or_name = f"{model_dir_or_name}/query_encoder"
            doc_dir_or_name = f"{model_dir_or_name}/doc_encoder"
            query_encoder_class = get_class_from_str(
                class_config["query_encoder_class"]
            )
            doc_encoder_class = get_class_from_str(class_config["doc_encoder_class"])
            query_encoder = query_encoder_class.from_pretrained(query_dir_or_name)
            doc_encoder = doc_encoder_class.from_pretrained(doc_dir_or_name)
            return cls(query_encoder, doc_encoder, shared=False)
