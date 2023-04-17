from typing import Dict, List, Optional
import torch
import os
from tqdm import tqdm
import transformers
import logging
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import ir_measures
from ir_measures import *
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HFTrainer(transformers.trainer.Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(
        self,
        *args,
        eval_collator=None,
        data_type="triple",
        train_only_bias_and_layer_norm=False,
        **kwargs,
    ) -> None:
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.data_type = data_type
        # self.total_q_length = 0
        # self.total_d_length = 0
        # self.total_q_reg = 0
        # self.total_d_reg = 0
        # self.total_loss_without_reg = 0
        self.customed_log = defaultdict(lambda: 0.0)
        self.train_only_bias_and_layer_norm = train_only_bias_and_layer_norm
        self.eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_collator
        eval_sampler = self._get_eval_sampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        data_loader = self.get_eval_dataloader(eval_dataset)
        rerank_run = defaultdict(dict)
        self.model.eval()
        for batch in tqdm(data_loader, desc="Evaluating the model"):
            qids = batch.pop("query_ids")
            dids = batch.pop("doc_ids")
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            # queries = batch["queries"].to(self.args.device)
            # docs = batch["docs"].to(self.args.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                scores = self.model.score_pairs(**batch).tolist()
            for qid, did, score in zip(qids, dids, scores):
                rerank_run[qid][did] = score
        self.model.train()
        run_file_path = (
            Path(self.args.output_dir)
            / f"run_dev_windows_{'-'.join([str(ws) for ws in self.model.window_sizes])}_proximity_{self.model.proximity}_step_{self.state.global_step}.json"
        )
        logger.info(f"Writing run file to: {run_file_path}")
        json.dump(rerank_run, open(run_file_path, "w"))
        qrels = (
            eval_dataset.qrels if eval_dataset is not None else self.eval_dataset.qrels
        )
        metrics = ir_measures.calc_aggregate(
            [nDCG @ 10, MRR @ 10, R @ 1000], qrels, rerank_run
        )
        metrics = {metric_key_prefix + "_" +
                   str(k): v for k, v in metrics.items()}
        metrics["epoch"] = self.state.epoch
        self.log(metrics)
        return metrics

    # return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            log = {}
            for metric in self.customed_log:
                log[metric] = (
                    self._nested_gather(
                        self.customed_log[metric]).mean().item()
                )
                log[metric] = round(
                    (
                        log[metric]
                        / (self.state.global_step - self._globalstep_last_logged)
                        / self.args.gradient_accumulation_steps
                    ),
                    4,
                )
            self.log(log)
            for metric in self.customed_log:
                self.customed_log[metric] -= self.customed_log[metric]
            self.control.should_log = True
        super()._maybe_log_save_evaluate(
            tr_loss, model, trial, epoch, ignore_keys_for_eval
        )

    def create_optimizer(self):
        """Setup the optimizer"""
        if not self.train_only_bias_and_layer_norm:
            return super().create_optimizer()
        if self.optimizer is None:
            params_to_optimize = []
            for name, p in self.model.named_parameters():
                if (
                    "bias" in name
                    or "layer_norm" in name
                    or "linear" in name
                    or "LayerNorm" in name
                ):
                    params_to_optimize.append(p)
                else:
                    p.requires_grad = False
            optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                params_to_optimize, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """Computing loss with diffent formats of training data"""
        loss_output, q_reg, d_reg, to_log = model(**inputs)
        for log_metric in to_log:
            self.customed_log[log_metric] += to_log[log_metric]
        return loss_output + q_reg + d_reg

    def save_model(self, model_dir=None, _internal_call=False):
        """Save model checkpoint"""
        logger.info("Saving model checkpoint to %s", model_dir)
        if model_dir is None:
            model_dir = os.path.join(self.args.output_dir, "model")
        self.model.save_pretrained(model_dir)
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        self.data_collator.tokenizer.save_pretrained(tokenizer_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from a checkpoint to continue traning"""
        # Load model from checkpoint
        logger.info("Loading model from checkpoint %s", resume_from_checkpoint)
        self.model.load_state_dict(
            self.model.from_pretrained(resume_from_checkpoint).state_dict()
        )
