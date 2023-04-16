from torch import nn
import torch
from lsr.losses import Loss, dot_product, num_non_zero


class DistilKLLoss(Loss):
    """Implementation of KL loss for distillation"""

    def __init__(self, q_regularizer=None, d_regularizer=None) -> None:
        super(DistilKLLoss, self).__init__(q_regularizer, d_regularizer)
        self.loss = torch.nn.KLDivLoss(reduction="none")

    def forward(self, q_reps, d_reps, labels):
        batch_size = q_reps.size(0)
        p_reps, n_reps = d_reps.view(batch_size, 2, -1).transpose(0, 1)
        teacher_scores = torch.softmax(labels.view(batch_size, 2), dim=1)
        # similarity with negative documents
        p_rel = dot_product(q_reps, p_reps)
        # similarity with positive documents
        n_rel = dot_product(q_reps, n_reps)
        student_scores = torch.stack([p_rel, n_rel], dim=1)
        student_scores = torch.log_softmax(student_scores, dim=1)
        reg_q_output = (
            torch.tensor(0.0, device=q_reps.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_reps)
        )
        reg_d_output = (
            torch.tensor(0.0, device=p_reps.device)
            if (self.d_regularizer is None)
            else (self.d_regularizer(p_reps) + self.d_regularizer(n_reps)) / 2
        )
        kl_loss = self.loss(student_scores, teacher_scores).sum(dim=1).mean(dim=0)
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()
        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
            "loss_no_reg": kl_loss.detach(),
        }
        return (kl_loss, reg_q_output, reg_d_output, to_log)
