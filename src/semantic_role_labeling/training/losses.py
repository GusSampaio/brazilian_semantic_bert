import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        if targets.numel() == 0:
            return logits.sum() * 0.0

        target_log_probabilities = F.log_softmax(logits.float(), dim=-1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        target_probabilities = target_log_probabilities.exp().clamp(max=1.0)
        losses = -(
            (1.0 - target_probabilities).clamp_min(0.0).pow(self.gamma)
            * target_log_probabilities
        )

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses


class TokenCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, logits, targets):
        return super().forward(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )