from transformers import Trainer

from src.semantic_role_labeling.training.losses import (
    FocalLoss,
    TokenCrossEntropyLoss,
)


class SRLTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_strategy: str = "baseline",
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_strategy == "baseline":
            self.loss_function = TokenCrossEntropyLoss(ignore_index=-100)
        elif loss_strategy == "focal_loss":
            self.loss_function = FocalLoss(
                gamma=focal_gamma,
                ignore_index=-100,
            )
        else:
            raise ValueError("Loss strategy must be 'baseline' or 'focal_loss'.")

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        loss = self.loss_function(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss