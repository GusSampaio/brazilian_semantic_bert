import torch
from transformers import AutoModelForTokenClassification


class SRLModel:

    def __init__(
        self,
        model_name,
        tokenizer,
        num_labels,
        id2label,
        label2id,
    ):

        self.model = (
            AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
        )

        self.model.resize_token_embeddings(len(tokenizer.tokenizer))