import random
from collections import defaultdict


class SRLSplitter:
    """
    Splits the dataset into train/dev/test sets based on sentence IDs. 
    All sentences with the same sentence ID will be in the same split.
    """

    def __init__(
        self,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    ):

        total = train_ratio + dev_ratio + test_ratio

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "train/dev/test ratios must sum to 1"
            )

        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, dataset):
        
        unique_sentences = sorted(
            list(set(inst["sentence_id"] for inst in dataset))
        )

        rng = random.Random(self.seed)
        rng.shuffle(unique_sentences)
        n_docs = len(unique_sentences)

        # All sentences in the dataset that have the same sentence_id should go to the same split
        sentence_to_split = {}

        for i, sentence_id in enumerate(unique_sentences):
            if i < int(n_docs * self.train_ratio):
                sentence_to_split[sentence_id] = "train"
            elif i < int(n_docs * (self.train_ratio + self.dev_ratio)+1):
                sentence_to_split[sentence_id] = "dev"
            else:
                sentence_to_split[sentence_id] = "test"

        # Divinding sentences based on the sentence_to_split mapping
        train_set = [inst for inst in dataset if sentence_to_split[inst["sentence_id"]] == "train"]
        dev_set = [inst for inst in dataset if sentence_to_split[inst["sentence_id"]] == "dev"]
        test_set = [inst for inst in dataset if sentence_to_split[inst["sentence_id"]] == "test"]

        return train_set, dev_set, test_set 