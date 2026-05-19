from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import os
import json

from data.conllu_parser import PBP_parser
from data.instance_builder import SRLInstanceBuilder
from data.splitter import SRLSplitter

from features.srl_tokenizer import SRLTokenizer

class SRLDataModule:

    def __init__(
        self,
        use_preprocessed_data=False,
        save_data=False,
        data_path="data/processed/",
        raw_dataset_path="data/raw/PBP-classic-complete.conllu",
        model_name="neuralmind/bert-base-portuguese-cased",
        predicate_signal="special_token",
        max_length=256,
        padding="max_length",
        truncation=True,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    ):
        self.raw_dataset_path = raw_dataset_path
        self.data_path = data_path

        self.use_preprocessed_data = use_preprocessed_data

        self.model_name = model_name
        self.predicate_signal = predicate_signal

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

        self.seed = seed

        self.label2id = None
        self.id2label = None

        if self.use_preprocessed_data:
            self.datasets = DatasetDict.load_from_disk(self.data_path + "data_splits")
            self.get_labels_and_ids()
            self.get_tokenizer()

        else:
            self.builder = SRLInstanceBuilder(
                predicate_signal=self.predicate_signal
            )
            
            self.splitter = SRLSplitter(
                train_ratio=self.train_ratio,
                dev_ratio=self.dev_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )

            self.datasets = self.prepare_data(self.raw_dataset_path)

            if save_data and self.data_path is not None:
                splits_path = os.path.join(self.data_path, "data_splits")
                self.datasets.save_to_disk(splits_path)

                label_and_ids_path = os.path.join(self.data_path, "labels_and_ids/")
                os.makedirs(label_and_ids_path, exist_ok=True)
                with open(label_and_ids_path + "label2id.json", "w") as f:
                    json.dump(self.label2id, f)
                with open(label_and_ids_path + "id2label.json", "w") as f:
                    json.dump(self.id2label, f)

    def get_labels_and_ids(self):
        label_and_ids_path = os.path.join(self.data_path, "labels_and_ids/")
        
        with open(label_and_ids_path + 'label2id.json', 'r') as file:
            self.label2id = json.load(file)
        
        with open(label_and_ids_path + 'id2label.json', 'r') as file:
            self.id2label = json.load(file)

    def get_tokenizer(self):
        self.tokenizer = SRLTokenizer(
            model_name=self.model_name,
            label2id=self.label2id,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            using_special_tokens=(self.predicate_signal == "special_token"),
        )

    def build_label_vocab(self, instances):

        unique_labels = set()

        for inst in instances:
            labels = inst.get("model_labels", inst["labels"])

            for label in labels:
                # ignore masked labels
                if label == -100:
                    continue

                unique_labels.add(label)

        unique_labels = sorted(unique_labels)

        label2id = {
            label: idx
            for idx, label in enumerate(unique_labels)
        }

        id2label = {
            idx: label
            for label, idx in label2id.items()
        }

        return label2id, id2label

    def prepare_data(self, file_path):
        instances = PBP_parser(file_path)
        instances = self.builder.build(instances)

        self.label2id, self.id2label = (
            self.build_label_vocab(instances)
        )

        train_instances, dev_instances, test_instances = self.splitter.split(instances)

        self.get_tokenizer()

        train_tokenized = self._tokenize_split(train_instances)
        dev_tokenized = self._tokenize_split(dev_instances)
        test_tokenized = self._tokenize_split(test_instances) 

        hf_datasets = DatasetDict({
            "train": Dataset.from_list(train_tokenized),
            "validation": Dataset.from_list(dev_tokenized),
            "test": Dataset.from_list(test_tokenized),
        })

        hf_datasets.set_format(
            type="torch",
            columns=[
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "labels",
                "predicate_indicator",
            ]
        )

        return hf_datasets
    
    def _tokenize_split(self, instances):
        return [
            self.tokenizer.tokenize_and_align(inst)
            for inst in instances
        ]

    def get_split(self, split="train"):
        return self.datasets[split]
    
    def get_dataloader(
        self,
        dataset,
        batch_size=2,
        shuffle=False,
    ):
        """
        Creates a PyTorch DataLoader from a HuggingFace dataset.
        """

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )