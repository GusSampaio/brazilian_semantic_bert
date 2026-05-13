from datasets import Dataset, DatasetDict

from parser import PBP_parser
from instance_builder import SRLInstanceBuilder
from srl_tokenizer import SRLTokenizer
from splitter import SRLSplitter


class SRLDataModule:

    def __init__(
        self,
        dataset_path="PBP-classic-complete.conllu",
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
        self.dataset_path = dataset_path

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

        self.builder = SRLInstanceBuilder(
            predicate_signal=self.predicate_signal
        )

        self.splitter = SRLSplitter(
            train_ratio=self.train_ratio,
            dev_ratio=self.dev_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

        self.datasets = self.prepare_data(self.dataset_path)

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

        self.tokenizer = SRLTokenizer(
            model_name=self.model_name,
            label2id=self.label2id,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            using_special_tokens=(self.predicate_signal == "special_token"),
        )

        train_tokenized = self._tokenize_split(train_instances)
        dev_tokenized = self._tokenize_split(dev_instances)
        test_tokenized = self._tokenize_split(test_instances)


        hf_datasets = DatasetDict({
            "train": Dataset.from_list(train_tokenized),
            "validation": Dataset.from_list(dev_tokenized),
            "test": Dataset.from_list(test_tokenized),
        })

        return hf_datasets
    
    def _tokenize_split(self, instances):
        return [
            self.tokenizer.tokenize_and_align(inst)
            for inst in instances
        ]

    def get_dataset(self, split="train"):
        return self.datasets[split]