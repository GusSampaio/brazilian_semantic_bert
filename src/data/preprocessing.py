from parser import PBP_parser
from dataset_builder import SRLDatasetBuilder
from srl_tokenizer import SRLTokenizer

def build_label_vocab(instances):

    unique_labels = set()

    for inst in instances:
        unique_labels.update(inst["labels"])

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


instances = PBP_parser("../../PBP-classic-complete.conllu")

builder = SRLDatasetBuilder(
    predicate_signal="binary_feature"
)

dataset = builder.build(instances)

print("PASSOU SÓ PELO BUILDER")
for i in dataset[0]:
    print(f"{i}: {dataset[0][i]}\n")

label2id, id2label = build_label_vocab(dataset)

# Salvar em json os label2id e id2label
import json
with open("label2id.json", "w") as f:
    json.dump(label2id, f)

with open("id2label.json", "w") as f:
    json.dump(id2label, f)

tokenizer = SRLTokenizer(
    model_name="neuralmind/bert-base-portuguese-cased",
    label2id=label2id,
)

tokenized_dataset = []
for instance in dataset:
    tokenized_instance = tokenizer.tokenize_and_align(instance)
    tokenized_dataset.append(tokenized_instance)

print("PASSOU PELO TOKENIZER")
print(f"Texto tokenizado: {tokenizer.convert_ids_to_tokens(tokenized_dataset[0]['input_ids'])}\n")
for i in tokenized_dataset[0]:
    print(f"{i}: {tokenized_dataset[0][i]}\n")