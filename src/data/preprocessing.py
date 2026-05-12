from parser import PBP_parser
from instance_builder import SRLInstanceBuilder
from srl_tokenizer import SRLTokenizer
from splitter import SRLSplitter
from label_vocab import build_label_vocab

instances = PBP_parser("../../PBP-classic-complete.conllu")

builder = SRLInstanceBuilder(
    predicate_signal="special_token",
    # predicate_signal="binary_feature",
)

dataset = builder.build(instances)

for i in dataset[0]:
    print(f"{i}: {dataset[0][i]}\n")

label2id, id2label = build_label_vocab(dataset, save=True)

tokenizer = SRLTokenizer(
    model_name="neuralmind/bert-base-portuguese-cased",
    label2id=label2id,
)

tokenized_dataset = []
for instance in dataset:
    tokenized_instance = tokenizer.tokenize_and_align(instance)
    tokenized_dataset.append(tokenized_instance)

print(f"Texto tokenizado: {tokenizer.convert_ids_to_tokens(tokenized_dataset[0]['input_ids'])}\n")
for i in tokenized_dataset[0]:
    print(f"{i}: {tokenized_dataset[0][i]}\n")

# splitter = SRLSplitter(
#     train_ratio=0.8,
#     dev_ratio=0.1,
#     test_ratio=0.1,
#     seed=42,
# )

# train_set, dev_set, test_set = splitter.split(dataset)
# print(f"Número de instâncias no conjunto de treino: {len(train_set)}")
# print(f"Número de instâncias no conjunto de desenvolvimento: {len(dev_set)}")
# print(f"Número de instâncias no conjunto de teste: {len(test_set)}")