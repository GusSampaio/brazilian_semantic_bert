import torch
from data.data_module import SRLDataModule
from models.srl_model import SRLModel

module = SRLDataModule(save_data=True, save_data_path="data/processed/data_splits")
# module = SRLDataModule(use_preprocessed_data=True, preprocess_data_path="preprocessed_data")

train_set = module.get_split("train")
print(f"Texto tokenizado: {module.tokenizer.convert_ids_to_tokens(train_set[0]['input_ids'])}\n")
# for example in train_set:
#     print(example)
#     break

train_loader = module.get_dataloader(train_set, batch_size=32)

model = SRLModel(
    model_name="neuralmind/bert-base-portuguese-cased",
    tokenizer=module.tokenizer,
    num_labels=len(module.label2id),
    id2label=module.id2label,
    label2id=module.label2id,
)

batch = next(iter(train_loader))
print(f"Batch input_ids shape: {batch['input_ids'].shape}")

outputs = model.model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)

print(f"Model outputs shape: {outputs.logits.shape}")
print(f"Model loss: {outputs.loss}")

