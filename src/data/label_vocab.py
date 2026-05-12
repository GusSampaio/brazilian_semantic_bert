def build_label_vocab(instances, save=False, label2id_path="label2id.json", id2label_path="id2label.json"):
    """
    Builds a label vocabulary from a list of SRL instances.
    Each instance is expected to have the following structure:
    {
        "sentence_id": str,
        "text": str,
        "predicate_id": str,
        "predicate_index": int,
        "tokens": List[str],
        "labels": List[str]
    }
    """
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

    if save:
        import json
        with open(label2id_path, "w") as f:
            json.dump(label2id, f)

        with open(id2label_path, "w") as f:
            json.dump(id2label, f)

    return label2id, id2label