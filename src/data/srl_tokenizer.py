from transformers import AutoTokenizer

class SRLTokenizer:
    """
    Responsible for transforming SRL instances into
    transformer-ready inputs.

    This class:
    - tokenizes model_tokens
    - aligns labels with subtokens
    - aligns predicate indicators
    - converts labels to ids
    """

    def __init__(
        self,
        model_name,
        label2id,
        max_length=512,
        padding="max_length",
        truncation=True,
    ):  

        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        self.label2id = label2id
        self.id2label = {
            v: k for k, v in label2id.items()
        }

        self.max_length = max_length

    def tokenize_and_align(self, instance):

        tokens = instance["model_tokens"]

        labels = instance["labels"]

        predicate_indicator = instance["predicate_indicator"]

        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
        )

        word_ids = tokenized.word_ids()

        aligned_labels = []

        aligned_predicate_indicator = []

        previous_word_id = None

        for word_id in word_ids:

            # Special tokens like [CLS], [SEP]
            if word_id is None:

                aligned_labels.append(-100)

                aligned_predicate_indicator.append(0)

            # First subtoken of a word
            elif word_id != previous_word_id:

                aligned_labels.append(
                    self.label2id[labels[word_id]]
                )

                aligned_predicate_indicator.append(
                    predicate_indicator[word_id]
                )

            # Remaining subtokens
            else:

                aligned_labels.append(-100)

                aligned_predicate_indicator.append(
                    predicate_indicator[word_id]
                )

            previous_word_id = word_id

        tokenized["labels"] = aligned_labels

        tokenized["predicate_indicator"] = (
            aligned_predicate_indicator
        )

        # pad labels and predicate indicatores if necessary
        if self.padding == "max_length":
            pad_size = len(tokenized["input_ids"]) - len(aligned_labels)
            aligned_labels.extend([-100] * pad_size)
            aligned_predicate_indicator.extend([0] * pad_size)

        return tokenized
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def convert_ids_to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)