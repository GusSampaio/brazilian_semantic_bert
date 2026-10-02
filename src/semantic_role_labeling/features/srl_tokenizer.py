from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

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
        using_special_tokens=True,
        max_length=512, # why truncate here?
        truncation=True,
    ):  
        self.truncation = truncation
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        
        self._fix_byte_level_prefix_space()

        if using_special_tokens:
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": [
                    "<PRED>",
                    "</PRED>"
                ]
            })

        self.label2id = label2id
        self.id2label = {
            v: k for k, v in label2id.items()
        }

        self.max_length = max_length

    def _fix_byte_level_prefix_space(self):
        """
        Fixes the prefix space issue for ByteLevel tokenizers.
        This is necessary because the tokenizer may not add a prefix space
        to the first token of a sentence, which can lead to incorrect tokenization.
        """
        if isinstance(self.tokenizer.backend_tokenizer.pre_tokenizer, ByteLevelPreTokenizer):
            self.tokenizer.backend_tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True)

        if isinstance(self.tokenizer.backend_tokenizer.decoder, ByteLevelDecoder):
            self.tokenizer.backend_tokenizer.decoder = ByteLevelDecoder()

    def tokenize_and_align(self, instance):
        tokens = instance["model_tokens"]
        labels = instance["model_labels"]
        predicate_indicator = instance["predicate_indicator"]

        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=self.truncation,
            max_length=self.max_length,
            spaces_between_special_tokens=True,
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
                current_label = labels[word_id]

                if tokens[word_id] in ["<PRED>", "</PRED>"] or current_label == -100:
                    aligned_labels.append(-100)
                else:
                    aligned_labels.append(
                        self.label2id.get(current_label, -100)
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

        return tokenized
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def convert_ids_to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)