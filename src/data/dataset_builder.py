from copy import deepcopy

class SRLDatasetBuilder:
    """
    Responsible for transforming "base" SRL instances
    into different input formats for training.

    Each "base" instance is expected to have the following structure:

    {
        "sentence_id": str,
        "text": str,
        "predicate_id": str,
        "predicate_index": int,
        "tokens": List[str],
        "labels": List[str]
    }
    """

    def __init__(
        self,
        predicate_signal="special_token",
        special_start_token="<PRED>",
        special_end_token="</PRED>",
    ):
        """
        predicate_signal:
            - "binary_feature"
            - "special_token"
        """

        valid = {"binary_feature", "special_token"}

        if predicate_signal not in valid:
            raise ValueError(
                f"predicate_signal must be one of {valid}"
            )

        self.predicate_signal = predicate_signal
        self.special_start_token = special_start_token
        self.special_end_token = special_end_token

    def build(self, instances):
        """
        Receives a list of SRL instances and returns
        new processed instances.
        """

        processed = []

        for instance in instances:
            processed.append(
                self._build_instance(instance)
            )

        return processed

    def _build_instance(self, instance):

        inst = deepcopy(instance)

        tokens = inst["tokens"]
        pred_idx = inst["predicate_index"]

        # Just create a binary feature for each token indicating whether it's the predicate or not
        if self.predicate_signal == "binary_feature":

            inst["model_tokens"] = tokens

            inst["predicate_indicator"] = [
                1 if i == pred_idx else 0
                for i in range(len(tokens))
            ]

        # Add special tokens around the predicate and create  new a binary feature indicating the position of the predicate
        elif self.predicate_signal == "special_token":

            new_tokens = []

            for i, tok in enumerate(tokens):

                if i == pred_idx:
                    new_tokens.append(self.special_start_token)
                    new_tokens.append(tok)
                    new_tokens.append(self.special_end_token)

                else:
                    new_tokens.append(tok)

            inst["model_tokens"] = new_tokens

            inst["predicate_indicator"] = [
                1 if i == pred_idx else 0
                for i in range(len(tokens))
            ]
        

        return inst