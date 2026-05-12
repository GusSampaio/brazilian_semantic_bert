from copy import deepcopy

class SRLInstanceBuilder:
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
        """
        Processes a single SRL instance.
        If predicate_signal is "binary_feature", adds a binary feature for each token indicating whether it's the predicate or not.
        If predicate_signal is "special_token", adds special tokens around the predicate and creates a new binary feature indicating the position of the predicate.
        """

        inst = deepcopy(instance)

        tokens = inst["tokens"]
        pred_idx = inst["predicate_index"]

        # Create a binary feature for each token indicating whether it's the predicate or not
        if self.predicate_signal == "binary_feature":

            inst["model_tokens"] = tokens

            inst["predicate_indicator"] = [
                1 if i == pred_idx else 0
                for i in range(len(tokens))
            ]

        # Add special tokens around the predicate and create  new a binary feature indicating the position of the predicate
        elif self.predicate_signal == "special_token":

            new_tokens = []
            new_labels = []

            for i, tok in enumerate(tokens):

                if i == pred_idx:
                    new_tokens.append(self.special_start_token)
                    new_labels.append(-100)

                    new_tokens.append(tok)
                    new_labels.append(inst["labels"][i])
                    
                    new_tokens.append(self.special_end_token)
                    new_labels.append(-100)
                else:
                    new_tokens.append(tok)
                    new_labels.append(inst["labels"][i])

            inst["model_tokens"] = new_tokens
            inst["model_labels"] = new_labels

            inst["predicate_indicator"] = [
                1 if i == pred_idx+1 else 0
                for i in range(len(new_tokens))
            ]
        
        return inst