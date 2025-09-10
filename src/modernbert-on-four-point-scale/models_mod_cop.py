import os
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer

bleurt_model = "vaiibhavgupta/finetuned-bleurt-large"
mpnet_model = "lear-lab/short-answer-classification"
modernbert_model = "/home/jovyan/active-projects/itell-question-generation/results/modernbert_authentic_multirc"
bleurt_threshold = 0.7


@dataclass
class ModelInput:
    """Unified input structure for all models."""

    candidate: str
    reference: str
    text: str = None
    question: str = None

    @classmethod
    def from_dict(cls, data_dict: dict):
        """Create ModelInput from dict, filtering out extra keys."""
        # Get valid field names from the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        # Filter the dictionary to only include valid fields
        filtered_dict = {k: v for k, v in data_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


class Bleurt:
    model_name = bleurt_model
    threshold = bleurt_threshold

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            # device="cuda",
        )

    def __call__(self, model_input: ModelInput) -> int:
        sequence = f"{model_input.candidate}[SEP]{model_input.reference}"
        result = self.classifier(sequence)
        score = result[0]["score"]
        return 1 if score > self.threshold else 0


class Mpnet:
    # model_name = mpnet_model
    revision = "77b846ec4606bfcfdf913888d7f0ab51f977a579"

    def __init__(self, model_path: str):
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            revision=self.revision,
            # device="cuda",
        )

    def __call__(self, model_input: ModelInput) -> int:
        sequence = f"{model_input.candidate}</s>{model_input.reference}"
        result = self.classifier(sequence)
        label = result[0]["label"]
        return 1 if label == "correct_answer" else 0


class ModernBERT:
    model_name = modernbert_model
    tokenizer_name = "answerdotai/ModernBERT-base"

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_name),
            # device="cuda",
        )

    def __call__(self, model_input: ModelInput) -> int:
        if not model_input.text or not model_input.question:
            raise ValueError("ModernBERT requires 'text' and 'question' fields")

        seq = f"{model_input.text}\n\n\n{model_input.question}\n\n\n{model_input.candidate}"
        out = self.classifier(seq)[0]
        return 1 if out["score"] > 2 else 0
