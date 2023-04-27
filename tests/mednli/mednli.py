from datasets import load_dataset
from typing import List, Optional
from manitest import PromptForClassification, TaskType, Task

####################################
# Prompt definitions
####################################


class MedNLIPrompt(PromptForClassification):
    verbalizer: dict = {
        "entailment": ["yes"],
        "not entailment": ["no"],
    }

    def get_label(self, example: dict):
        """Maps attributes of a dataset example to a class (i.e. a [key] in `verbalizer`)"""
        if example["label"] in ["entailment"]:
            return "entailment"
        elif example["label"] in ["neutral", "contradiction"]:
            return "not entailment"
        else:
            raise ValueError(f"Unknown label {example['label']}")


class Prompt1(MedNLIPrompt):
    name: str = "suppose"
    verbalizer: dict = {
        "entailment": ["entailment"],
        "not entailment": ["neutral"],
    }

    def generate_query(self, example: dict) -> str:
        return f"Suppose {example['premise']} Can we infer that {example['hypothesis']}?"


class Prompt2(MedNLIPrompt):
    name: str = "two_sentences"

    def generate_query(self, example: dict) -> str:
        return (
            f"Sentence 1: {example['premise']}\n\nSentence 2: {example['hypothesis']}\n\n"
            "Question: Does Sentence 1 entail Sentence 2?  yes or no"
        )


class Prompt3(MedNLIPrompt):
    name: str = "does_it_follow"

    def generate_query(self, example: dict) -> str:
        return f"Given that {example['premise']} Does it follow that {example['hypothesis']}  yes or no"


class Prompt4(MedNLIPrompt):
    name: str = "licensed_to_say"
    verbalizer: dict = {
        "entailment": ["true"],
        "not entailment": ["false"],
    }

    def generate_query(self, example: dict) -> str:
        return f"{example['premise']} Therefore, we are licensed to say that {example['hypothesis']}  true or false"


class Prompt5(MedNLIPrompt):
    name: str = "does_passage_support_claim"

    def generate_query(self, example: dict) -> str:
        return f"{example['premise']} Does the previous passage support the claim that {example['hypothesis']}?"


####################################
# Task definition
####################################


class MedNLI(Task):
    name: str = "mednli"
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION

    def __init__(self):
        self.prompts: List[PromptForClassification] = [
            Prompt1(),
            Prompt2(),
            Prompt3(),
            Prompt4(),
            Prompt5(),
        ]

    def load_dataset(self, dataloader: Optional[str] = None, data_dir: Optional[str] = None):
        return load_dataset(
            "bigbio/mednli" if dataloader is None else dataloader, data_dir=data_dir, name="mednli_bigbio_te"
        )


# Important -- needed so that the task can be imported by the eval harness!
Export = MedNLI
