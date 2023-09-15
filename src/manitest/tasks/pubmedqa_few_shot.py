import os, sys
import random
from datasets import load_dataset, DatasetDict
from typing import List, Optional
from manitest.base import Prompt, PromptForClassification, TaskType, Task


class Prompt1(PromptForClassification):
    name: str = "mcq"
    instruction: str = "Based on the context, answer the question."
    verbalizer: dict = {"yes": ["yes"], "no": ["no"], "maybe": ["maybe"]}

    def get_label(self, example: dict):
        """Maps attributes of a dataset example to a class (i.e. a [key] in `verbalizer`)"""
        if example["label"] in ["yes"]:
            return "yes"
        elif example["label"] in ["no"]:
            return "no"
        elif example["label"] in ["maybe"]:
            return "maybe"
        else:
            raise ValueError(f"Unknown label {example['label']}")

    def generate_query(self, example: dict) -> str:
        """Takes a dataset example and returns a version of that example formulated as a query
        without its corresponding answer, e.g.
            "Suppose X. Can we infer Y?"
        """
        return f"\n\nContext: {example['sentence2']}\nQuestion: {example['sentence1']} yes, no or maybe?\n\nAnswer: "

    def get_shots(self, example: dict, n_shots: int = 0, **kwargs) -> List[str]:
        # Randomly select `n_shots` examples from the `in_context_shot_dataset` dataset
        dataset = kwargs.get("in_context_shot_dataset")
        shots = dataset.select(random.choices(range(len(dataset)), k=n_shots))
        return [self.generate_query(shot) + " " + self.get_label(shot) for shot in shots]


####################################
# Task definition
####################################


class PubMedQAEval(Task):
    name: str = "pubmedqa"
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION

    def __init__(self):
        self.prompts: List[Prompt] = [
            Prompt1(),
        ]

    def load_dataset(self, dataloader: Optional[str] = None, data_dir: Optional[str] = None):
        data_file = {
            "train": os.path.join(data_dir, "train.json"),
            "test": os.path.join(data_dir, "test.json"),
            "dev": os.path.join(data_dir, "dev.json"),
        }
        return load_dataset("json", data_files=data_file)


# Important -- needed so that the task can be imported by the eval harness!
Export = PubMedQAEval
