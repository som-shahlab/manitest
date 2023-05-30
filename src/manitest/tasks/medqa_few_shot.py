import os, sys
import random
from datasets import load_dataset, DatasetDict
from typing import List, Optional
from manitest.base import Prompt, PromptForClassification, TaskType, Task


class Prompt1(PromptForClassification):
    name: str = "mcq"
    verbalizer: dict = {
        "A": ["(A)"],
        "B": ["(B)"],
        "C": ["(C)"],
        "D": ["(D)"],
    }
    instruction: str = "The following are multiple choice questions (with answers) about medical knowledge. Choose (A) or (B) or (C) or (D)."

    def get_shots(self, example: dict, n_shots: int = 0, **kwargs) -> List[str]:
        # Randomly select `n_shots` examples from the `in_context_shot_dataset` dataset
        dataset = kwargs.get("in_context_shot_dataset")
        shots = dataset.select(random.choices(range(len(dataset)), k=n_shots))
        return [self.generate_query(shot) + " " + self.get_label(shot) for shot in shots]

    def get_label(self, example: dict):
        """Maps attributes of a dataset example to a class (i.e. a [key] in `verbalizer`)"""
        if example['answer'] in ["(A)"]:
            return "A"
        elif example['answer'] in ["(B)"]:
            return "B"
        elif example['answer'] in ["(C)"]:
            return "C"
        elif example['answer'] in ["(D)"]:
            return "D"
        else:
            raise ValueError(f"Unknown label {example['answer']}")

    def generate_query(self, example: dict) -> str:
        """Takes a dataset example and returns a version of that example formulated as a query
            without its corresponding answer, e.g.
                "Suppose X. Can we infer Y?"
        """
        return f"Question: {example['question']}\nThe choices are: (A) {example['choices'][0]} (B) {example['choices'][1]} (C) {example['choices'][2]} (D) {example['choices'][3]}\nAnswer: "


class MedQAEval(Task):
    name: str = "medqa"
    task_type: TaskType = TaskType.MULTICLASS_CLASSIFICATION

    def __init__(self):
        self.prompts: List[Prompt] = [
            Prompt1(),
        ]

    def load_dataset(self, dataloader: Optional[str] = None, data_dir: Optional[str] = None):
        return load_dataset(
            "bigbio/medqa" if dataloader is None else dataloader, data_dir=data_dir, name="med_qa_en_bigbio_qa"
        )


# Important -- needed so that the task can be imported by the eval harness!
Export = MedQAEval