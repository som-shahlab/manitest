import os, sys
import random
from datasets import load_dataset, DatasetDict
from typing import List, Optional
from manitest.base import Prompt, PromptForClassification, TaskType, Task

 
####################################
# Prompt definitions
####################################


class Prompt1(PromptForClassification):
    name: str = "suppose"
    verbalizer: dict = {
        "entailment": ["entailment"],
        "not entailment": ["neutral"],
    }
    instruction: str = "Decide whether the following sentences are entailments or not."

    def get_shots(self, example: dict, n_shots: int = 0, **kwargs) -> List[str]:
        # Randomly select `n_shots` examples from the `in_context_shot_dataset` dataset
        dataset = kwargs.get("in_context_shot_dataset")
        shots = dataset.select(random.choices(range(len(dataset)), k=n_shots))
        return [self.generate_query(shot) + " " + self.get_label(shot) for shot in shots]

    def get_label(self, example: dict):
        """Maps attributes of a dataset example to a class (i.e. a [key] in `verbalizer`)"""
        if example["label"] in ["entailment"]:
            return "entailment"
        elif example["label"] in ["neutral", "contradiction"]:
            return "not entailment"
        else:
            raise ValueError(f"Unknown label {example['label']}")

    def generate_query(self, example: dict) -> str:
        return f"Suppose {example['premise']} Can we infer that {example['hypothesis']}?"


####################################
# Task definition
####################################

class MedNLIEval(Task):
    name: str = "mednli"
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION

    def __init__(self):
        shots: int = 2
        self.prompts: List[Prompt] = [
            # Prompt0(), 
            Prompt1(), 
            # Prompt2(), 
            # Prompt3(), 
            # Prompt4(), 
            # Prompt5(),
        ]

    def load_dataset(self, dataloader: Optional[str] = None, data_dir: Optional[str] = None):
        return load_dataset(
            "bigbio/mednli" if dataloader is None else dataloader, data_dir=data_dir, name="mednli_bigbio_te"
        )


# Important -- needed so that the task can be imported by the eval harness!
Export = MedNLIEval