import os, sys
import random
from datasets import load_dataset, DatasetDict
from typing import List, Optional
from manitest.base import Prompt, PromptForGeneration, TaskType, Task


class Prompt1(PromptForGeneration):
    name: str = "medparasimp"
    instruction: str = "Given a paragraph, summarize it."

    def get_label(self, example: dict):
        return example["text_2"]

    def get_shots(self, example: dict, n_shots: int = 0, **kwargs) -> List[str]:
        # Randomly select `n_shots` examples from the `in_context_shot_dataset` dataset
        dataset = kwargs.get("in_context_shot_dataset")
        shots = dataset.select(random.choices(range(len(dataset)), k=n_shots))
        return [self.generate_query(shot) + " " + self.get_label(shot) for shot in shots]

    def generate_query(self, example: dict) -> str:
        """Takes a dataset example and returns a version of that example formulated as a query
            without its corresponding answer, e.g.
                "Suppose X. Can we infer Y?"
        """
        return f"Paragraph: {example['text_1']} \n\nSummary: "

####################################
# Task definition
####################################

class MedParaSimp(Task):
    name: str = "medparasimp"
    task_type: TaskType = TaskType.GENERATION

    def __init__(self):
        self.prompts: List[Prompt] = [
            Prompt1(),
        ]

    def load_dataset(self, dataloader: Optional[str] = None, data_dir: Optional[str] = None):
        return load_dataset(dataloader, name="medparasimp_bigbio_t2t")


# Important -- needed so that the task can be imported by the eval harness!
Export = MedParaSimp



