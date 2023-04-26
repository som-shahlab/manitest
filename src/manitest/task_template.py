from datasets import load_dataset
from typing import List, Optional
from manitest import Prompt, TaskType, Task

####################################
# Prompt definitions
####################################


class YourPrompt1(Prompt):
    # Unique name for this prompt
    name: str = "<unique name for this prompt 1>"
    # Since this is a classification task, we need to define a verbalizer that maps
    # model outputs to class labels
    verbalizer: dict = {
        "very positive": ["super", "fantastic"],
        "positive": ["good", "not bad"],
        "negative": [
            "meh",
            "not good",
        ],
        "very negative": ["terrible"],
    }

    # This method takes a dataset example and returns a prompted version of that example
    def generate_prompt(self, example: dict) -> str:
        return (
            "On a scale of very positive to very negative,"
            f" what is the sentiment of the sentence {example['sentence']}?"
        )


class YourPrompt2(Prompt):
    # Unique name for this prompt
    name: str = "<unique name for this prompt 2>"
    # Since this is a classification task, we need to define a verbalizer that maps
    # model outputs to class labels
    verbalizer: dict = {
        "very positive": ["super", "fantastic"],
        "positive": ["good", "not bad"],
        "negative": [
            "meh",
            "not good",
        ],
        "very negative": ["terrible"],
    }

    # This method takes a dataset example and returns a prompted version of that example
    def generate_prompt(self, example: dict) -> str:
        return f"Given the sentence {example['sentence']}, does it have a positive or negative tone?"


####################################
# Task definition
####################################


class YourTask(Task):
    # name for your task
    name: str = "<name for task>"
    # TaskType is an enum with possible values:
    # BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION, GENERATION
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION

    # Initialize your task with a list of prompts
    def __init__(self):
        self.prompts: List[Prompt] = [
            YourPrompt1(),
            YourPrompt2(),
        ]

    # Load your dataset using HuggingFace's datasets library
    def load_dataset(self, dataloader: Optional[str] = None, data_dir: Optional[str] = None):
        return load_dataset(dataloader, data_dir=data_dir)


# Important -- you must set `Export` equal to your task so that it can be imported by the eval harness!
Export = YourTask
