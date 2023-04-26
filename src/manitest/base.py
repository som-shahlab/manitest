import os
import sys
import importlib.util
from abc import abstractmethod
from enum import Enum
from datasets import DatasetDict
from typing import List, Tuple, Optional, Dict
from loguru import logger


class TaskType(Enum):
    BINARY_CLASSIFICATION = 0
    MULTICLASS_CLASSIFICATION = 0
    MULTILABEL_CLASSIFICATION = 1
    GENERATION = 2

class Prompt:
    name: str
    instruction: Optional[str]

    @abstractmethod
    def generate_query(self, example: dict) -> str:
        """Takes a dataset example and returns a version of that example formulated as a query
            without its corresponding answer, e.g.
                "Suppose X. Can we infer Y?"
        """
        return ""

    @abstractmethod
    def get_label(self, example: dict) -> str:
        """Gets the ground truth label for a dataset example"""
        return ""

    @abstractmethod
    def get_shots(self, example: dict, n_shots: int = 0) -> List[str]:
        """Gets the few-shot context for a dataset example.
            Returns a list of strings, where each string is a `shot`, and 
            each shot contains both the query and the answer, e.g.
                "Suppose X. Can we infer Y? Yes"
        """
        return []

    def generate_prompt(self, example: dict, n_shots: int = 0) -> str:
        """Take a dataset example and returns a prompted version of that example. If `n_shots > 0`
            then inject the examples in `get_shots()` as few-shot context prior to the `example` we're interested in

        Args:
            example (dict): The actual dataset example we want to prompt
            n_shots (int, optional): Number of few-shot examples to include in context. Defaults to 0.
        Returns:
            str: Prompt for the given example
        """
        prompt: str = ''
        instruction_separator: str = '\n\n'
        shot_separator: str = '\n\n'

        # Add instruction prefix to prompt
        prompt += self.instruction + instruction_separator
        
        # Add few shot context to prompt
        if n_shots > 0:
            shots: List[str] = self.get_shots(example, n_shots=n_shots)
            for shot in shots:
                prompt += shot + shot_separator
        
        # Add query of interset to prompt (i.e. what we're actually predicting)
        prompt += self.generate_query(example)
        return prompt

    def __repr__(self) -> str:
        return f"Prompt(name={self.name})"

class PromptForClassification(Prompt):
    """Prompt for classification tasks, i.e. TaskType == BINARY_CLASSIFICATION or MULTICLASS_CLASSIFICATION or MULTILABEL_CLASSIFICATION
    """
    verbalizer: Dict[str, List[str]] # [key] = class, [value] = list of strings (i.e. verbalizations) for that class

    def __repr__(self) -> str:
        return f"PromptForClassification(name={self.name})"

class PromptForGeneration(Prompt):
    """Prompt for generation tasks, i.e. TaskType == GENERATION
    """
    def __repr__(self) -> str:
        return f"PromptForGeneration(name={self.name})"

class Task:
    name: str
    task_type: TaskType
    prompts: List[Prompt]

    def __init__(self):
        self.prompts: List[Prompt] = []

    def __repr__(self) -> str:
        return f"Task(name={self.name}, task_type={self.task_type}, prompts={self.prompts})"

    @abstractmethod
    def load_dataset(self, dataloader: Optional[str], data_dir: Optional[str]) -> DatasetDict:
        """Loads the dataset for this task. Returns a DatasetDict."""
        return DatasetDict()


def load_python_module_from_python_file(path_to_python_file: str):
    path_to_python_file = os.path.abspath(path_to_python_file)
    if (
        not os.path.exists(path_to_python_file)
        or not os.path.isfile(path_to_python_file)
        or not path_to_python_file.endswith(".py")
    ):
        raise ValueError(f"Python file @ path '{path_to_python_file}' does not exist")
    spec = importlib.util.spec_from_file_location("module.name", path_to_python_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module


def load_task(path_to_task: str, dataloader: Optional[str], data_dir: Optional[str]) -> Tuple[DatasetDict, Task]:
    # Load config.py file
    module = load_python_module_from_python_file(path_to_task)
    task: Task = module.Export()
    logger.info(f"Loaded task '{task.name}' with task type '{task.task_type}'")

    # Validate task
    unique_prompt_names: set = set([p.name for p in task.prompts])
    if len(unique_prompt_names) != len(task.prompts):
        raise ValueError(
            f"Duplicate prompt names found in task '{task.name}'."
            " All prompt's must have a unique `name` attribute set."
        )

    # Load dataset
    dataset = task.load_dataset(dataloader=dataloader, data_dir=data_dir)
    logger.info(
        f"Loaded dataset{' ' + str(dataloader) if dataloader else ''}"
        f" from '{data_dir if data_dir else 'HuggingFace Hub'}'"
    )
    logger.info(dataset)

    return dataset, task
