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


class Task:
    name: str
    task_type: TaskType

    def __init__(self):
        self.prompts: List[Prompt] = []

    def __repr__(self) -> str:
        return f"Task(name={self.name}, task_type={self.task_type}, prompts={self.prompts})"

class Prompt:
    name: str

    @abstractmethod
    def generate_prompt(self, example: dict) -> str:
        """Takes a dataset example and returns a prompted version of that example."""
        return ""

    @abstractmethod
    def get_label(self, example: dict) -> str:
        """Gets the ground truth label for a dataset example"""
        return ""

    def __repr__(self) -> str:
        return f"Prompt(name={self.name})"

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
    unique_prompt_names: set = set([ p.name for p in task.prompts])
    if len(unique_prompt_names) != len(task.prompts):
        raise ValueError(f"Duplicate prompt names found in task '{task.name}'. All prompt's must have a unique `name` attribute set.")

    # Load dataset
    dataset = task.load_dataset(dataloader=dataloader, data_dir=data_dir)
    logger.info(
        f"Loaded dataset {str(dataloader) + ' ' if dataloader else ''}from '{data_dir if data_dir else 'HuggingFace Hub'}'"
    )
    logger.info(dataset)

    return dataset, task
