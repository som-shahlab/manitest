import os
import sys
import importlib.util
from enum import Enum
from datasets import DatasetDict
from typing import List, Tuple, Optional, Dict
from loguru import logger
from abc import ABC, abstractmethod


class TaskType(Enum):
    BINARY_CLASSIFICATION = 0
    MULTICLASS_CLASSIFICATION = 0
    MULTILABEL_CLASSIFICATION = 1
    GENERATION = 2


class Prompt(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique ID for this prompt"""
        pass

    @property
    def instruction(self) -> Optional[str]:
        """Instruction prepended to start of prompt (if not `None`)."""
        return None

    @abstractmethod
    def generate_query(self, example: dict) -> str:
        """Takes a dataset example and returns a version of that example formulated as a query
        without its corresponding answer, e.g.
            "Suppose X. Can we infer Y?"
        """
        return ""

    @abstractmethod
    def get_label(self, example: dict) -> str:
        """Get the ground truth label for this dataset example.
        Given a query ("Suppose X. Can we infer Y?") it returns the
        class corresponding to this example ("entailment")
        """
        return ""

    def get_shots(self, example: dict, n_shots: int = 0, **kwargs) -> List[str]:
        """Get the few-shot context for a dataset example.
            Returns a list of strings, where each string is a `shot`, and
            each shot contains both the query and the answer, e.g.
                "Suppose X. Can we infer Y? Yes"

        Args:
            example (dict): The actual dataset example we are querying the model to answer
            n_shots (int, optional): Number of examples to include in context. Defaults to 0.
            kwargs (dict): Anything else you want to pass to this function

        Returns:
            List[str]: List of examples (i.e. 'shots') that will be injected into the prompt.
        """
        return []

    def generate_prompt(
        self,
        example: dict,
        n_shots: int = 0,
        instruction_separator: str = "\n\n",
        shot_separator: str = "\n",
        post_shot_separator: str = "\n",
        **kwargs,
    ) -> str:
        """Take a dataset example and returns a prompted version of that example.
            If `n_shots > 0` then inject the examples in `get_shots()` as few-shot
            context prior to the `example` we're interested in.

        Args:
            example (dict): The actual dataset example we want to prompt
            n_shots (int): Number of few-shot examples to include in context. Defaults to 0.
            instruction_separator (str): Text inserted after the content of `self.instruction`
                is preprended to the prompt (if `self.instruction is not None`). Defaults to \n\n.
            shot_separator (str): Text inserted after each shot from `self.get_shots()` is added
                to the prompt (if `self.n_shots > 0`). Defaults to \n.
            post_shot_separator (str): Text inserted after the last shot from `self.get_shots()` is added
                to the prompt (if `self.n_shots > 0`). Defaults to \n.
            **kwargs (dict): Passed to `get_shots()`

        Returns:
            str: Prompt for the given example
        """
        prompt: str = ""

        # Add instruction prefix to prompt
        if hasattr(self, "instruction") and self.instruction is not None:
            prompt += self.instruction + instruction_separator

        # Add few shot context to prompt
        if n_shots > 0:
            shots: List[str] = self.get_shots(example, n_shots=n_shots, **kwargs)
            for shot in shots:
                prompt += shot + shot_separator
            prompt += post_shot_separator

        # Add query of interset to prompt (i.e. what we're actually predicting)
        prompt += self.generate_query(example)
        return prompt

    def __repr__(self) -> str:
        return f"Prompt(name={self.name})"


class PromptForClassification(Prompt):
    """Prompt for classification tasks, i.e.
    TaskType in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION]
    """

    @property
    @abstractmethod
    def verbalizer(self) -> Dict[str, List[str]]:
        """Return dict where [key] = class, [value] = list of strings (i.e. verbalizations) that,
        if output by the LLM, are mapped to that class.

        Example:
        ```
            return {
                'entailment' : ['yes', 'true',],
                'not entailment' : ['no', 'false',],
            }
        ```
        """
        return {}

    def __repr__(self) -> str:
        return f"PromptForClassification(name={self.name}, verbalizer={self.verbalizer})"


class PromptForGeneration(Prompt):
    """Prompt for generation tasks, i.e. TaskType == GENERATION"""

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
    """Load a python module from a python file
        Loads from path if path is passed.
        Loads from`tasks/` directory provided with ManiTest if path is not passed.

    Args:
        path_to_python_file (str): Either a path or a `manitest.task.task_name`

    Returns:
        _type_: Python module
    """
    # If `manitest.tasks.` is passed, load from `manitest/tasks/` directory
    if not os.path.exists(path_to_python_file) and path_to_python_file.startswith("manitest.tasks."):
        logger.info(f"Loading module {path_to_python_file} from `manitest/tasks/` directory")
        module = importlib.import_module(path_to_python_file)
    else:
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
