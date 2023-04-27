# flake8: noqa
from .base import TaskType, Task, Prompt, PromptForClassification, PromptForGeneration, load_task
from .eval import run_eval, run_classification, run_generation, run_multilabel_classification
from .metrics import generation_metric

from importlib.metadata import version

__version__ = version("manitest")
