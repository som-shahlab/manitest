# Evaluation Harness for LLMs (Manifest + EleutherAI Harness)

A simplified version of the [Eleuther-AI LLM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) built on top of [Manifest](https://github.com/som-shahlab/manifest).

* [Eleuther-AI LLM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) allows you to evaluate a large language model (LLM) on tasks formulated as prompts.
* [Manifest](https://github.com/som-shahlab/manifest) is a model server that enables faster inference via built-in support for HuggingFace Parallelize, Accelerate, DeepSpeed, and BitsAndBytes.

_Note: We use our own fork of Manifest, which we hope to merge back into the main repo soon._

## Installation

```bash
git clone https://github.com/som-shahlab/llm_eval_harness
cd llm_eval_harness
pip3 install -r requirements.txt
```

## Quickstart

To run the eval harness, you must first have a Manifest server running in the background with your desired model.

```bash
# Run Manifest server with your desired model
python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path gpt2 \
    --model_generation_type text-generation &

# Run evaluation harness on your desired task
python3 main.py \
    --manifest_url http://127.0.0.1:5000 \
    --path_to_task tests/mednli/mednli.py \
    --output_dir ./ignore \
    --data_dir /Users/mwornow/Downloads/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0/ \
    --dataset_splits test,train
```

## Create your own task

We recommend starting from the `task_template.py` file as a template. You can also view `tests/mednli/mednli.py` or  `tests/scitail/scitail.py` for worked-out examples of tasks.

To create your own task, you must...

1. Create a file called `your_task.py`. You can save this anywhere.

2. Create a `Task` class that inherits from `manifest.base.Task`. It must define three attributes and one method. Attributes: `name`, `task_type`, and `prompts`. Methods: `load_dataset()`.

```python
from base import Task, TaskType
class YourTask(Task):
    name = "Your Task Name"
    task_type = TaskType.GENERATION

    def load_dataset(self, dataloader: Optional[str], data_dir: Optional[str]) -> DatasetDict:
        # Load your dataset here
        return DatasetDict()
```

3. Create a `Prompt` class that inherits from `manifest.base.Prompt` for each individual prompt associated with your task. It must define one attribute and two methods. Attributes: `name`. Methods: `generate_prompt()` and `get_label()`.

```python
from base import Prompt
class YourPrompt(Prompt)
    name = "Some globally unique name for this prompt"
    def generate_prompt(self, example: dict) -> str:
        """Takes a dataset example and returns a prompted version of that example."""
        return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}. Does the premise entail the hypothesis?"

    def get_label(self, example: dict) -> str:
        """Gets the ground truth label for a dataset example"""
        return example['true_label']
```

4. Run the evaluation harness with your task. This assumes a Manifest server is already running on `localhost:5000`

```bash
python3 main.py \
    --manifest_url http://localhost:5000 \
    --path_to_task path/to/your_task.py \
    --output_dir ./ignore
```
## Todos

- [ ] Combine Manifest command with `main.py` command into a single command
- [ ] Merge Manifest fork back into main repo
- [ ] Support specifying multiple tasks in a single run
- [ ] Pretty print / format results
- [ ] Support passing text generation flags to `main.py`, pass these along to Manifest
- [ ] Add tests
- [ ] Documentation
- [ ] Multi-label classification task support
- [X] Multi-class classification task support
- [X] Text generation task support
- [X] Test case with mednli replacement classification task
- [X] Abstract things to work on not Nero (e.g. load HF models from Hub, load local HF models, hit external APIs)
- [X] Clean up output to be more user friendly
- [X] Convert .yaml prompt files to .py `Task` files
