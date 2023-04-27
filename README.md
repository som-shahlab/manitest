# ManiTest

### An LLM evaluation harness built with Manifest + EleutherAI

**ManiTest** is a simplified version of the [Eleuther-AI LLM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) that uses [Manifest](https://github.com/som-shahlab/manifest) as its backend model server.

* [Eleuther-AI LLM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) allows you to evaluate a large language model (LLM) on tasks formulated as prompts.
* [Manifest](https://github.com/som-shahlab/manifest) is a model server that enables fast inference via built-in support for HuggingFace Parallelize, Accelerate, DeepSpeed, and BitsAndBytes.

_Note: We use our own fork of Manifest, which we hope to merge back into the main repo soon._

## Installation

```bash
pip3 install manitest "git+https://github.com/som-shahlab/manifest.git@eval-michael#egg=manifest-ml[api]"
```

## Quickstart

To run the eval harness, you must first have a Manifest server running in the background with your desired model. You can then run the eval harness on your desired task. ManiTest comes with a couple tasks pre-loaded in the `manitest.tasks` module, such as `manitest.tasks.mednli` and `manitest.tasks.scitail`.

```bash
# Run Manifest server with your desired model
python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path gpt2 \
    --model_generation_type text-generation \
    --port 5001 &

# Run ManiTest evaluation harness on your desired task
# Note: To run the MedNLI task, you must first download the dataset from: https://physionet.org/content/mednli/1.0.0/
python3 -m manitest.main  \
    --manifest_url http://127.0.0.1:5001 \
    --path_to_task manitest.tasks.mednli \
    --output_dir ./ignore \
    --data_dir /Users/mwornow/Downloads/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0/ \
    --dataset_splits test

# Test a few-shot prompting setup
python3 src/manitest/main.py \
    --manifest_url http://127.0.0.1:5001 \
    --path_to_task manitest.tasks.mednli_fewshot \
    --output_dir ./ignore \
    --data_dir /Users/mwornow/Downloads/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0/ \
    --dataset_splits test \
    --n_shots 3
```

## Tips

If you're using a causal LM (e.g. GPT, OPT, Llama, Bloom)...
* Run Manifest with the `--model_generation_type text-generation` flag

If you're using a seq2seq LM (e.g. T5, T0)...
* Run Manifest with the `--model_generation_type text2text-generation` flag

## How to create your own task

We recommend starting from the `task_template.py` file as a template. You can also view `tests/mednli/mednli.py` or  `tests/scitail/scitail.py` for worked-out examples of tasks.

To create your own task, you must...

1. Create a file called `your_task.py`. You can save this anywhere.

2. Create a `Task` class that inherits from `manifest.base.Task`.

It must define three attributes (`name`, `task_type`, and `prompts`) and one methods (`load_dataset()`).

```python
from base import Task, TaskType

class YourTask(Task):
    name: str = "Your Task Name"
    task_type: TaskType = TaskType.GENERATION

    def load_dataset(self, dataloader: Optional[str], data_dir: Optional[str]) -> DatasetDict:
        # Load your dataset here
        return DatasetDict()
```

3. Create a `Prompt` class that inherits from `manifest.base.Prompt` for each individual prompt associated with your task.

It must define one attribute (`name`) and two methods (`generate_prompt()` and `get_label()`).

```python
from base import Prompt

class YourPrompt(Prompt)
    name: str = "Some globally unique name for this prompt"

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

## Development

Installation:

```bash
# Download repo
git clone https://github.com/som-shahlab/manitest
cd manitest

# Create virtual environment + install dependencies
conda create --name manitest_env python=3.10 -y
conda activate manitest_env
poetry install
```

## Special setup instructions for computers without Internet access

If you are running this on a computer without internet access (e.g. Stanford Nero), you will need to download the HuggingFace dataset, dataloader, and model that you want to use.

Assuming you've downloaded these, your commands will look like the following:

```
python3 -m manifest.api.app \
    --model_type huggingface \
    # Path to locally downloaded HuggingFace model
    --model_name_or_path /local-scratch-nvme/nigam/huggingface/pretrained/gpt2-small \
    --model_generation_type text-generation

python3 main.py \
    --manifest_url http://127.0.0.1:5000 \
    --path_to_task tests/mednli/mednli.py \
    --output_dir ./ignore \
    # Path to locally downloaded HuggingFace dataset
    --data_dir /local-scratch/nigam/projects/clinical_llm/data/mednli/ \
    --dataset_splits test \
    # Path to locally downloaded HuggingFace dataloader
    --dataloader /local-scratch/nigam/projects/clinical_llm/dataloaders/mednli/mednli.py
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
- [X] Few-shot in-context examples
- [X] Multi-class classification task support
- [X] Text generation task support
- [X] Test case with mednli replacement classification task
- [X] Abstract things to work on not Nero (e.g. load HF models from Hub, load local HF models, hit external APIs)
- [X] Clean up output to be more user friendly
- [X] Convert .yaml prompt files to .py `Task` files

## Development

To run pre-commit checks:

```bash
pre-commit run --all-files
```

To run tests:
```bash
pytest tests
```
