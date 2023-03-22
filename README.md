# Manifest-Based Evaluation Harness for LLMs

A simplified version of the [Eleuther-AI LLM evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness) built on top of [Manifest](https://github.com/som-shahlab/manifest).

[Manifest](https://github.com/som-shahlab/manifest) is a model server that enables faster inference via built-in support for HuggingFace Parallelize, Accelerate, DeepSpeed, and BitsAndBytes.

_Note: We use our own fork of Manifest, which we hope to merge back into the main repo soon._

## Installation

```bash
git clone https://github.com/som-shahlab/llm_eval_harness
cd llm_eval_harness
pip3 install -r requirements.txt
```

## Quickstart

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

## Todos

- [ ] Support passing text generation flags to `main.py`, pass these along to Manifest
- [ ] Documentation
- [ ] Multi-label classification task support
- [X] Multi-class classification task support
- [X] Text generation task support
- [X] Test case with mednli replacement classification task
- [X] Abstract things to work on not Nero (e.g. load HF models from Hub, load local HF models, hit external APIs)
- [X] Clean up output to be more user friendly
- [X] Convert .yaml prompt files to .py `Task` files
