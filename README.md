# Manifest-Based Evaluation Harness for LLMs

A light-weight, simplified version of the Eleuther-AI LLM evaluation harness built ontop of Manifest, a backend that enables faster inference via built-in support for HuggingFace Parallelize, HuggingFace Accelerate, DeepSpeed, and BitsAndBytes.


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
	--model_name_or_path /local-scratch-nvme/nigam/huggingface/pretrained/gpt-j-6B \
	--model_generation_type text-generation \
    --port 5009 \
	--use_hf_parallelize

# Run evaluation harness on your desired datasets
python3 main.py \
    --manifest_url http://127.0.0.1:5009 \
    --path_to_dataset_config prompts/mednli.yaml \
    --path_to_dataset_dir /local-scratch/nigam/projects/clinical_llm/data/mednli \
    --path_to_output_dir ./ignore/mednli_gptj_update/
```

## Todos

- [ ] Test case with medparasimp generation task
- [ ] Test case with mednli replacement classification task
- [ ] Abstract things to work on not Nero (e.g. load HF models from Hub, load local HF models, hit external APIs)
- [ ] Support text generation flags in `main.py`, pass to Manifest
- [ ] Clean up output to be more user friendly
- [ ] Convert .yaml prompt files to .py `Task` files
- [ ] Documentation
