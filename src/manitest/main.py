"""
Script for querying language model.

Example Usage:

python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path gpt2 \
    --model_generation_type text-generation

python3 main.py \
    --manifest_url http://127.0.0.1:5000 \
    --path_to_task tests/mednli/mednli.py \
    --output_dir ./ignore \
    --data_dir ~/Downloads/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0/ \
    --dataset_splits test,train
"""
import os
import argparse
import requests
from loguru import logger
from manifest import Manifest
from datasets import DatasetDict, Dataset
from urllib.parse import urlparse

from manitest.eval import run_eval
from manitest.base import load_task


def main(args):
    # Load dataset + prompts for specific task
    dataset, task = load_task(args.path_to_task, args.dataloader, args.data_dir)
    logger.info(f"Finished loading '{task.name}' dataset and task")

    # Setup directory where we will save our outputs / logs
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Will be saving outputs to: '{args.output_dir}'")

    # Manifest
    manifest_hostname = urlparse(args.manifest_url).hostname
    os.environ["no_proxy"] = f"localhost, 127.0.0.1, {manifest_hostname}"  # Needed on Nero
    manifest = Manifest(
        client_name="huggingface",
        client_connection=args.manifest_url,
    )

    # Test Manifest connection
    try:
        requests.get(args.manifest_url)
    except Exception as e:
        print(str(e))
        raise ConnectionRefusedError(f"Error connecting to Manifest server. Is it running at {args.manifest_url} ?")

    # Get dataset for in-context shots
    in_context_shot_dataset: Dataset = dataset["train"]

    # Get dataset splits that we evaluate model on
    try:
        splits = args.dataset_splits.split(",")
    except Exception as e:
        print(str(e))
        raise ValueError(
            f"Error parsing `--dataset_splits`. It should be a comma-separated list, but got: {args.dataset_splits}"
        )
    dataset: DatasetDict = DatasetDict({split: dataset[split] for split in splits})
    logger.info(f"Evaluating on dataset splits: {splits}")

    # Run evaluations
    run_eval(
        manifest,
        dataset,
        task,
        args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        n_shots=args.n_shots,
        in_context_shot_dataset=in_context_shot_dataset,
    )

    logger.info("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument(
        "--manifest_url",
        type=str,
        help=(
            "Full URL where Manifest server is running, e.g. 'http://localhost:5000'."
            " Make sure to include 'http' or 'https'."
        ),
        required=True,
    )
    parser.add_argument(
        "--path_to_task",
        type=str,
        help="Path to configuration .py file for this task",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to DIRECTORY to output logs / results / metrics",
        required=True,
    )

    # Dataset loading/splits
    parser.add_argument(
        "--dataloader",
        type=str,
        help=(
            "Path to dataloader .py file (if applicable)."
            "Passed to huggingface's load_dataset() as `dataloader` kwarg."
        ),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help=(
            "Path to DIRECTORY containing your dataset (if applicable)."
            "Passed to huggingface's load_dataset() as `data_dir` kwarg."
        ),
        required=False,
    )
    parser.add_argument(
        "--dataset_splits",
        type=str,
        help="Comma-separated list of splits to evaluate, e.g. 'train,test,val'. Default is 'test'",
        required=False,
        default="test",
    )

    # Optional
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of examples per batch",
        default=10,
    )
    parser.add_argument(
        "--is_show_log",
        action=argparse.BooleanOptionalAction,
        help="If specified, print logs to console",
        default=False,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Max new generation token length",
        default=512,
    )

    # Decoder args (optional)
    parser.add_argument(
        "--do_sample",
        type=bool,
        help="Sampling while generating from model",
        default=True,
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        help="Stop early while generating",
        default=False,
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        help="Number of sequences to generate",
        default=1,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        help="Number of beams to search",
        default=5,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="For nucleus sampling",
        default=0.9,
    )

    # In-context few shot (optional)
    parser.add_argument(
        "--n_shots",
        type=int,
        help="Number of examples to insert into prompt before your query as additional in-context examples",
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (for sampling shots)",
        default=0,
    )

    args = parser.parse_args()
    if not (args.manifest_url.startswith("http://") or args.manifest_url.startswith("https://")):
        raise ValueError(
            "Please include 'http://' or 'https://' in your manifest_url."
            " If you're running on 'localhost:5000', try 'http://localhost:5000'"
        )

    main(args)
