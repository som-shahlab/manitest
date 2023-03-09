"""
Hack script for querying language model.

Example Usage:

python3 -m manifest.api.app \
    --model_type huggingface \
    --model_name_or_path /local-scratch-nvme/nigam/huggingface/pretrained/gpt-j-6B \
    --model_generation_type text-generation \
    --use_hf_parallelize

python3 main.py \
    --manifest_url http://127.0.0.1:5009 \
    --path_to_dataset_config prompts/mednli.yaml \
    --path_to_dataset_dir /local-scratch/nigam/projects/clinical_llm/data/mednli \
    --path_to_output_dir ./ignore/mednli_gptj_update/

python3 main.py \
    --manifest_url http://127.0.0.1:5000 \
    --path_to_dataset_config prompts/medparasimp.yaml \
    --path_to_dataset_dir /local-scratch/nigam/projects/clinical_llm/data/medparasimp \
    --path_to_output_dir ./ignore/test/

Some example `model_name_or_path` for Nero:
    /local-scratch-nvme/nigam/huggingface/pretrained/BioMedLM
    /local-scratch-nvme/nigam/huggingface/pretrained/Bio_ClinicalBERT
    /local-scratch-nvme/nigam/huggingface/pretrained/gpt2-small

Some example `args.path_to_dataset_dir` for Nero:
    /local-scratch/nigam/projects/clinical_llm/data/mednli
    /local-scratch/nigam/projects/clinical_llm/data/mediqa_nli
    /local-scratch/nigam/projects/clinical_llm/data/mediqa_rqe
    /local-scratch/nigam/projects/clinical_llm/data/medparasimp
    /local-scratch/nigam/projects/clinical_llm/data/scitail
"""
import argparse
import os
import numpy as np
import json
import requests
from manifest import Manifest

# Custom scripts
from data import load_data
from eval import run_classification, run_generation, run_multilabel_classification


def main(args):
    # Load data + prompts
    dataset, tasks, is_classification, is_multilabel = load_data(args.path_to_dataset_config, args.path_to_dataset_dir)

    # Logging
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Manifest
    os.environ["no_proxy"] = "localhost, 127.0.0.1"  # Needed on Nero
    manifest = Manifest(
        client_name="huggingface",
        client_connection=args.manifest_url,
    )

    # Test Manifest connection
    try:
        requests.get(args.manifest_url)
    except Exception:
        raise ConnectionRefusedError(f"Error connecting to Manifest server. Is it running at {args.manifest_url} ?")

    # Determine which splits to evaluation on
    dataset = {"test": dataset["test"]}

    # Run experiments
    results, metrics = {}, {}
    for task in tasks:
        if not is_multilabel and is_classification:
            # Classification task
            result, metric = run_classification(
                manifest,
                task,
                dataset,
                batch_size=args.batch_size,
                path_to_output_dir=args.path_to_output_dir,
            )
        elif not is_multilabel:
            # Generation task
            result, metric = run_generation(
                manifest,
                task,
                dataset,
                batch_size=args.batch_size,
                path_to_output_dir=args.path_to_output_dir,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            result, metric = run_multilabel_classification(
                manifest,
                task,
                dataset,
                batch_size=args.batch_size,
                path_to_output_dir=args.path_to_output_dir,
                max_new_tokens=args.max_new_tokens,
            )

        results[task.id] = result
        metrics[task.id] = metric

    # Save metrics
    path_to_metrics_file: str = os.path.join(args.path_to_output_dir, "metrics.json")
    with open(path_to_metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Print metrics
    if is_multilabel:
        print("")
        print("------------------------------------------------------")
        print(f"----- Accuracy across {len(tasks)} templates --------------------")
        for label in tasks[0].verbalizer.keys():
            print("------------------------------------------------------")
            print(f"---------- {label} ---------------")
            print("------------------------------------------------------")
            print(
                "Average: ",
                round(
                    sum([r["test"]["multilabel_score"][label]["accuracy"] for r in metrics.values()]) / len(metrics), 3
                ),
            )
            print(
                "Std: ", round(np.std([r["test"]["multilabel_score"][label]["accuracy"] for r in metrics.values()]), 3)
            )
            print("Min: ", round(min([r["test"]["multilabel_score"][label]["accuracy"] for r in metrics.values()]), 3))
            print("Max: ", round(max([r["test"]["multilabel_score"][label]["accuracy"] for r in metrics.values()]), 3))
            print("All: ", [r["test"]["multilabel_score"][label]["accuracy"] for r in metrics.values()])
    elif is_classification:
        print("")
        print("------------------------------------------------------")
        print(f"---------- Accuracy across {len(tasks)} templates ---------------")
        print("------------------------------------------------------")
        print("Average: ", round(sum([r["test"]["accuracy"] for r in metrics.values()]) / len(metrics), 3))
        print("Std: ", round(np.std([r["test"]["accuracy"] for r in metrics.values()]), 3))
        print("Min: ", round(min([r["test"]["accuracy"] for r in metrics.values()]), 3))
        print("Max: ", round(max([r["test"]["accuracy"] for r in metrics.values()]), 3))
        print("All: ", [r["test"]["accuracy"] for r in metrics.values()])
    else:
        with open(os.path.join(args.path_to_output_dir, "metrics.txt"), "w") as f:
            f.write("------------------------------------------------------\n")
            f.write(f"----- BLEU across {len(tasks)} templates --------------------\n")
            f.write("------------------------------------------------------\n")
            f.write(f"Average: {round(sum([ r['test']['bleu'] for r in metrics.values()]) / len(metrics), 3)}\n")
            f.write(f"Std: {round(np.std([ r['test']['bleu'] for r in metrics.values()]), 3)}\n")
            f.write(f"Min: {round(min([ r['test']['bleu'] for r in metrics.values()]), 3)}\n")
            f.write(f"Max: {round(max([ r['test']['bleu'] for r in metrics.values()]), 3)}\n")
            f.write(f"All: {[ r['test']['bleu'] for r in metrics.values()]}\n")
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument(
        "--manifest_url",
        type=str,
        help="Host + port where Manifest server is running, e.g. 'localhost:5000'",
        required=True,
    )
    parser.add_argument(
        "--path_to_dataset_config",
        type=str,
        help="Path to configuration YAML for this dataset",
        required=True,
    )
    parser.add_argument(
        "--path_to_dataset_dir",
        type=str,
        help="Path to DIRECTORY containing your dataset",
        required=True,
    )
    parser.add_argument(
        "--path_to_output_dir",
        type=str,
        help="Path to DIRECTORY to output logs / results / metrics",
        required=True,
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
    args = parser.parse_args()
    main(args)
