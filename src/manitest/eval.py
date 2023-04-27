"""
Helper functions for classification tasks
"""
import os
import json
from typing import List, Dict, Tuple, Union, Any, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from datasets import DatasetDict
from loguru import logger
from manitest.metrics import generation_metric
from manitest.base import Task, TaskType, Prompt
from manitest.utils import (
    logsumexp,
    manifest_model_config,
    manifest_tokenizer_config,
    manifest_score_sequences,
    manifest_generate_text,
    generation_multilabel_metric,
)

####################################
# Master evaluation runner
####################################


def run_eval(
    manifest,
    dataset: DatasetDict,
    task: Task,
    output_dir: str,
    batch_size: int = 10,
    seed: int = 0,
    n_shots: int = 0,
    in_context_shot_dataset: Optional[DatasetDict] = None,
    *args,
    **kwargs,
) -> Dict[str, Dict[str, Union[pd.DataFrame, Dict]]]:
    """Run all examples across all splits in the given `dataset` through the prompts of the given `task`
    Also responsible for logging + saving results.
    """

    # Setup logging
    path_to_log_file: str = os.path.join(output_dir, "info.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file

    # Log model + tokenizer + task configs
    model_config: Dict = manifest_model_config(manifest)
    tokenizer_config: Dict = manifest_tokenizer_config(manifest)
    logger.info(f"Model config:\n{model_config}")
    logger.info(f"Tokenizer config:\n{tokenizer_config}")
    logger.info(f"Task config:\nName: {task.name}\nType: {task.task_type}")
    logger.info("Prompts:\n" + "\n".join([f"{idx + 1}. " + str(p) for idx, p in enumerate(task.prompts)]))

    # Run classification/generation task
    if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
        eval_func = run_classification
        metric_func = metric_classification
        metric_agg_func = metric_agg_classification
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        eval_func = run_multilabel_classification
        metric_func = metric_classification
        metric_agg_func = metric_agg_classification
    elif task.task_type == TaskType.GENERATION:
        eval_func = run_generation
        metric_func = metric_generation
        metric_agg_func = metric_agg_generation
    else:
        raise ValueError(f"Task type '{task.task_type}' not supported")

    prompts: List[Prompt] = task.prompts
    prompt_to_metrics: Dict[str, Dict[str, Dict]] = {}
    for prompt in prompts:
        # Run prompt through Manifest
        results = eval_func(
            manifest,
            dataset,
            prompt,
            output_dir,
            batch_size=batch_size,
            n_shots=n_shots,
            in_context_shot_dataset=in_context_shot_dataset,
            *args,
            **kwargs,
        )

        # Calculate metrics
        # [key] = split, [value] = Dict of metrics
        metrics: Dict[str, Dict] = metric_func(results)
        prompt_to_metrics[prompt.name] = metrics

        # Save results / metrics
        for split in dataset.keys():
            results[split].to_csv(os.path.join(output_dir, f"results_{split}_{prompt.name}.csv"), index=False)
            json.dump(
                metrics[split], open(os.path.join(output_dir, f"metrics_{split}_{prompt.name}.json"), "w"), indent=4
            )
        logger.info(f"Prompt '{prompt.name}' metrics:\n{metrics}")

    # Calculate overall metrics across all prompts
    metrics_agg: Dict[str, Any] = metric_agg_func(prompt_to_metrics)

    # Save metrics
    for split in dataset.keys():
        json.dump(metrics_agg[split], open(os.path.join(output_dir, f"metrics_{split}.json"), "w"), indent=4)
    logger.info(f"Aggregated metrics across ALL prompts:\n{metrics}")


####################################
# Specific runners for each task type
####################################


def run_classification(
    manifest,
    dataset: DatasetDict,
    prompt: Prompt,
    output_dir: str,
    batch_size: int = 10,
    seed: int = 0,
    n_shots: int = 0,
    in_context_shot_dataset: Optional[DatasetDict] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Run a binary/multi-class classification task by first
    converting each example into a set of prompts
    (one for each term for each class label, as specified by the prompt's verbalizer),
    then running each prompt through Manifest,
    and finally comparing the model output to the ground truth label.
    """
    results: Dict[str, pd.DataFrame] = {}  # [key] = split, [value] = pd.DataFrame containing results

    # Adjust batch size to account for multiple terms per class for each prompt
    n_terms: int = len([x for x in prompt.verbalizer.values()])
    actual_batch_size: int = batch_size // n_terms

    logger.info(
        f"Requested batch size: {batch_size} examples |"
        f" Actual batch size: {actual_batch_size} examples |"
        f" # prompts per example: {n_terms}"
    )

    for split in dataset.keys():
        # Feed prompts through model, one 'term' at a time, where the terms are
        # taken from the verbalizer for this prompt. For each example, we generate
        # a unique prompt, pair it with a term, then feed the pair into the model.
        dfs: List[pd.DataFrame] = []
        # Store (y, y_hat, prompt, model generation) for every example (where y is the idx in `classes` for that label)
        for batch_idx in tqdm(
            range(0, len(dataset[split]), actual_batch_size), desc=f"Prompt: '{prompt.name}' | Split: '{split}'"
        ):
            # Convert batch (which is a dict of lists) into a list of dicts (one dict per example)
            batch_as_dict: Dict[list] = dataset[split][batch_idx : batch_idx + actual_batch_size]
            batch: List[Tuple] = [dict(zip(batch_as_dict, t)) for t in zip(*batch_as_dict.values())]

            # For each example in the batch...
            sequences: List[Dict] = []
            for example_idx, example in enumerate(batch):
                prompt_text: str = prompt.generate_prompt(
                    example, n_shots=n_shots, seed=seed, in_context_shot_dataset=in_context_shot_dataset
                )
                true_label: str = prompt.get_label(example)
                example_id: int = batch_idx * actual_batch_size + example_idx
                # For each class label `pred_label`...
                for pred_label, terms in prompt.verbalizer.items():
                    # For each token(s) `term` corresponding to this class label `pred_label`
                    for term in terms:
                        sequences.append(
                            {
                                "example_id": example_id,  # unique ID for each example
                                "prompt": prompt_text,
                                "term": term,  # will get appended to end of `prompt_text`
                                "true_label": true_label,
                                "pred_label": pred_label,
                                "logprob": None,  # will get filled in later by `manifest_score_sequences()`
                            }
                        )
            df = pd.DataFrame(sequences)
            prompts_with_labels: List[Tuple[str, str]] = list(zip(df["prompt"].tolist(), df["term"].tolist()))
            df["logprob"] = manifest_score_sequences(manifest, prompts_with_labels)

            # Verbalize: Aggregate probabilities across all model outputs corresponding to each class
            df = (
                df.groupby(["example_id", "pred_label"])
                .agg(
                    {
                        "logprob": logsumexp,
                        "prompt": "first",
                        "true_label": "first",
                    }
                )
                .reset_index()
            )
            # Create generation of each sequence for interpretability
            df["generation"] = [
                x[0] for x in manifest_generate_text(manifest, df["prompt"].tolist(), max_new_tokens=16, max_tokens=16)
            ]
            # Save dataframes for logging
            dfs.append(df)

        # Set prediction as most likely generated label, save this as our result
        df_raw: pd.DataFrame = pd.concat(dfs, axis=0)
        df_pred = (
            df_raw.sort_values("logprob", ascending=False).drop_duplicates(["example_id"]).sort_values("example_id")
        )
        results[split] = df_pred[["example_id", "true_label", "pred_label", "logprob", "prompt", "generation"]]

        # Log raw model outputs
        df_raw.to_csv(os.path.join(output_dir, f"raw_{split}_{prompt.name}.csv"), index=False)

    return results


def run_generation(
    manifest,
    dataset: DatasetDict,
    prompt: Prompt,
    output_dir: str,
    batch_size: int = 10,
    max_new_tokens: int = 100,
    n_shots: int = 0,
    in_context_shot_dataset: Optional[DatasetDict] = None,
    seed: int = 0,
    **kwargs,
):
    """Run a text generation task by first converting each example
    into a prompt, then running each prompt through Manifest,
    and finally comparing the model output to the ground truth label.
    """
    results: Dict[str, pd.DataFrame] = {}  # [key] = split, [value] = pd.DataFrame containing results

    # Run prompts through model
    for split in dataset.keys():
        # Feed prompts through model, one per example
        rows: List[Tuple] = []
        for batch_idx in tqdm(
            range(0, len(dataset[split]), batch_size), desc=f"Prompt: '{prompt.name}' | Split: '{split}'"
        ):
            # Convert batch (which is a dict of lists) into a list of dicts (one dict per example)
            batch_as_dict: Dict[list] = dataset[split][batch_idx : batch_idx + batch_size]
            batch: List[Tuple] = [dict(zip(batch_as_dict, t)) for t in zip(*batch_as_dict.values())]

            # Get prompts + true labels for each example in this batch
            prompts: List[str] = [
                prompt.generate_prompt(
                    example, n_shots=n_shots, seed=seed, in_context_shot_dataset=in_context_shot_dataset
                )
                for example in batch
            ]
            true_labels: List[str] = [prompt.generate_label(example) for example in batch]
            generations: List[str] = [
                x[0] for x in manifest_generate_text(manifest, prompts, max_new_tokens=max_new_tokens)
            ]
            example_ids: List[int] = [batch_idx * batch_size + i for i in range(len(batch))]
            rows.extend(list(zip(example_ids, prompts, generations, true_labels)))
        results[split] = pd.DataFrame(rows, columns=["example_id", "prompt", "generation", "true_label"])

    return results


def run_multilabel_classification(
    manifest,
    dataset: DatasetDict,
    prompt: Prompt,
    output_dir: str,
    batch_size: int = 10,
    max_new_tokens: int = 100,
    n_shots: int = 0,
    in_context_shot_dataset: Optional[DatasetDict] = None,
    seed: int = 0,
    **kwargs,
):
    # TODO - update
    # Run prompts through model
    results: Dict[str, np.ndarray] = {}  # [key] = split, [value] = np.ndarray of shape (num_examples, num_classes)
    for split in dataset.keys():
        # Store (y, y_hat, prompt, model generation) for every example (where y is the idx in `classes` for that label)
        labels_prompts_outputs: List[Tuple[str, str, str]] = []
        for i in tqdm(range(0, len(dataset[split]), batch_size)):
            batch: List[Tuple] = dataset[split][i : i + batch_size]
            prompts: List[str] = [x[0] for x in batch]
            ys: List[str] = [x[1] for x in batch]
            generations: List[str] = [
                x[0] for x in manifest_generate_text(manifest, prompts, max_new_tokens=max_new_tokens)
            ]
            for prompt, generation, y in zip(prompts, generations, ys):
                labels_prompts_outputs.append((prompt, generation, *y))
        results[split] = np.array(labels_prompts_outputs)

    # Calculate metrics
    metrics: Dict = {}
    for split in results.keys():
        generations = results[split][:, 1]
        truth = results[split][:, 2:]
        score = generation_multilabel_metric(generations, truth, prompt.verbalizer)
        metrics[split] = {
            "multilabel_score": score,
        }

    return results, metrics


####################################
# Calculate metrics
####################################


def metric_classification(results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Compute classification metrics for a specific prompt for all splits in `results`"""
    metrics: Dict[str, Dict] = {}
    for split, df in results.items():
        metrics[split] = classification_report(df["true_label"], df["pred_label"], output_dict=True)
        metrics[split]["confusion_matrix"] = confusion_matrix(df["true_label"], df["pred_label"]).tolist()
    return metrics


def metric_generation(results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Compute generation metrics for a specific prompt for all splits in `results`"""
    metrics: Dict[str, Dict] = {}
    for split, df in results.items():
        generations: List[str] = df["generation"].tolist()
        true_labels: List[str] = df["true_label"].tolist()
        metrics[split] = {
            "bleu": generation_metric(generations, true_labels, "sentence_bleu"),
            "rouge": generation_metric(generations, true_labels, "rouge"),
        }
    return metrics


def metric_agg_classification(metrics: Dict[str, Dict]) -> Dict[str, Any]:
    """Aggregate classification metrics across multiple prompts for all splits in `results`.

    Input format:
        metrics = {
            'prompt.name' : {
                'train' : { metrics... },
                'test' : { metrics... },
                'val' : { metrics... },
                ...
            },
            ...
        }
    """
    splits: List[str] = metrics[list(metrics.keys())[0]].keys()
    return {
        split: {
            "average": round(np.mean([r[split]["accuracy"] for r in metrics.values()]), 3),
            "std": round(np.std([r[split]["accuracy"] for r in metrics.values()]), 3),
            "min": round(min([r[split]["accuracy"] for r in metrics.values()]), 3),
            "max": round(max([r[split]["accuracy"] for r in metrics.values()]), 3),
            "all": [r[split]["accuracy"] for r in metrics.values()],
        }
        for split in splits
    }


def metric_agg_generation(metrics: Dict[str, Dict]) -> Dict[str, Any]:
    """Aggregate generation metrics across multiple prompts for all splits in `results`."""
    splits: List[str] = metrics[list(metrics.keys())[0]].keys()
    return {
        split: {
            "average": round(np.mean([r[split]["bleu"] for r in metrics.values()]), 3),
            "std": round(np.std([r[split]["bleu"] for r in metrics.values()]), 3),
            "min": round(min([r[split]["bleu"] for r in metrics.values()]), 3),
            "max": round(max([r[split]["bleu"] for r in metrics.values()]), 3),
            "all": [r[split]["bleu"] for r in metrics.values()],
        }
        for split in splits
    }
