"""
Helper functions for classification tasks
"""
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report
from openprompt.utils.metrics import generation_metric
import numpy as np
from utils import (
    generate_prompts_with_injected_examples,
    logsumexp,
    manifest_score_sequences,
    manifest_generate_text,
    generation_multilabel_metric,
)
from utils import (
    log_model_tokenizer_config,
    log_task_config,
    log_classification_results,
    log_generations,
    log_generation_results,
)
from data import Task
import pandas as pd


def run_task(func):
    def wrapper(
        manifest, task: Task, dataset, batch_size: int = 10, path_to_output_dir: str = "./output/", *args, **kwargs
    ):
        # Setup
        prompt_template: str = task.template
        output_column: str = task.output_column
        output_template: str = task.output_template

        # Load dataset into prompt format
        inputs_by_split: Dict[str, List[Tuple[str, str]]] = generate_prompts_with_injected_examples(
            dataset, prompt_template, output_column, output_template
        )

        # Logging
        log_task_config(path_to_output_dir, task, dataset)
        log_model_tokenizer_config(path_to_output_dir, manifest)

        # Run classification/generation task
        results, metrics = func(manifest, task, inputs_by_split, batch_size, path_to_output_dir, *args, **kwargs)

        # Logging
        for split in results.keys():
            prompts = results[split][:, 0]
            generations = results[split][:, 1]
            log_generations(path_to_output_dir, task, split, prompts, generations)

        return results, metrics

    return wrapper


@run_task
def run_classification(
    manifest,
    task,
    inputs_by_split: Dict[str, List[Tuple[str, str]]] = {},
    batch_size: int = 10,
    path_to_output_dir: str = "",
    **kwargs,
) -> Tuple:
    # Run prompts through model
    classes: List[str] = list(task.verbalizer.keys())
    results: Dict[str, np.ndarray] = {}  # [key] = split, [value] = np.ndarray of shape (num_examples, num_classes)
    for split in inputs_by_split.keys():
        dfs_raw: List[pd.DataFrame] = []
        dfs_pred: List[pd.DataFrame] = []
        # Store (y, y_hat, prompt, model generation) for every example (where y is the idx in `classes` for that label)
        labels_preds_prompts_outputs: List[Tuple[str, str, int, int]] = []
        terms_len: int = len([x for x in task.verbalizer.values()])
        batch_len: int = (
            batch_size // terms_len
        )  # adjust batch size to account for multiple terms per class for each prompt
        for i in tqdm(range(0, len(inputs_by_split[split]), batch_len)):
            batch: List[Tuple] = inputs_by_split[split][i : i + batch_len]
            # Create unique prompt for each class
            sequences: List[Dict] = []
            for prompt_idx, (prompt, y) in enumerate(batch):
                for label, terms in task.verbalizer.items():
                    for term in terms:
                        sequences.append(
                            {
                                "prompt_idx": prompt_idx,
                                "prompt": prompt,
                                "y": y,
                                "term": term,
                                "y_hat": label,
                                "logprob": None,
                            }
                        )
            df = pd.DataFrame(sequences)
            prompts_with_labels: List[Tuple[str, str]] = list(zip(df["prompt"].tolist(), df["term"].tolist()))
            df["logprob"] = manifest_score_sequences(manifest, prompts_with_labels)
            # Verbalize: Aggregate probabilities across all model outputs corresponding to each class
            df = (
                df.groupby(["prompt_idx", "y_hat"])
                .agg(
                    {
                        "logprob": logsumexp,
                        "prompt": "first",
                        "y": "first",
                    }
                )
                .reset_index()
            )
            # Create generation of each sequence for interpretability
            generations: List[str] = [
                x[0] for x in manifest_generate_text(manifest, df["prompt"].tolist(), max_new_tokens=32)
            ]
            df["generation"] = generations
            # Set prediction as most likely token
            df_preds = (
                df.sort_values("logprob", ascending=False).drop_duplicates(["prompt_idx"]).sort_values("prompt_idx")
            )
            for idx, row in df_preds.iterrows():
                labels_preds_prompts_outputs.append((row["prompt"], row["generation"], row["y"], row["y_hat"]))
            # Save dataframes for logging
            dfs_raw.append(df)
            dfs_pred.append(df_preds)
        # Log dataframes
        pd.concat(dfs_raw, axis=0).to_csv(f"{path_to_output_dir}/df_raw_{task.id}.csv", index=False)
        pd.concat(dfs_pred, axis=0).to_csv(f"{path_to_output_dir}/df_pred_{task.id}.csv", index=False)
        results[split] = np.array(labels_preds_prompts_outputs)
    # Calculate metrics
    metrics: Dict = {}
    for split in results.keys():
        y = results[split][:, 2]
        y_hat = results[split][:, 3]
        metrics[split] = classification_report(y, y_hat, target_names=classes, output_dict=True)

    # Logging
    log_classification_results(path_to_output_dir, task, results)

    return results, metrics


@run_task
def run_generation(
    manifest,
    task,
    inputs_by_split: Dict[str, List[Tuple[str, str]]] = {},
    batch_size: int = 10,
    path_to_output_dir: str = "",
    max_new_tokens: int = 100,
    **kwargs,
):
    # Run prompts through model
    results: Dict[str, np.ndarray] = {}  # [key] = split, [value] = np.ndarray of shape (num_examples, num_classes)
    for split in inputs_by_split.keys():
        # Store (y, y_hat, prompt, model generation) for every example (where y is the idx in `classes` for that label)
        labels_prompts_outputs: List[Tuple[str, str, str]] = []
        for i in tqdm(range(0, len(inputs_by_split[split]), batch_size)):
            batch: List[Tuple] = inputs_by_split[split][i : i + batch_size]
            prompts: List[str] = [x[0] for x in batch]
            ys: List[str] = [x[1] for x in batch]
            generations: List[str] = [
                x[0] for x in manifest_generate_text(manifest, prompts, max_new_tokens=max_new_tokens)
            ]
            for (prompt, generation, y) in zip(prompts, generations, ys):
                labels_prompts_outputs.append((prompt, generation, y))
        results[split] = np.array(labels_prompts_outputs)

    # Calculate metrics
    metrics: Dict = {}
    for split in results.keys():
        generations = results[split][:, 1]
        truth = results[split][:, 2]
        bleu = generation_metric(generations, truth, "sentence_bleu")
        metrics[split] = {
            "bleu": bleu,
        }

    # Logging
    log_generation_results(path_to_output_dir, task, results)

    return results, metrics


@run_task
def run_multilabel_classification(
    manifest,
    task,
    inputs_by_split: Dict[str, List[Tuple[str, str]]] = {},
    batch_size: int = 10,
    path_to_output_dir: str = "",
    max_new_tokens: int = 100,
    **kwargs,
):
    # Run prompts through model
    results: Dict[str, np.ndarray] = {}  # [key] = split, [value] = np.ndarray of shape (num_examples, num_classes)
    for split in inputs_by_split.keys():
        # Store (y, y_hat, prompt, model generation) for every example (where y is the idx in `classes` for that label)
        labels_prompts_outputs: List[Tuple[str, str, str]] = []
        for i in tqdm(range(0, len(inputs_by_split[split]), batch_size)):
            batch: List[Tuple] = inputs_by_split[split][i : i + batch_size]
            prompts: List[str] = [x[0] for x in batch]
            ys: List[str] = [x[1] for x in batch]
            generations: List[str] = [
                x[0] for x in manifest_generate_text(manifest, prompts, max_new_tokens=max_new_tokens)
            ]
            for (prompt, generation, y) in zip(prompts, generations, ys):
                labels_prompts_outputs.append((prompt, generation, *y))
        results[split] = np.array(labels_prompts_outputs)

    # Calculate metrics
    metrics: Dict = {}
    for split in results.keys():
        generations = results[split][:, 1]
        truth = results[split][:, 2:]
        score = generation_multilabel_metric(generations, truth, task.verbalizer)
        metrics[split] = {
            "multilabel_score": score,
        }

    # Logging
    log_generation_results(path_to_output_dir, task, results)

    return results, metrics
