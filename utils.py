from typing import List, Dict, Tuple
from langchain import PromptTemplate
import os
from jinja2schema import infer
import jinja2
import torch
from datasets import DatasetDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import requests
from manifest import Manifest
import numpy as np
import re

########################################################
########################################################
# Manifest querying
########################################################
########################################################


def manifest_model_config(manifest: Manifest) -> Dict:
    model_config: Dict = requests.get(manifest.client.host + "/model_config").json()
    return model_config


def manifest_tokenizer_config(manifest: Manifest) -> str:
    tokenizer_config: str = requests.get(manifest.client.host + "/tokenizer_config").text
    return tokenizer_config


def manifest_score_sequences(manifest: Manifest, prompts_with_labels: List[Tuple[str, str]]) -> List[float]:
    try:
        results: Dict = requests.post(
            manifest.client.host + "/score_sequence_eleuther_lm_eval", json={"prompts_with_labels": prompts_with_labels}
        ).json()
        scores = [(x["label_prob"]) for x in results]
    except Exception as e:
        print(str(e))
        raise RuntimeError(
            "No value returned from Manifest. This is probably a CUDA out-of-memory error. Try reducing the batch size."
        )
    return scores


def manifest_generate_text(manifest: Manifest, sequences: List[str], **kwargs) -> List[Tuple[str, float]]:
    try:
        results: Dict = requests.post(
            manifest.client.host + "/completions", json={"prompt": sequences, **kwargs}
        ).json()
        generations = [(x["text"], x["logprob"]) for x in results["choices"]]
    except Exception as e:
        print(str(e))
        raise RuntimeError(
            "No value returned from Manifest. This is probably a CUDA out-of-memory error. Try reducing the batch size."
        )
    return generations


########################################################
########################################################
# Helper functions
########################################################
########################################################
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def generation_multilabel_metric(generations, truth, verbalizer):
    """
    Compute multilabel classification metrics for a list of generations and truth labels.
    Args:
        generations: list of strings
        truth: list of lists of strings
        verbalizer: dict of label strings to list of label strings
    """
    results = {}
    label_idx = 0
    for label_string, label_list in verbalizer.items():
        label_regex = "|".join(label_list)
        predictions = []
        labels = []
        for i, gen in enumerate(generations):
            if re.search(label_regex, gen):
                predictions.append(1)
            else:
                predictions.append(0)
            label = truth[i][label_idx]
            if label == "yes":
                labels.append(1)
            else:
                labels.append(0)

        results[label_string] = classification_report(labels, predictions, output_dict=True)

    return results


def generate_prompts_with_injected_examples(
    dataset: DatasetDict, prompt_template: str, output_column: str, output_template: str
) -> Dict[str, List[Tuple[str, str]]]:
    """Create prompts and inject dataset examples into prompt template.
    Assumes that the variable in the Jinja template has the same name as the
    attribute in each dataset example that we want to inject into the prompt.

    Args:
        dataset (DatasetDict): HuggingFace DatasetDict
        prompt_template (str): Jinja template for prompt
        output_column (str): Name of column in dataset that contains the ground truth output
        output_template (str): Jinja template for transforming `output_column` into actual expected output

    Returns:
        Dict[str, List[Tuple[str, str]]]: {
            'train' : [
                (injected prompt, expected output),
                (injected prompt, expected output),
                ...
            ],
            'test' : [
                (injected prompt, expected output),
                (injected prompt, expected output),
                ...
            ],
            ...
        }
    """
    inputs_by_split: Dict[str, List[Tuple[str, str]]] = {}

    # Generate prompt template
    input_variables: List[str] = list(infer(prompt_template).keys())
    prompt = PromptTemplate(template=prompt_template, input_variables=input_variables, template_format="jinja2")

    # Jinja2 parsing
    jinja_renderer = jinja2.Environment()

    # Inject examples into prompt
    for split in dataset.keys():
        inputs_by_split[split] = []
        for example in dataset[split]:
            # Inject example into prompt
            injected_prompt: str = prompt.format(**example)
            # Format output
            # if output column is a list
            if isinstance(output_column, list):
                expected_output = []
                for i in range(len(output_column)):
                    expected_output.append(
                        jinja_renderer.from_string(output_template).render(output=example[output_column[i]])
                    )
            else:
                expected_output = jinja_renderer.from_string(output_template).render(output=example[output_column])
            inputs_by_split[split].append((injected_prompt, expected_output))

    return inputs_by_split


def generate_text(model, input_ids, max_new_tokens: int = 10) -> Dict[str, torch.Tensor]:
    """
    Use a generative model to generate text + logits ("scores") for each output token.
    Use `default_settings` to specify a default setup
    """
    # NOTE: Unused args in Eleuther Eval harness, so commenting out
    #   default_settings: Optional[str] = None,
    #   is_greedy_decoding: bool = True,
    #   top_k: float = 0.0,
    #   top_p: float = 0.0,
    #   num_beams: int = 5,
    #   max_new_tokens: int = 20
    # if default_settings == 'greedy':
    #     is_greedy_decoding = True
    # elif default_settings == 'beam':
    #     is_greedy_decoding = True
    #     num_beams = 5
    # elif default_settings == 'nucleus':
    #     is_greedy_decoding = False
    #     top_p = 0.9

    # TODO - more intelligent prompt truncation
    # Limit the number of new tokens to fit within the model's context window
    max_new_tokens = max(0, min(model.config.n_ctx - input_ids.shape[1], max_new_tokens))
    outputs = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # do_sample=not is_greedy_decoding,
        # top_k=None if is_greedy_decoding else top_k,
        # top_p=None if is_greedy_decoding else top_p,
        # num_beams=num_beams,
        output_scores=True,
        return_dict_in_generate=True,
    )  # dict, each element is: batch_size x seq_len x vocab_size
    return outputs


def get_predicted_token_logits(model, tokenizer, input_ids):
    """
    Get the logits for the predicted token
    (i.e. the token that was masked for a BERT model, or the first output token from a Generative model)
    """
    batch_size: int = input_ids.shape[0]
    prompt_len: int = input_ids.shape[1]  # prompt_len = len(input) + len(special tokens)
    if model.config.model_type in ["bert"]:
        token_logits = model(input_ids)[0].detach().cpu().numpy()  # batch_size x prompt_len x vocab_size
        mask_token_idx = input_ids.detach().cpu().numpy() == tokenizer.mask_token_id  # batch_size x prompt_len
        predicted_token_logits = token_logits[
            mask_token_idx, :
        ]  # batch_size x vocab_size (just the logits for the predicted/masked token)
        generation = ""  # bert models don't generate text
        assert mask_token_idx.shape == (
            batch_size,
            predicted_token_logits.shape[1],
        ), f"Mask token idx shape {mask_token_idx.shape} != predicted token logits shape {predicted_token_logits.shape}"
    elif model.config.model_type in [
        "gpt2",
        "t5",
        "bart",
    ]:
        outputs = generate_text(model, input_ids)
        # NOTE: outputs['scores'] starts after the prompt, while outputs['sequences']
        # includes the prompt in its sequence
        # (batch_size * num_beams * num_return_sequences) x vocab_size (just the logits for the predicted/masked token)
        predicted_token_logits = outputs["scores"][0].detach().cpu().numpy()
        predicted_sequences = outputs["sequences"].detach().cpu().numpy()  # batch_size x max_length_of_output
        generation = tokenizer.decode(
            predicted_sequences[0][prompt_len:], skip_special_tokens=False
        )  # get actual text generated by model
    else:
        raise ValueError(f"Model type `{model.config.model_type}` not supported by `get_predicted_token_logits()`")
    return predicted_token_logits, generation


def tokenize_prompts(tokenizer, inputs_by_split) -> Dict[str, List[Tuple[str, str]]]:
    """
    Tokenize the prompts in `inputs_by_split`.

    Returns List of (tokenized_prompt, y) tuples, where y is the label for the prompt.
    """
    tokenized_inputs: Dict[str, List[Tuple[str, str]]] = {}  # [key] = split, [value] = list of (tokenized input_ids, y)

    for split in inputs_by_split.keys():
        tokenized_inputs[split] = []
        for input, y in tqdm(inputs_by_split[split], total=len(inputs_by_split[split])):
            input_ids = tokenizer.encode(input, return_tensors="pt")  # batch_size x seq_len
            tokenized_inputs[split].append((input_ids, y))
    return tokenized_inputs


########################################################
########################################################
# Logging
########################################################
########################################################


def log_model_tokenizer_config(path_to_output_dir: str, manifest: Manifest):
    model_config: Dict = manifest_model_config(manifest)
    tokenizer_config: Dict = manifest_tokenizer_config(manifest)
    with open(os.path.join(path_to_output_dir, "config_model.txt"), "w") as f:
        f.write("==========================================\n")
        f.write("========== MODEL + TOKENIZER =============\n")
        f.write("==========================================\n")
        f.write("\n")
        f.write(f"Model config:\n\n{model_config}")
        f.write("\n\n")
        f.write(f"Tokenizer config:\n\n{tokenizer_config}\n")


def log_task_config(path_to_output_dir: str, task, dataset: DatasetDict):
    with open(os.path.join(path_to_output_dir, f"config_task_{task.id}.txt"), "w") as f:
        f.write("==========================================\n")
        f.write("============== TASK CONFIG ===============\n")
        f.write("==========================================\n")
        f.write("\n")
        f.write(f"Template:\n```\n{task.template}\n```\n")
        f.write("\n\n")
        f.write(f"Output column:\n{task.output_column}\n")
        f.write("\n\n")
        f.write(f"Output template:\n```\n{task.output_template}\n```\n")
        f.write("\n\n")
        f.write(f"Dataset config:\n\n{dataset}")
        f.write("\n\n")
        if task.verbalizer is not None:
            classes: List[str] = list(task.verbalizer.keys())
            f.write(f"Classes:\n\n{classes}\n")
            f.write("\n\n")
            f.write(f"Label Map:\n\n{task.verbalizer}\n")
            f.write("\n\n")
            # Print count of each type of label
            f.write("Count of each type of label:\n")
            for split in dataset.keys():
                total = len(dataset[split])
                f.write(f"\tSplit {split}: {total} examples\n")
                # FIXME: update this for multilabel
                if isinstance(task.output_column, str):
                    for label in classes:
                        label_count = len([1 for example in dataset[split] if example[task.output_column] == label])
                        f.write(f"\t\t{label}:\t{label_count} ({int(100 * label_count / total)}%)\n")


def log_generation_results(path_to_output_dir: str, task, results):
    with open(os.path.join(path_to_output_dir, f"metrics_{task.id}.txt"), "w") as f:
        for split in results.keys():
            f.write("==========================================\n")
            f.write(f"================= {split} ===================\n")
            f.write("==========================================\n")
            # TODO


def log_classification_results(path_to_output_dir: str, task, results):
    classes: List[str] = list(task.verbalizer.keys())
    with open(os.path.join(path_to_output_dir, f"metrics_{task.id}.txt"), "w") as f:
        for split in results.keys():
            f.write("==========================================\n")
            f.write(f"================= {split} ===================\n")
            f.write("==========================================\n")
            f.write("\n")
            y = results[split][:, 2]
            y_hat = results[split][:, 3]
            f.write("\n------------\n")
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(y, y_hat)
            f.write(f"TN: = {cm[0,0]} | FP = {cm[0,1]}\n")
            f.write(f"FN: = {cm[1,0]} | TP = {cm[1,1]}\n")
            f.write("------------\n")
            f.write(f"Accuracy = {round(accuracy_score(y, y_hat), 3)}\n")
            f.write("------------\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y, y_hat, target_names=classes))


def log_generations(path_to_output_dir: str, task, split: str, prompts: List[str], generations: List[str]):
    path_to_file: str = os.path.join(path_to_output_dir, f"generations_{task.id}_{split}.txt")
    with open(path_to_file, "w") as f:
        for (prompt, generation) in zip(prompts, generations):
            f.write(f"=> Prompt:\n```\n{prompt}\n```\n")
            f.write(f"=> Generation:\n```\n{generation}\n```\n")
            f.write("\n")
