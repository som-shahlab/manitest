import re
import requests
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report
from manifest import Manifest

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
    # TODO - update
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
