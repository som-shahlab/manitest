"""
Dataset-specific settings for the prompting framework.

Important: You can't simply do `load_dataset('bigbio/mednli')` because Nero
will throw a "ConnectionError: Couldn't reach 'bigbio/mediqa_nli' on the Hub (ProxyError)"

Instead, you must download the bigbio scripts themselves into the `bigbio_datasets` folder,
and download the raw data loaded by the bigbio scripts onto Nero.
"""
import os
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Tuple, NamedTuple, Optional
from ruamel.yaml import YAML


class Task(NamedTuple):
    id: str
    template: str
    # Determining expected output (e.g. "ground truth") from model
    output_column: str
    output_template: str
    # Classification-specific params
    verbalizer: Optional[Dict[str, List[str]]]


def parse_yaml_config(path_to_dataset_config: str):
    yaml_dict: Dict = YAML().load(open(path_to_dataset_config, "r"))

    # Validate

    # All required keys are specified
    REQUIRED_KEYS: List[str] = [
        "dataset_name",
        "dataloader",
        "dataloader_subset",
        "is_classification",
        "output_column",
        "prompts",
    ]
    for key in REQUIRED_KEYS:
        if key not in yaml_dict:
            raise ValueError(
                f"You must specify the '{key}' attribute in your YAML config file @ {path_to_dataset_config}"
            )

    # `is_classification` is a boolean
    if not isinstance(yaml_dict["is_classification"], bool):
        raise ValueError(
            "You must specify the 'is_classification' attribute as a Boolean"
            f" in your YAML config file @ {path_to_dataset_config}"
        )

    # BigBio data loader script exists
    if not os.path.exists(yaml_dict["dataloader"]):
        print("====>", os.path.abspath(yaml_dict["dataloader"]))
        raise ValueError(f"A data loader script @ {yaml_dict['dataloader']} does not exist")

    # Prompts are correctly specified
    prompts: Dict[str, Dict] = yaml_dict["prompts"]
    if len(prompts) < 1:
        raise ValueError(f"You must specify at least one prompt in your YAML config file @ {path_to_dataset_config}")
    for id, data in prompts.items():
        if "template" not in data:
            raise ValueError(
                f"You must specify the 'template' attribute for prompt '{id}'"
                f" in your YAML config file @ {path_to_dataset_config}"
            )

    # Classification-specific
    # All labels that columns are mapped to are valid labels - check that column_map matches label_map's
    # TODO

    # Return contents of YAML file
    return yaml_dict


def load_data(path_to_dataset_config: str, path_to_data_dir: str) -> Tuple[DatasetDict, List[Task], bool]:
    # Load config.yaml file
    yaml_dict: Dict = parse_yaml_config(path_to_dataset_config)
    is_classification: bool = yaml_dict["is_classification"]

    is_multilabel: bool = False
    if isinstance(yaml_dict["output_column"], list):
        is_multilabel = True
    # Load dataset
    dataloader: str = yaml_dict["dataloader"]
    dataloader_subset: str = yaml_dict["dataloader_subset"]
    dataset = load_dataset(dataloader, data_dir=path_to_data_dir, name=dataloader_subset)

    # Ground truth output
    output_column: str = yaml_dict["output_column"]

    # Column remapping
    if "column_map" in yaml_dict:
        # Remap every attribute of this dataset's examples per the `column_map`
        for column, updates in yaml_dict["column_map"].items():

            def remap(example):
                for original_value, updated_value in updates.items():
                    if example[column] == original_value:
                        example[column] = updated_value
                return example

            dataset = dataset.map(remap)

    # Load prompts
    tasks: List[Task] = []
    prompts: Dict[str, Dict] = yaml_dict["prompts"]
    for id, data in prompts.items():
        # Jinja prompt template
        template: str = data["template"]

        # Column corresponding to ground truth label
        # Defaults to `{{output}}` (i.e. identity function) if not specified
        output_template: str = data.get("output_template", "{{output}}")

        if is_classification:
            # CLASSIFICATION-SPECIFIC PARAMS
            # Label map for classification tasks
            verbalizer = {
                k: v for d in data["verbalizer"] for k, v in d.items()
            }  # [key] = class, [value] = list of terms mapping to this class
            tasks.append(Task(id, template, output_column, output_template, verbalizer))
        else:
            # GENERATION-SPECIFIC PARAMS
            tasks.append(Task(id, template, output_column, output_template, None))

    return dataset, tasks, is_classification, is_multilabel
