# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The files comprising this dataset must be on the users local machine in a single directory that is
passed to `datasets.load_datset` via the `data_dir` kwarg. This loader script will read the archive
files directly (i.e. the user should not uncompress, untar or unzip any of the files). For example,
if `data_dir` is `"radqa"` it should contain the following files:
radqa
├── /local-scratch/nigam/projects/clinical_llm/data/radqa-a-natural-language-inference-dataset-for-the-radiology-domain-1.0.0.zip
"""

import json
import os
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import qa_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks


_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = True
_CITATION = """\

"""

_DATASETNAME = "radqa"
_DISPLAYNAME = "RadQA"

_DESCRIPTION = """\

"""


_HOMEPAGE = "https://physionet.org/content/radqa-report-inference/1.0.0/"

_LICENSE = "PHYSIONET_LICENSE_1p5"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class RadqaDataset(datasets.GeneratorBasedBuilder):
    """radqa"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="radqa_source",
            version=SOURCE_VERSION,
            description="radqa source schema",
            schema="source",
            subset_id="radqa",
        ),
        BigBioConfig(
            name="radqa_bigbio_qa",
            version=BIGBIO_VERSION,
            description="radqa BigBio schema",
            schema="bigbio_qa",
            subset_id="radqa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "radqa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": "0",
                    "document_id": "24267510",
                    "question_id": "55031181e9bde69634000014",
                    "question": "Is RANKL secreted from the cells?",
                    "type": "yesno",
                    "choices": [],
                    "context": "Osteoprotegerin (OPG) is a soluble secreted factor that acts as a decoy receptor for receptor activator of NF-\u03baB ligand (RANKL)",
                    "answer": ["yes"],
                }
            )

        # need to modify:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    
                }
            )
        # simplified schema for QA tasks

        elif self.config.schema == "bigbio_qa":
            features = qa_features


        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            extract_dir = dl_manager.extract(
                os.path.join(
                    self.config.data_dir,
                    "radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0.zip",
                )
            )
            data_dir = os.path.join(
                extract_dir,
                "radqa-a-question-answering-dataset-to-improve-comprehension-of-radiology-reports-1.0.0",
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.json"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        with open(filepath, "r") as f:
            if self.config.schema == "source":
                for line in f:
                    json_line = json.loads(line)
                    yield json_line["title"], json_line

            elif self.config.schema == "bigbio_qa":
                file_data = json.loads(f.read())
                file_data = file_data['data']
                for line in file_data:
                    json_line = line["paragraphs"]
                    for qa_pairs in json_line:
                        for qa in qa_pairs['qas']:
                            try:
                                yield qa['id'], {
                                "id": qa['id'],
                                "question_id": qa['id'],
                                "document_id": qa_pairs['document_id'],
                                "question": qa['question'],
                                "type": 'factoid',
                                "choices": [],
                                "context": qa_pairs["context"],
                                "answer": [qa["answers"][0]['text']],
                                }
                            except:
                                yield qa['id'], {
                                "id": qa['id'],
                                "question_id": qa['id'],
                                "document_id": qa_pairs['document_id'],
                                "question": qa['question'],
                                "type": 'factoid',
                                "choices": [],
                                "context": qa_pairs["context"],
                                "answer": [''],
                                }


if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__, 
                                    data_dir='/local-scratch/nigam/projects/clinical_llm/data/radqa',
                                    name='radqa_bigbio_qa')
    print(dataset)