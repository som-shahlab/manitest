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
if `data_dir` is `"medicationqa"` it should contain the following files:
medicationqa
├──/local-scratch/nigam/projects/clinical_llm/data/medicationqa/MedInfo2019-QA-Medications.xlsx.zip
"""

import pandas as pd
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

_DATASETNAME = "medicationqa"
_DISPLAYNAME = "medicationQA"

_DESCRIPTION = """\

"""


_HOMEPAGE = ""

_LICENSE = "PHYSIONET_LICENSE_1p5"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class medicationqaDataset(datasets.GeneratorBasedBuilder):
    """medicationqa"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="medicationqa_source",
            version=SOURCE_VERSION,
            description="medicationqa source schema",
            schema="source",
            subset_id="medicationqa",
        ),
        BigBioConfig(
            name="medicationqa_bigbio_qa",
            version=BIGBIO_VERSION,
            description="medicationqa BigBio schema",
            schema="bigbio_qa",
            subset_id="medicationqa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medicationqa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    
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
                    "MedInfo2019-QA-Medications.xlsx.zip",
                )
            )
            data_dir = os.path.join(
                extract_dir,
                "",
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "MedInfo2019-QA-Medications.xlsx"),
                    "split": "test",
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
                df = pd.read_excel(filepath)

                for idx, row in df.iterrows():
                    if row['Question'].endswith('?'):
                
                        yield idx, {
                        "id": idx,
                        "question_id": idx,
                        "document_id": 'NULL',
                        "question": row['Question'],
                        "type": row['Question Type'],
                        "choices": [],
                        "context": row['Section Title'],
                        "answer": [row['Answer']],
                        }
                    else:
                        yield idx, {
                        "id": idx,
                        "question_id": idx,
                        "document_id": 'NULL',
                        "question": row['Question']+'?',
                        "type": row['Question Type'],
                        "choices": [],
                        "context": row['Section Title'],
                        "answer": [row['Answer']],
                        }
                            


if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__, 
                                    data_dir='/local-scratch/nigam/projects/clinical_llm/data/medicationqa',
                                    name='medicationqa_bigbio_qa')
    print(dataset)