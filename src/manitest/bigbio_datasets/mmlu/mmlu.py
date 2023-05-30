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
if `data_dir` is `"mmlu"` it should contain the following files:
mmlu
├── /local-scratch/nigam/projects/clinical_llm/data/mmlu/data.tar
"""

import json
import os
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import qa_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

import pandas as pd


_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = True
_CITATION = """\
"""

_DATASETNAME = "mmlu"
_DISPLAYNAME = "mmlu"

_DESCRIPTION = """\
"""


_HOMEPAGE = "https://github.com/hendrycks/test"

_LICENSE = "PHYSIONET_LICENSE_1p5"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

clinical_topics = ["anatomy", "clinical_knowledge", "college_medicine", 
                    "medical_genetics", "professional_medicine", "college_biology"]


class mmluDataset(datasets.GeneratorBasedBuilder):
    """mmlu"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mmlu_source",
            version=SOURCE_VERSION,
            description="mmlu source schema",
            schema="source",
            subset_id="mmlu",
        ),
        BigBioConfig(
            name="mmlu_bigbio_qa",
            version=BIGBIO_VERSION,
            description="mmlu BigBio schema",
            schema="bigbio_qa",
            subset_id="mmlu",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mmlu_source"

    def _info(self) -> datasets.DatasetInfo:

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
                    "data.tar",
                )
            )
            data_dir = os.path.join(
                extract_dir,
                "data",
            )

        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            #     gen_kwargs={
            #         "dirpath": os.path.join(data_dir, "auxiliary_train"),
            #         "split": "train",
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "dirpath": os.path.join(data_dir, "test"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "dirpath": os.path.join(data_dir, "val"),
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(self, dirpath, split: str) -> Tuple[int, Dict]:
        
        # if in 6 clinical topics
        for topic in clinical_topics:
            filename = f'{topic}_{split}.csv'

            if filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
            
                df = pd.read_csv(filepath)
                #print(filepath, '='*50)
                if self.config.schema == "source":
                    for line in f:
                        json_line = json.loads(line)
                        yield json_line["title"], json_line

                elif self.config.schema == "bigbio_qa":
                    
                    for row in df.iterrows():
                        
                        yield topic+str(row[0]), {
                            "id": row[0],
                            "question_id": row[0],
                            "document_id": row[0],
                            "question": row[1].iloc[0],
                            "type": 'multi-choice',
                            "choices": [row[1].iloc[1],row[1].iloc[2],row[1].iloc[3],row[1].iloc[4],],
                            "context": 'NULL',
                            "answer": [row[1].iloc[5]],
                            }


if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__, 
                                    data_dir='/local-scratch/nigam/projects/clinical_llm/data/mmlu',
                                    name='mmlu_bigbio_qa')
    print(dataset)