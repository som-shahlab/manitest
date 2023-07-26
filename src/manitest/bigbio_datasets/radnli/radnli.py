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
if `data_dir` is `"radnli"` it should contain the following files:
radnli
├── /local-scratch/nigam/projects/clinical_llm/data/radnli-a-natural-language-inference-dataset-for-the-radiology-domain-1.0.0.zip
"""

import json
import os
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import entailment_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks


_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = True
_CITATION = """\
@misc{https://doi.org/10.13026/c2rs98,
    title        = {RadNLI: A natural language inference dataset for the radiology domain},
    author       = {Miura, Y., Zhang, Y., Tsai, E., Langlotz, C., & Jurafsky, D.},
    year         = 2021,
    publisher    = {physionet.org},
    doi          = {10.13026/mmab-c762},
    url          = {https://physionet.org/content/radnli-report-inference/1.0.0/}
}
"""

_DATASETNAME = "radnli"
_DISPLAYNAME = "RadNLI"

_DESCRIPTION = """\
State of the art models using deep neural networks have become very good in learning an accurate
mapping from inputs to outputs. However, they still lack generalization capabilities in conditions
that differ from the ones encountered during training. This is even more challenging in specialized,
and knowledge intensive domains, where training data is limited. To address this gap, we introduce
RadNLI - a dataset annotated by doctors, performing a natural language inference task (NLI),
grounded in the medical history of patients. As the source of premise sentences, we used the
MIMIC-CXR. More specifically, to minimize the risks to patient privacy, we worked with clinical
notes corresponding to the deceased patients. The clinicians in our team suggested the Past Medical
History to be the most informative section of a clinical note, from which useful inferences can be
drawn about the patient.
"""


_HOMEPAGE = "https://physionet.org/content/radnli-report-inference/1.0.0/"

_LICENSE = "PHYSIONET_LICENSE_1p5"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class RadnliDataset(datasets.GeneratorBasedBuilder):
    """radnli"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="radnli_source",
            version=SOURCE_VERSION,
            description="radnli source schema",
            schema="source",
            subset_id="radnli",
        ),
        BigBioConfig(
            name="radnli_bigbio_te",
            version=BIGBIO_VERSION,
            description="radnli BigBio schema",
            schema="bigbio_te",
            subset_id="radnli",
        ),
    ]

    DEFAULT_CONFIG_NAME = "radnli_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pair_id": datasets.Value("string"),
                    "gold_label": datasets.Value("string"),
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_te":
            features = entailment_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            extract_dir = dl_manager.extract(
                os.path.join(
                    self.config.data_dir,
                    "radnli-a-natural-language-inference-dataset-for-the-radiology-domain-1.0.0.zip",
                )
            )
            data_dir = os.path.join(
                extract_dir,
                "radnli-a-natural-language-inference-dataset-for-the-radiology-domain-1.0.0",
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "radnli_test_v1.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "radnli_dev_v1.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        with open(filepath, "r") as f:
            if self.config.schema == "source":
                for line in f:
                    json_line = json.loads(line)
                    yield json_line["pair_id"], json_line

            elif self.config.schema == "bigbio_te":
                for line in f:
                    json_line = json.loads(line)
                    entailment_example = {
                        "id": json_line["pair_id"],
                        "premise": json_line["sentence1"],
                        "hypothesis": json_line["sentence2"],
                        "label": json_line["gold_label"],
                    }
                    yield json_line["pair_id"], entailment_example


if __name__ == "__main__":
    dataset = datasets.load_dataset(
        __file__, data_dir="/local-scratch/nigam/projects/clinical_llm/data/radnli", name="radnli_bigbio_te"
    )
    print(dataset)
