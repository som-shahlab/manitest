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

import json
import os
from typing import Dict, List, Tuple
import re

import datasets

from .bigbiohub import text_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks


_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = True
_CITATION = """\
@article{demner2016preparing,
  title={Preparing a collection of radiology examinations for distribution and retrieval},
  author={Demner-Fushman, Dina and Kohli, Marc D and Rosenman, Marc B and Shooshan, Sonya E and Rodriguez, Laritza and Antani, Sameer and Thoma, George R and McDonald, Clement J},
  journal={Journal of the American Medical Informatics Association},
  volume={23},
  number={2},
  pages={304--310},
  year={2016},
  publisher={Oxford University Press}
}
"""

_DATASETNAME = "openi"
_DISPLAYNAME = "OpenI"

_DESCRIPTION = """\
Open-i service of the National Library of Medicine enables search and retrieval of abstracts
and images (including charts, graphs, clinical images, etc.) from the open source literature,
and biomedical image collections. Searching may be done using text queries as well as query
images. Open-i provides access to over 3.7 million images from about 1.2 million PubMed Central®
articles; 7,470 chest x-rays with 3,955 radiology reports; 67,517 images from NLM History of
Medicine collection; and 2,064 orthopedic illustrations.
"""


_HOMEPAGE = "https://openi.nlm.nih.gov/"

_LICENSE = ""

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_CARDIOMEGALY_SYNSET = [
    "cardiomegaly",
    "cardiac enlargement",
    "enlarged heart",
    "enlargement heart",
    "heart size increased",
    "megalocardia",
]
_EDEMA_SYNSET = ["edema", "wet lungs"]
_CONSOLIDATION_SYNSET = ["consolidation"]
_PNEUMONIA_SYNSET = [
    "pneumonia",
    "inflammation lung",
    "pulmonary inflammation",
    "pneumoniae",
    "pneumonitides",
    "pneumonitis",
]
_ATELECTASIS_SYNSET = ["atelectasis", "atelectases", "collapsed lung", "collapse; pulmonary"]
_PNEUMOTHORAX_SYNSET = ["pneumothorax", "free air in the chest outside the lung", "pleural air collection"]
_PLEURAL_EFFUSION_SYNSET = ["pleural effusion", "fluid in the chest", "pleural cavity effusion"]

DISEASE_LIST = [
    _CARDIOMEGALY_SYNSET,
    _EDEMA_SYNSET,
    _CONSOLIDATION_SYNSET,
    _PNEUMONIA_SYNSET,
    _ATELECTASIS_SYNSET,
    _PNEUMOTHORAX_SYNSET,
    _PLEURAL_EFFUSION_SYNSET,
]


class OpenIDataset(datasets.GeneratorBasedBuilder):
    """openi"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="openi_source",
            version=SOURCE_VERSION,
            description="radnli source schema",
            schema="source",
            subset_id="radnli",
        ),
        BigBioConfig(
            name="openi_bigbio_tf",
            version=BIGBIO_VERSION,
            description="openi BigBio schema",
            schema="bigbio_tf",
            subset_id="openi",
        ),
    ]

    DEFAULT_CONFIG_NAME = "openi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "cardiomegaly": datasets.Value("string"),
                    "edema": datasets.Value("string"),
                    "consolidation": datasets.Value("string"),
                    "pneumonia": datasets.Value("string"),
                    "atelectasis": datasets.Value("string"),
                    "pneumothorax": datasets.Value("string"),
                    "pleural effusion": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_tf":
            features = text_features

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
                )
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "datapath": os.path.join(extract_dir, "ecgen-radiology"),
                    "split": "test",
                },
            )
        ]

    def _generate_examples(self, datapath, split: str) -> Tuple[int, Dict]:
        for i, filename in enumerate(os.listdir(datapath)):
            if not filename.endswith(".xml"):
                assert "Should only be xml files in the directory"
            else:
                xml_path = os.path.join(datapath, filename)
                with open(xml_path, "r") as f:
                    xml_text = f.read()
                    text = ""
                    try:
                        text += re.search(r"<AbstractText Label=\"FINDINGS\">(.*?)<\/AbstractText>", xml_text).group(1)
                    except:
                        pass
                    try:
                        if text != "":
                            text += " "
                        text += re.search(r"<AbstractText Label=\"IMPRESSION\">(.*?)<\/AbstractText>", xml_text).group(
                            1
                        )
                    except:
                        pass

                    mesh_text = re.search(r"<MeSH>(.*?)<\/MeSH>", xml_text, re.DOTALL).group(1)

                    label_list = []
                    for disease in DISEASE_LIST:
                        disease_regex = "|".join(disease)
                        if re.search(disease_regex, mesh_text, re.IGNORECASE):
                            if self.config.schema == "source":
                                label_list.append("yes")
                            else:
                                label_list.append(1)
                        else:
                            if self.config.schema == "source":
                                label_list.append("no")
                            else:
                                label_list.append(0)

                if self.config.schema == "source":
                    yield i, {
                        "cardiomegaly": label_list[0],
                        "edema": label_list[1],
                        "consolidation": label_list[2],
                        "pneumonia": label_list[3],
                        "atelectasis": label_list[4],
                        "pneumothorax": label_list[5],
                        "pleural effusion": label_list[6],
                        "text": text,
                    }

                elif self.config.schema == "bigbio_tf":
                    yield i, {"id": i, "document_id": i, "text": text, "labels": label_list}


if __name__ == "__main__":
    dataset = datasets.load_dataset(
        __file__,
        data_dir="/dataNAS/people/lblankem/clinical-llm/datasets/ecgen-radiology-20230211T022908Z-001.zip",
        name="openi_source",
    )
    print(dataset["test"][2])
    print(dataset)
