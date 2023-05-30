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
import pandas as pd

from .bigbiohub import text2text_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks


_LANGUAGES = ["English"]
_PUBMED = False
_LOCAL = True
_CITATION = """
@article{gao2023bionlp,
    title={BioNLP Workshop 2023 Shared Task 1A: Problem List Summarization},
    author={Yan Gao and Thomas Miller and Mohammad Afshar and David Dligach},
    journal={PhysioNet},
    year={2023},
    doi={10.13026/1z6g-ex18}
}

@inproceedings{gao2022summarizing,
  title={Summarizing Patients' Problems from Hospital Progress Notes Using Pre-trained Sequence-to-Sequence Models},
  author={Gao, Yu and Dligach, Dmitry and Miller, Timothy and Xu, Di and Churpek, Mary M and Afshar, Mahdieh},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  pages={2979--2991},
  year={2022},
  month={October},
}

@article{goldberger2000physiobank,
    title={PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals},
    author={A. L. Goldberger and L. A. Amaral and L. Glass and J. M. Hausdorff and P. C. Ivanov and R. C. Mark and J. M. Mietus and G. B. Moody and C. K. Peng and H. E. Stanley},
    journal={Circulation},
    volume={101},
    number={23},
    pages={e215--e220},
    year={2000},
    publisher={Lippincott Williams \& Wilkins}
}
"""

_DATASETNAME = "bio_nlp_2023_st_1a"
_DISPLAYNAME = "bio_nlp_2023_st_1a"

_DESCRIPTION = """
Problem list summarization.
"""

_HOMEPAGE = "https://www.physionet.org/content/bionlp-workshop-2023-task-1a/1.0.0/"

_LICENSE = ""

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioNLP2023_ST_1A_Dataset(datasets.GeneratorBasedBuilder):
    """openi"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_2023_st_1a_source",
            version=SOURCE_VERSION,
            description="radnli source schema",
            schema="source",
            subset_id="bionlp_2023_st_1a",
        ),
        BigBioConfig(
            name="bionlp_2023_st_1a_bigbio_tf",
            version=BIGBIO_VERSION,
            description="bionlp_2023_st_1a BigBio schema",
            schema="bigbio_tf",
            subset_id="bionlp_2023_st_1a",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_2023_st_1a_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features({"text_1": datasets.Value("string"), "text_2": datasets.Value("string")})

        elif self.config.schema == "bigbio_tf":
            features = text2text_features

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
                    "datapath": os.path.join(
                        extract_dir, "bionlp-workshop-2023-shared-task-1a-problem-list-summarization-1.0.0"
                    ),
                    "split": "test",
                },
            )
        ]

    def _generate_examples(self, datapath, split: str) -> Tuple[int, Dict]:
        """Generate examples."""
        data = pd.read_csv(os.path.join(datapath, "BioNLP2023-1A-Train.csv"))
        data.dropna(inplace=True)

        print(data)

        src_ids = []
        src_masks = []
        tgt_ids = []
        tgt_masks = []

        fids = data["File ID"].tolist()
        srcs = data["Assessment"].tolist()
        tgts = data["Summary"].tolist()
        # sos = data["SO"].tolist() # Added August 21: adding Subjective and Objective sections
        ss = data["Subjective Sections"].tolist()
        ob = data["Objective Sections"].tolist()

        i = 0

        for fid, src, gts, s, o in zip(fids, srcs, tgts, ss, ob):
            tgt = re.sub("\W+", " ", gts)
            # noteid = self.fids[idx]
            # gts = self.tgts[idx]
            # tgt = re.sub('\W+',' ',gts)
            # if self.summ_type == "All":
            """
            input_str = self.prefix + " <ASSESSMENT> "+ src + " <SUBJECTIVE> "+ s +" <OBJECTIVE> " + o
            elif self.summ_type == "S+A":
                input_str = self.prefix + " <ASSESSMENT> "+ src + " <SUBJECTIVE> "+ s
            else:
                input_str = self.prefix + " <ASSESSMENT> "+ src
            """

            input_str = "<ASSESSMENT> " + src + " <SUBJECTIVE> " + s + " <OBJECTIVE> " + o

            if self.config.schema == "source":
                yield i, {"text_1": input_str, "text_2": tgt}

            elif self.config.schema == "bigbio_tf":
                yield i, {
                    "id": i,
                    "document_id": fid,
                    "text_1": input_str,
                    "text_2": tgt,
                    "text_1_name": "note",
                    "text_2_name": "summary",
                }

            i += 1


"""
if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__,
                                    data_dir='/dataNAS/people/lblankem/clinical-llm/datasets/bionlp-workshop-2023-shared-task-1a-problem-list-summarization-1.0.0.zip',
                                    name='bionlp_2023_st_1a_source')
    print(dataset['test'][2])
    print(dataset)
"""
