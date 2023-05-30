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
We introduce PubMedQA, a novel biomedical question answering (QA) dataset collected from PubMed abstracts. 
The task of PubMedQA is to answer research questions with yes/no/maybe (e.g.: Do preoperative statins reduce atrial 
fibrillation after coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA has 1k expert-annotated,
 61.2k unlabeled and 211.3k artificially generated QA instances. Each PubMedQA instance is composed of (1) a question 
 which is either an existing research article title or derived from one, (2) a context which is the corresponding 
 abstract without its conclusion, (3) a long answer, which is the conclusion of the abstract and, presumably, 
 answers the research question, and (4) a yes/no/maybe answer which summarizes the conclusion. PubMedQA is the 
 first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is 
 required to answer the questions. Our best performing model, multi-phase fine-tuning of BioBERT with long 
 answer bag-of-word statistics as additional supervision, achieves 68.1% accuracy, compared to single human 
 performance of 78.0% accuracy and majority-baseline of 55.2% accuracy, leaving much room for improvement.
  PubMedQA is publicly available at https://pubmedqa.github.io.

pubmedqa
├── /local-scratch/nigam/projects/clinical_llm/data/pubmedqa
"""

import json
import os
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from .bigbiohub import entailment_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['English']
_PUBMED = False
_LOCAL = True
_CITATION = """\
@inproceedings{jin-etal-2019-pubmedqa,
    title = "{P}ub{M}ed{QA}: A Dataset for Biomedical Research Question Answering",
    author = "Jin, Qiao  and
      Dhingra, Bhuwan  and
      Liu, Zhengping  and
      Cohen, William  and
      Lu, Xinghua",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1259",
    doi = "10.18653/v1/D19-1259",
    pages = "2567--2577",
    abstract = "We introduce PubMedQA, a novel biomedical question answering (QA) dataset collected from PubMed abstracts. The task of PubMedQA is to answer research questions with yes/no/maybe (e.g.: Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA has 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA instances. Each PubMedQA instance is composed of (1) a question which is either an existing research article title or derived from one, (2) a context which is the corresponding abstract without its conclusion, (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and (4) a yes/no/maybe answer which summarizes the conclusion. PubMedQA is the first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is required to answer the questions. Our best performing model, multi-phase fine-tuning of BioBERT with long answer bag-of-word statistics as additional supervision, achieves 68.1{\%} accuracy, compared to single human performance of 78.0{\%} accuracy and majority-baseline of 55.2{\%} accuracy, leaving much room for improvement. PubMedQA is publicly available at https://pubmedqa.github.io.",
}
"""

_DATASETNAME = "pubmedqa"
_DISPLAYNAME = "PUBMEDQA"

_DESCRIPTION = """\
We introduce PubMedQA, a novel biomedical question answering (QA) dataset collected from PubMed abstracts. 
The task of PubMedQA is to answer research questions with yes/no/maybe (e.g.: Do preoperative statins reduce atrial 
fibrillation after coronary artery bypass grafting?) using the corresponding abstracts. PubMedQA has 1k expert-annotated,
 61.2k unlabeled and 211.3k artificially generated QA instances. Each PubMedQA instance is composed of (1) a question 
 which is either an existing research article title or derived from one, (2) a context which is the corresponding 
 abstract without its conclusion, (3) a long answer, which is the conclusion of the abstract and, presumably, 
 answers the research question, and (4) a yes/no/maybe answer which summarizes the conclusion. PubMedQA is the 
 first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is 
 required to answer the questions. Our best performing model, multi-phase fine-tuning of BioBERT with long 
 answer bag-of-word statistics as additional supervision, achieves 68.1% accuracy, compared to single human 
 performance of 78.0% accuracy and majority-baseline of 55.2% accuracy, leaving much room for improvement.
  PubMedQA is publicly available at https://pubmedqa.github.io.
"""


_HOMEPAGE = "https://aclanthology.org/D19-1259/"

_LICENSE = 'PhysioNet Credentialed Health Data License'

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXT_PAIRS_CLASSIFICATION]

_SOURCE_VERSION = "1.0.1"
_BIGBIO_VERSION = "1.0.0"


def dump_jsonl(data, fpath):
    with open(fpath, "w") as outf:
        for d in data:
            print (json.dumps(d), file=outf)

def process_pubmedqa(file_path, fname):
    dname = "pubmedqa"
    print (dname, fname)
    if fname in ["train", "dev"]:
        data = json.load(open(f"{file_path}/pqal_fold0/{fname}_set.json"))
    elif fname == "test":
        data = json.load(open(f"{file_path}/{fname}_set.json"))
    else:
        assert False
    outs, lens = [], []
    for id in data:
        obj = data[id]
        context = " ".join([c.strip() for c in obj["CONTEXTS"] if c.strip()])
        question = obj["QUESTION"].strip()
        label = obj["final_decision"].strip()
        assert label in ["yes", "no", "maybe"]
        outs.append({"id": id, "sentence1": question, "sentence2": context, "label": label})
        lens.append(len(question) + len(context))
    print ("total", len(outs), "seqlen mean", int(np.mean(lens)), "median", int(np.median(lens)), "95th", int(np.percentile(lens, 95)), "max", np.max(lens))

    os.system(f"mkdir -p {file_path}/{dname}_hf")
    dump_jsonl(outs, f"{file_path}/{dname}_hf/{fname}.json")



class PubMedQADataset(datasets.GeneratorBasedBuilder):
    """Free-form multiple-choice OpenQA dataset covering three languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pubmedqa_source",
            version=SOURCE_VERSION,
            description="PubMedQA source schema",
            schema="source",
            subset_id="pubmedqa",
        ),
        BigBioConfig(
            name="pubmedqa_bigbio_pairs",
            version=BIGBIO_VERSION,
            description="PubMedQA BigBio schema",
            schema="bigbio_pairs",
            subset_id="pubmedqa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pubmedqa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            pass
        elif self.config.schema == "bigbio_pairs":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            #license=str(_LICENSE),
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            #TODO later on would want to go from raw file, but for now, importing processed files
            data_dir = os.path.join(
                self.config.data_dir,
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
                pass

            elif self.config.schema == "bigbio_pairs":
                for line in f:
                    json_line = json.loads(line)
                    pairs_features = {
                        "id": json_line["id"],
                        "document_id": json_line["id"],
                        "sentence1": json_line["sentence1"],
                        "sentence2": json_line["sentence2"],
                        "label": json_line["label"],
                    }
                    yield json_line["id"], pairs_features


if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__, 
                                    data_dir='/local-scratch/nigam/projects/clinical_llm/data/pubmedqa_hf',
                                    name='pubmedqa_bigbio_pairs')
    print(dataset)