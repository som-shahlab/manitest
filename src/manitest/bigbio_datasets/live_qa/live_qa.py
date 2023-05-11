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
if `data_dir` is `"liveqa"` it should contain the following files:
liveqa
├── /local-scratch/nigam/projects/live_llm/data/liveqa/LiveQA_MedicalTask_TREC2017-master.zip
"""

import xmltodict
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

_DATASETNAME = "liveqa"
_DISPLAYNAME = "liveQA"

_DESCRIPTION = """\
"""


_HOMEPAGE = "https://physionet.org/content/liveqa-report-inference/1.0.0/"

_LICENSE = "PHYSIONET_LICENSE_1p5"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.TEXTUAL_ENTAILMENT]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class liveqaDataset(datasets.GeneratorBasedBuilder):
    """liveqa"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="liveqa_source",
            version=SOURCE_VERSION,
            description="liveqa source schema",
            schema="source",
            subset_id="liveqa",
        ),
        BigBioConfig(
            name="liveqa_bigbio_qa",
            version=BIGBIO_VERSION,
            description="liveqa BigBio schema",
            schema="bigbio_qa",
            subset_id="liveqa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "liveqa_source"

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
                    "LiveQA_MedicalTask_TREC2017-master.zip",
                )
            )
            data_dir = os.path.join(
                extract_dir,
                "LiveQA_MedicalTask_TREC2017-master",
            )

        return [
            
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "TestDataset/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"),
                    "split": "test",
                },
            ),
            
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "TrainingDatasets"),
                    "split": "train",
                },
            ),
            
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        
        if self.config.schema == "source":
            with open(filepath, "r") as f:
                for line in f:
                    json_line = json.loads(line)
                    yield json_line["title"], json_line

        elif self.config.schema == "bigbio_qa":
            if split == 'train':
                for i, filename in enumerate(os.listdir(filepath)):

                    # 1.xml
                    if filename.endswith("1.xml"):
                        xml_path  = os.path.join(filepath, filename)
                        with open(xml_path, 'r') as myfile:
                            obj = xmltodict.parse(myfile.read())
                            for qa in obj['NLM-QUESTIONS']['NLM-QUESTION']:
                                try: 
                                    #train1
                                    qs_id = qa['@questionid']
                                except:
                                    #train2
                                    qs_id = qa['@qid']
                                subqs = qa['SUB-QUESTIONS']['SUB-QUESTION']
                                #multiple subqs
                                if type(subqs) == list:
                                    for i, subq in enumerate(subqs):
                                        qs_id = subq['@subqid']
                                        ans = subq['ANSWERS']
                                        # multiple subqs + multiple ans
                                        if type(subq['ANSWERS']['ANSWER']) == list:
                                            anss = subq['ANSWERS']['ANSWER']
                                            for ans in anss:
                                                yield ans['@answerid'], {
                                                "id": ans['@answerid'],
                                                "question_id": ans['@answerid'],
                                                "document_id": filename,
                                                "question": qa['MESSAGE'],
                                                "type": 'NULL',
                                                "choices": [],
                                                "context": [],
                                                "answer": [ans['#text']]
                                                #"answer": [qa['SUB-QUESTIONS']['SUB-QUESTION']['ANSWERS']['ANSWER'][0]['#text']],
                                                }
                                        # multiple subqs + single ans
                                        else:
                                            ans = subq['ANSWERS']['ANSWER']
                                            yield ans['@answerid'], {
                                            "id": ans['@answerid'],
                                            "question_id": ans['@answerid'],
                                            "document_id": filename,
                                            "question": qa['MESSAGE'],
                                            "type": 'NULL',
                                            "choices": [],
                                            "context": [],
                                            "answer": [ans['#text']]
                                            #"answer": [qa['SUB-QUESTIONS']['SUB-QUESTION']['ANSWERS']['ANSWER'][0]['#text']],
                                            }
                                #single subq
                                else:
                                    # single subq + multiple ans
                                    if type(subqs['ANSWERS']['ANSWER']) == list:
                                        anss = subqs['ANSWERS']['ANSWER']
                                        for ans in anss:
                                            yield ans['@answerid'], {
                                            "id": ans['@answerid'],
                                            "question_id": ans['@answerid'],
                                            "document_id": filename,
                                            "question": qa['MESSAGE'],
                                            "type": 'NULL',
                                            "choices": [],
                                            "context": [],
                                            "answer": [ans['#text']]
                                            #"answer": [qa['SUB-QUESTIONS']['SUB-QUESTION']['ANSWERS']['ANSWER'][0]['#text']],
                                            }
                                    #single subq + single ans
                                    else:
                                        ans = subqs['ANSWERS']['ANSWER']
                                        yield ans['@answerid'], {
                                            "id": ans['@answerid'],
                                            "question_id": ans['@answerid'],
                                            "document_id": filename,
                                            "question": qa['MESSAGE'],
                                            "type": 'NULL',
                                            "choices": [],
                                            "context": [],
                                            "answer": [ans['#text']]
                                            #"answer": [qa['SUB-QUESTIONS']['SUB-QUESTION']['ANSWERS']['ANSWER'][0]['#text']],
                                            }
                    # 2.xml
                    else:            
                        xml_path  = os.path.join(filepath, filename) 
                        with open(xml_path, 'r') as myfile:
                            obj = xmltodict.parse(myfile.read())
                            for qa in obj['NLM-QUESTIONS']['NLM-QUESTION']:
                                try: 
                                    #train1
                                    qs_id = qa['@questionid']
                                except:
                                    #train2
                                    qs_id = qa['@qid']
                                subqs = qa['SUB-QUESTIONS']['SUB-QUESTION']
                                yield qs_id, {
                                            "id": qs_id,
                                            "question_id": qs_id,
                                            "document_id": filename,
                                            "question": qa['MESSAGE'],
                                            "type": 'NULL',
                                            "choices": [],
                                            "context": [],
                                            "answer": [subqs['ANSWERS']['ANSWER']]
                                            #"answer": [qa['SUB-QUESTIONS']['SUB-QUESTION']['ANSWERS']['ANSWER'][0]['#text']],
                                            }

                                
            elif split == 'test':
                xml_path = filepath
                with open(xml_path, 'r') as myfile:
                    obj = xmltodict.parse(myfile.read())
                    #print(qa['NLM-Summary'])
                    #print(qa['ReferenceAnswers']['RefAnswer'][0]['ANSWER'])
                    for qa in obj['LiveQA2017-Medical-Test-Set-Full']['NLM-QUESTION']:
                        #print(qa['NLM-Summary'])
                        #print(qa['ReferenceAnswers']['RefAnswer'][0]['ANSWER'])
                        #print('='*50, qa['ANNOTATIONS'], '='*50)
                        yield qa['@qid'], {
                        "id": qa['@qid'],
                        "question_id": qa['@qid'],
                        "document_id": xml_path,
                        "question": qa['NLM-Summary'],
                        "type": qa['ANNOTATIONS']['TYPE'],
                        "choices": [],
                        "context": [],
                        "answer": [qa['ReferenceAnswers']]
                        #"answer": [qa['ReferenceAnswers']['RefAnswer'][0]['ANSWER']],
                        }

if __name__ == "__main__":
    dataset = datasets.load_dataset(__file__, 
                                    data_dir='/local-scratch/nigam/projects/clinical_llm/data/live_.qa',
                                    name='liveqa_bigbio_qa')
    print(dataset)