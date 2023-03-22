from abc import abstractmethod
from enum import Enum

class TaskType(Enum):
  BINARY_CLASSIFICATION = 0
  MULTICLASS_CLASSIFICATION = 0
  MULTILABEL_CLASSIFICATION = 1
  GENERATION = 2

class Task:
  name: str
  task_type: TaskType

  def __init__(self):
    self.prompts: List[Prompt] = []

class Prompt:
  name: str
  
  @abstractmethod
  def generate_prompt(self, example: dict) -> str:
    """Takes a dataset example and returns a prompted version of that example."""
    return ''

  @abstractmethod
  def get_label(self, example: dict) -> str:
    """Gets the ground truth label for a dataset example"""
    return ''

  @abstractmethod
  def generate_llm_queries(self, example: dict) -> dict:
    """Generate queries to feed into the LLM for evaluating the given dataset example.
    Each key in the returned dict is a label, and each label contains a list of queries."""
    return {}
  
  def generate_llm_queries_using_verbalizer(self, example: dict, verbalizer: dict) -> dict:
    """Use a simple dictionary as a verbalizer to generate LLM queries for a prompt"""
    prompt: str = self.generate_prompt(example)
    return {
      label : [
        f"{prompt} {term}"
        for term in terms
      ]
      for label, terms in verbalizer.items()
    }

class MedNLIPrompt(Prompt):
  verbalizer: dict = {
    'entailment' : [ 'yes' ],
    'not entailment' : [ 'no ' ],
  }

  def generate_llm_queries(self, example: dict) -> dict:
    return super().generate_llm_queries_using_verbalizer(example, self.verbalizer)
  
  def get_label(self, example: dict):
    """Maps attributes of a dataset example to a class (i.e. a [key] in `verbalizer`)"""
    if example['label'] in [ 'entailment' ]:
      return 'entailment'
    elif example['label'] in [ 'neutral', 'contradiction' ]:
      return 'not entailment'
    else:
      raise ValueError(f"Unknown label {example['label']}")

class Prompt1(MedNLIPrompt):
  name: str = 'suppose'
  verbalizer: dict = {
    'entailment' : [ 'entailment' ],
    'not entailment' : [ 'neutral ' ],
  }
  def generate_prompt(self, example: dict) -> str:
    return f"Suppose {example['premise']} Can we infer that {example['hypothesis']}?"

class Prompt2(MedNLIPrompt):
  name: str = 'two_sentences'
  def generate_prompt(self, example: dict) -> str:
    return f"Sentence 1: {example['premise']}\n\nSentence 2: {example['hypothesis']}\n\nQuestion: Does Sentence 1 entail Sentence 2?  yes or no"

class Prompt3(MedNLIPrompt):
  name: str = 'does_it_follow'
  def generate_prompt(self, example: dict) -> str:
    return f"Given that {example['premise']} Does it follow that {example['hypothesis']}  yes or no"

class Prompt4(MedNLIPrompt):
  name: str = 'licensed_to_say'
  verbalizer: dict = {
    'entailment' : [ 'true' ],
    'not entailment' : [ 'false ' ],
  }
  def generate_prompt(self, example: dict) -> str:
    return f"{example['premise']} Therefore, we are licensed to say that {example['hypothesis']}  true or false"

class Prompt5(MedNLIPrompt):
  name: str = 'does_passage_support_claim'
  def generate_prompt(self, example: dict) -> str:
    return f"{example['premise']} Does the previous passage support the claim that {example['hypothesis']}?"

class MedNLI(Task):
  name: str = 'mednli'
  task_type: TaskType = TaskType.BINARY_CLASSIFICATION

  def __init__(self):
    self.prompts = [ Prompt1, Prompt2, Prompt3, Prompt4, Prompt5, ]

  def load_dataset(self):
    return load_dataset('bigbio/mednli', name='mednli_bigbio_te')


