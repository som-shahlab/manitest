from typing import Optional, List
import evaluate

AVAILABLE_METRICS: List[str] = [
        "bertscore",
        "rouge",
        "bleu",
        "meteor",
        "bleurt",
        "google_bleu",
        "chrf", 
        "comet"
    ]

def process_metric(dict_: dict, metric_name: str) -> float:
    if metric_name == "bleu":
        return float(dict_["bleu"])

    elif metric_name == "rouge":
        return float(dict_["rougeL"])

    elif metric_name == "meteor":
        return float(dict_["meteor"])

    elif metric_name == "bertscore":
        return float(np.array(dict_["f1"]).mean())

    elif metric_name == "bleurt":
        return float(np.array(dict_["scores"]).mean())

    elif metric_name == "google_bleu":
        return float(dict_["google_bleu"])

    elif metric_name == "chrf":
        return float(dict_["score"])

    else:
        raise ValueError(f"Metric name '{metric_name}' not supported!")


def generation_metric(predictions, references, metric_list: List[str] = ["rouge", "bleu"]):
    selected_metrics = [m for m in metric_list if m in AVAILABLE_METRICS]
    loaded_metrics = {}
    for metric_name in selected_metrics:
    if metric_name == "bleurt":
        loaded_metrics[metric_name] = evaluate.load("bleurt", "bleurt-base-512")
    else:
        loaded_metrics[metric_name] = evaluate.load(metric_name)
    
    metrics = {}
    if not selected_metrics:
        print(
            f"No supported metrics found!\nMetrics provided: {metric_list}\nMetrics Supported: {AVAILABLE_METRICS}"
        )
        return metrics
    
    for metric_name, metric in loaded_metrics.items():
        if metric_name == "bertscore":
            score_dict = metric.compute(
                predictions=predictions,
                references=references,
                model_type="distilbert-base-uncased",
            )
        elif metric_name == "chrf":
            score_dict = metric.compute(
                predictions=predictions,
                references=references,
                word_order=2
            )
        else:
            score_dict = metric.compute(
                predictions=predictions, references=references
            )
        
        score = None
        if score_dict is not None:
            score = process_metric(dict_=score_dict, metric_name=metric_name)
        
        metrics[metric_name] = score
    return metrics

# def generation_metric(hypos, refs, metric: Optional[str] = "sentence_bleu"):
#     r"""Some basic metric function for generation. However, many generation tasks
#     has their own evaluation bash scripts.
#     Args:
#         hypos (:obj:`str`) : the generated sentence.
#         refs (:obj:`list(str)`) : the referenced (ground-truth) sentence.
#         metric (:obj:`str`, `optional`) : the type of metric option
#     Returns:
#         score (float): evaluate score
#     """
#     if metric == "sentence_bleu":
#         # a simple criterion to visualize the performance, not rigorous.
#         import nltk

#         try:
#             nltk_path = str(nltk.data.find("tokenizers/punkt"))
#             print(f"using nltk from: {nltk_path}")
#         except LookupError:
#             nltk.download("punkt")

#         from nltk.translate.bleu_score import sentence_bleu
#         from nltk.tokenize import word_tokenize
#         from nltk.translate.bleu_score import SmoothingFunction

#         smoothie = SmoothingFunction().method4  # a function for smooth
#         scores = []

#         for ref, hypo in zip(refs, hypos):
#             tokenized_rs = []
#             ref = ref.split("\n")
#             for r in ref:
#                 tokenized_rs.append(word_tokenize(r))
#             hypo = word_tokenize(hypo)
#             try:
#                 sc = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
#             except ValueError:  # TODO ZeroDivisionError
#                 print("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
#                 sc = 0.0
#             scores.append(sc)
#         score = sum(scores) / len(scores)
#         return score

#     elif metric == "rouge":

#         def rouge(refs, preds):
#             """
#             Returns `t5` style ROUGE scores. See the related implementation:
#             https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68
#             :param refs:
#                 A `list` of reference `strs`.
#             :param preds:
#                 A `list` of predicted `strs`.
#             """
#             rouge_types = ["rouge1", "rouge2", "rougeLsum"]
#             scorer = rouge_scorer.RougeScorer(rouge_types)
#             # Add newlines between sentences to correctly compute `rougeLsum`.

#             def _prepare_summary(summary):
#                 summary = summary.replace(" . ", ".\n")
#                 return summary

#             # Accumulate confidence intervals.
#             aggregator = scoring.BootstrapAggregator()
#             for ref, pred in zip(refs, preds):
#                 ref = _prepare_summary(ref)
#                 pred = _prepare_summary(pred)
#                 aggregator.add_scores(scorer.score(ref, pred))
#             result = aggregator.aggregate()
#             return {type: result[type].mid.fmeasure * 100 for type in rouge_types}

#         return rouge(refs, hypos)

#     elif metric == "meteor":
#         import nltk
#         from nltk.translate import meteor_score
#         from nltk.tokenize import word_tokenize

#         try:
#             nltk_path = str(nltk.data.find("tokenizers/punkt"))
#             print(f"Using nltk from: {nltk_path}")
#         except LookupError:
#             nltk.download("punkt")

#         scores = []

#         for ref, hypo in zip(refs, hypos):
#             tokenized_hypo = word_tokenize(hypo)
#             tokenized_refs = [word_tokenize(r) for r in ref.split("\n")]

#             try:
#                 sc = meteor_score.meteor_score(tokenized_refs, tokenized_hypo)
#             except ValueError:
#                 print("Math domain error in Meteor, set to 0.0. Generated sentence: {}".format(hypo))
#                 sc = 0.0

#             scores.append(sc)

#         score = sum(scores) / len(scores)
#         return score

#     else:
#         raise ValueError("'{}' is not a valid metric type.".format(metric))
