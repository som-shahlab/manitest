from typing import Optional
from rouge_score import rouge_scorer, scoring


def generation_metric(hypos, refs, metric: Optional[str] = "sentence_bleu"):
    r"""Some basic metric function for generation. However, many generation tasks
    has their own evaluation bash scripts.
    Args:
        hypos (:obj:`str`) : the generated sentence.
        refs (:obj:`list(str)`) : the referenced (ground-truth) sentence.
        metric (:obj:`str`, `optional`) : the type of metric option
    Returns:
        score (float): evaluate score
    """
    if metric == "sentence_bleu":
        # a simple criterion to visualize the performance, not rigorous.
        import nltk

        try:
            nltk_path = str(nltk.data.find("tokenizers/punkt"))
            print(f"using nltk from: {nltk_path}")
        except LookupError:
            nltk.download("punkt")

        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import SmoothingFunction

        smoothie = SmoothingFunction().method4  # a function for smooth
        scores = []

        for ref, hypo in zip(refs, hypos):
            tokenized_rs = []
            ref = ref.split("\n")
            for r in ref:
                tokenized_rs.append(word_tokenize(r))
            hypo = word_tokenize(hypo)
            try:
                sc = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
            except ValueError:  # TODO ZeroDivisionError
                print("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
                sc = 0.0
            scores.append(sc)
        score = sum(scores) / len(scores)
        return score

    elif metric == "rouge":

        def rouge(refs, preds):
            """
            Returns `t5` style ROUGE scores. See the related implementation:
            https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68
            :param refs:
                A `list` of reference `strs`.
            :param preds:
                A `list` of predicted `strs`.
            """
            rouge_types = ["rouge1", "rouge2", "rougeLsum"]
            scorer = rouge_scorer.RougeScorer(rouge_types)
            # Add newlines between sentences to correctly compute `rougeLsum`.

            def _prepare_summary(summary):
                summary = summary.replace(" . ", ".\n")
                return summary

            # Accumulate confidence intervals.
            aggregator = scoring.BootstrapAggregator()
            for ref, pred in zip(refs, preds):
                ref = _prepare_summary(ref)
                pred = _prepare_summary(pred)
                aggregator.add_scores(scorer.score(ref, pred))
            result = aggregator.aggregate()
            return {type: result[type].mid.fmeasure * 100 for type in rouge_types}

        return rouge(refs, hypos)

    else:
        raise ValueError("'{}' is not a valid metric type.".format(metric))
