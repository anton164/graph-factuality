from datasets import load_metric
import datasets

_bertscore_metric = load_metric("bertscore")
_rouge_metric = datasets.load_metric("rouge")


def bertscore(predictions: list, references: list):
    # There is an optional device argument which will load the model on cuda
    # if it's available, otherwise it defaults to cpu
    results = _bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
    )
    n = len(predictions)
    scores = {
        "bertscore": {
            "precision": sum(results["precision"]) / n,
            "recall": sum(results["recall"]) / n,
            "f1": sum(results["f1"]) / n,
        },
    }
    return scores


def rouge(predictions: list, references: list):
    results = _rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    # types: rouge1, rouge2, rougeL
    # Use mid for every score because it is the mean of all inputs
    # high gives the 95th %ile score and low gives the 5th %ile
    scores = {
        "rouge1": {
            "precision": results["rouge1"].mid.precision,
            "recall": results["rouge1"].mid.recall,
            "f1": results["rouge1"].mid.fmeasure,
        },
        "rouge2": {
            "precision": results["rouge2"].mid.precision,
            "recall": results["rouge2"].mid.recall,
            "f1": results["rouge2"].mid.fmeasure,
        },
        "rougeL": {
            "precision": results["rougeL"].mid.precision,
            "recall": results["rougeL"].mid.recall,
            "f1": results["rougeL"].mid.fmeasure,
        },
    }
    return scores


if __name__ == "__main__":
    p1 = [
        "The impact of flooding in Dumfries and Galloway and the Borders is continuing to be felt.",
        "Two tour buses have been destroyed in a suspected arson attack in Londonderry.",
    ]
    r1 = [
        "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
        "Two tourist buses have been destroyed by fire in a suspected arson attack in Belfast city centre.",
    ]

    print("Bertscore:", bertscore(p1, r1))
    print("Rouge:", rouge(p1, r1))
