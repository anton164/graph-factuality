import json
from metrics import bertscore, rouge
from datasets import load_dataset

if __name__ == "__main__":
    xsum_data = load_dataset("xsum")["test"]
    scores = {}
    with open("./data/xsum/facebook-bart-large-xsum-summaries.json") as f:
        predictions = json.load(f)
    for x in xsum_data:
        i = x["id"]
        pred = [predictions[i]["summary"]]
        gt = [x["summary"]]
        scores[i] = {}
        scores[i].update(bertscore(pred, gt))
        scores[i].update(rouge(pred, gt))
    with open("./data/xsum/facebook-bart-large-xsum-scores.json", "w") as f:
        json.dump(scores, f, indent=2)
