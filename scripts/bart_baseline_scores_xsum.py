import json
from metrics import bertscore, rouge
from datasets import load_dataset

if __name__ == "__main__":
    scores = {}
    with open("./data/xsum/facebook-bart-large-xsum-scores.json", "r") as f:
        scores = json.load(f)

    if len(scores) == 0:
        xsum_data = load_dataset("xsum")["test"]
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

    bertscore = sum([scores[i]['bertscore']['f1'] for i in scores])/len(scores)
    r1 = sum([scores[i]['rouge1']['f1'] for i in scores])/len(scores)
    r2 = sum([scores[i]['rouge2']['f1'] for i in scores])/len(scores)
    rl = sum([scores[i]['rougeL']['f1'] for i in scores])/len(scores)
    print(f'XSUM avg f1 n={len(scores)}:')
    print(f'  bertscore: {bertscore}')
    print(f'  rouge1:    {r1}')
    print(f'  rouge2:    {r2}')
    print(f'  rougeL:    {rl}')
