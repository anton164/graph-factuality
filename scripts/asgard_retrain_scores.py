import os
import json
from metrics import bertscore, rouge

if __name__ == "__main__":
    scores = {}
    ref_files = os.listdir(r"results/refs")
    asgard_files = os.listdir(r"results/output-us")
    for ref_fn, asgard_fn in zip(ref_files, asgard_files):
        ref_path = os.path.join(r"results/refs", ref_fn)
        pred_path = os.path.join(r"results/output-us", asgard_fn)
        with open(ref_path, 'r') as f:
            gt = f.read()
        with open(pred_path, 'r') as f:
            pred = f.read()
        scores[ref_fn] = ({})
        scores[ref_fn].update(bertscore([pred], [gt]))
        scores[ref_fn].update(rouge([pred], [gt]))
    with open("./data/results/us-scores.json", "w") as f:
        json.dump(scores, f, indent=2)