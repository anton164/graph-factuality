import os
import json
from sys import argv
from metrics import bertscore, rouge
from tqdm import tqdm

if __name__ == "__main__":
    who = 'baseline'
    if len(argv) > 1:
        who = argv[1]
    scores = {}
    if os.path.exists(f"./data/results/{who}-scores.json"):
        with open(f"./data/results/{who}-scores.json", "r") as f:
            scores = json.load(f)

            scores = {k:v for k,v in scores.items() if k.endswith(".ref")}

    if len(scores) == 0:
        ref_files = set([x for x in os.listdir(r"results/refs") if x.endswith(".ref")])
        asgard_files = sorted([x for x in os.listdir(f"results/output-{who}") if x.endswith(".dec")])
        for asgard_fn in tqdm(asgard_files):
            ref_fn = asgard_fn.replace(".dec", ".ref")
            ref_path = os.path.join(r"results/refs", ref_fn)
            pred_path = os.path.join(f"results/output-{who}", asgard_fn)
            with open(ref_path, 'r') as f:
                gt = f.read()
            with open(pred_path, 'r') as f:
                pred = f.read()
            scores[ref_fn] = ({})
            scores[ref_fn].update(bertscore([pred], [gt]))
            scores[ref_fn].update(rouge([pred], [gt]))
        with open(f"./data/results/{who}-scores.json", "w") as f:
            json.dump(scores, f, indent=2)

    bertscore = sum([scores[i]['bertscore']['f1'] for i in scores])/len(scores)
    r1 = sum([scores[i]['rouge1']['f1'] for i in scores])/len(scores)
    r2 = sum([scores[i]['rouge2']['f1'] for i in scores])/len(scores)
    rl = sum([scores[i]['rougeL']['f1'] for i in scores])/len(scores)
    print(f'CNN/DM avg f1 ({who}) n={len(scores)}:')
    print(f'  bertscore: {bertscore}')
    print(f'  rouge1:    {r1}')
    print(f'  rouge2:    {r2}')
    print(f'  rougeL:    {rl}')