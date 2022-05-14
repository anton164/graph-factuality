import os
import json
from sys import argv
from metrics import bertscore, rouge

if __name__ == "__main__":
    who = 'baseline'
    if len(argv) > 1:
        who = argv[1]
    scores = {}
    with open(f"./data/results/{who}-scores.json", "r") as f:
        scores = json.load(f)

    if len(scores) == 0:
        ref_files = os.listdir(r"results/refs")
        asgard_files = os.listdir(f"results/output-{who}")
        for ref_fn, asgard_fn in zip(ref_files, asgard_files):
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