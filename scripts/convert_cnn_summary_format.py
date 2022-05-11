from sumtool.storage import get_summaries
import os

summaries = get_summaries("cnn_dailymail", "patrickvonplaten-roberta2roberta-cnn_dailymail-fp16")

ref_files = [x for x in os.listdir(r"results/refs") if x.endswith("-id")]
id_to_idx = {}
for id_fn in ref_files:
    idx = id_fn.replace("-id", "")
    id_path = os.path.join(r"results/refs", id_fn)
    with open(id_path, 'r') as f:
        sum_id = f.read()
        id_to_idx[sum_id] = idx


for sum_id, data in summaries.items():
    with open(f"results/output-baseline/{id_to_idx[sum_id]}.dec", "w") as f:
        f.write(data["summary"])

