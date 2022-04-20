import argparse
from sumtool.generate_xsum_summary import (
    load_summarization_model_and_tokenizer,
    generate_summaries,
)
from datasets import load_dataset
from sumtool.storage import store_model_summaries
from metrics import bertscore, rouge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate and store xsum test summaries using the baseline model"
    )

    model, tokenizer = load_summarization_model_and_tokenizer()

    xsum_data = load_dataset("xsum")["test"]
    total = len(xsum_data)
    print(str(total) + " total documents to summarize")
    i = 0
    for x in xsum_data:
        i += 1
        gen_summary = generate_summaries(model, tokenizer, x["document"])
        gt_summary = x["summary"]

        store_model_summaries(
            "xsum",
            model.config.name_or_path,
            model.config.to_dict(),
            {x["id"]: gen_summary},
        )
        if i % 10 == 0:
            print(str(i) + " of " + str(total) + " summaries completed.")
