import benepar
import nltk
from feqa import FEQA
import os
import numpy as np
import argparse


def batched_feqa(scorer, document_dir, summary_dir, batch_size = 100):
    """
    Batched FEQA on a directory of documents and summaries.
    params:
        scorer: FEQA object
        document_dir: directory of documents
        summary_dir: directory of summaries
        batch_size: number of documents to process at a time
    """
    all_f1_scores = []

    document_files = [f for f in os.listdir(document_dir)]
    summary_files = [f for f in os.listdir(summary_dir)]


    assert len(document_files) == len(summary_files), "Number of documents and summaries must match"

    for i in range(0, len(document_files), batch_size):
        print(f"Processing batch {i}", end="\r")
        documents = [open(document_dir + f).read() for f in document_files[i: min(len(document_dir), i + batch_size)]]
        summaries = [open(summary_dir + f).read() for f in summary_files[i:min(len(document_dir), i + batch_size)]]

        f1_scores = scorer.compute_score(documents, summaries, aggregate=False)
        all_f1_scores.extend(f1_scores)

    return all_f1_scores, np.mean(all_f1_scores)



if __name__ == "__main__":
    benepar.download('benepar_en2')
    nltk.download('stopwords')
    nltk.download('punkt')

    scorer = FEQA(use_gpu=False)

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--document_dir",
        default=None,
        type=str,
        required=True,
        help="Directory with original documents"
    )

    parser.add_argument(
        "--summary_dir",
        default=None,
        type=str,
        required=True,
        help="Directory with summaries"
    )

    parser.add_argument(
        "--output_file",
        default="output.txt",
        type=str,
        required=False,
        help="Output file"
    )

    args = parser.parse_args()

    if args.document_dir is None:
        print("--document_dir argument is required")
    
    if args.summary_dir is None:
        print("--summary_dir argument is required")

    with open(args.output_file, "w") as f:
        f.write(f"FEQA on {args.document_dir} and {args.summary_dir}\n")
        feqa_scores, mean_f1 = batched_feqa(scorer, args.document_dir, args.summary_dir)

        f.write("mean: " + str(mean_f1) + "\n")
        f.write("all: " + str(feqa_scores) + "\n")

