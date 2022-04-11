import argparse
import torch
import datasets
from typing import List, Tuple
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_summarization_model_and_tokenizer() -> Tuple[
    BartForConditionalGeneration, BartTokenizer
]:
    """
    Load summary generation model and move to GPU, if possible.

    Returns:
        (model, tokenizer)
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-xsum")
    model.to(device)

    return model, tokenizer


def generate_summaries(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizer,
    docs_to_summarize: List[str],
    num_beams: int = 4,
):
    """
    Given a trained summary generation model and appropriate tokenizer,

    1. Tokenize text (and move to device, if possible)
    2. Run inference on model to generate output vocabulary tokens for summary
    3. Decode tokens to a sentence using the tokenizer

    Args:
        model: model to run inference on
        tokenizer: tokenizer corresponding to model
        docs_to_summarize: documents to summarize
        num_beams: number of beams for beam search

    Returns:
        List[decoded_sentence]
    """
    inputs = tokenizer(
        docs_to_summarize,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding=True,
    )
    input_token_ids = inputs.input_ids.to(device)

    model_output = model.generate(
        input_token_ids,
        num_beams=num_beams,
        max_length=150,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_summaries = [
        tokenizer.decode(
            id, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for id in model_output.sequences
    ]

    return generated_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run inference on xsum data using a pre-trained model"
    )

    parser.add_argument(
        "data_split",
        type=str,
        choices=["train", "test", "validation"],
        help="xsum data split to index into with `data_index`",
    )

    parser.add_argument(
        "n_docs",
        type=int,
        default=1,
        help="number of docs to summairze",
    )

    args = parser.parse_args()

    model, tokenizer = load_summarization_model_and_tokenizer()

    xsum_test = datasets.load_dataset("xsum")[args.data_split]
    selected_data = xsum_test[: args.n_docs]
    source_docs = selected_data["document"]
    gt_sums = selected_data["summary"]
    ids = selected_data["id"]

    summaries = generate_summaries(model, tokenizer, source_docs, num_beams=4)

    for id, gen_summary, gt_sum in zip(ids, summaries, gt_sums):
        print("XSUM ID", id)
        print("GROUND TRUTH SUMMARY:", gt_sum)
        print("PREDICTED SUMMARY:", gen_summary)
